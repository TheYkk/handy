//! Remote (cloud or self-hosted) speech-to-text.
//!
//! Handy transcribes locally by default. When remote transcription is enabled,
//! the recorded PCM is encoded as a 16 kHz mono WAV and posted to an
//! OpenAI-compatible `/audio/transcriptions` endpoint instead. That one dialect
//! covers OpenRouter (and therefore models such as
//! `microsoft/mai-transcribe-2`), OpenAI, Groq, DeepInfra, and the self-hosted
//! servers that copy the same route (whisper.cpp's `server`, speaches,
//! faster-whisper-server, vLLM).
//!
//! Two transports exist for the same job:
//! * `multipart/form-data` — what every one of those servers accepts, used here.
//! * base64 JSON (`input_audio`) — OpenRouter-specific, needed only for uploads
//!   above the 25 MB multipart cap, which a dictation-length clip never hits.
//!
//! ElevenLabs Scribe is the one non-OpenAI dialect: `POST /speech-to-text`
//! with an `xi-api-key` header, `model_id`/`language_code` fields, and a
//! `keyterms` list instead of a free-form prompt. It also accepts raw 16 kHz
//! PCM (`file_format=pcm_s16le_16`), which skips server-side decoding.

use std::io::Cursor;
use std::time::Duration;

use anyhow::{anyhow, Result};
use log::debug;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, REFERER, USER_AGENT};

use crate::settings::{AppSettings, RemoteTranscriptionApiFormat};

/// Scribe models offered in the picker. ElevenLabs' `/models` lists only
/// text-to-speech models, and scoped API keys often can't read it at all.
const ELEVENLABS_STT_MODELS: &[&str] = &["scribe_v2", "scribe_v1"];

/// ElevenLabs rejects the whole request if any keyterm breaks these limits,
/// so terms are filtered before sending rather than failing the recording.
const ELEVENLABS_MAX_KEYTERMS: usize = 1000;
const ELEVENLABS_MAX_KEYTERM_CHARS: usize = 50;
const ELEVENLABS_MAX_KEYTERM_WORDS: usize = 5;
const ELEVENLABS_KEYTERM_FORBIDDEN: &[char] = &['<', '>', '{', '}', '[', ']', '\\'];

/// A fully-resolved remote endpoint: everything needed to make one request,
/// with no further settings lookups (and therefore no surprises if settings
/// change mid-transcription).
#[derive(Debug, Clone)]
pub struct RemoteSttConfig {
    pub provider_label: String,
    pub api_format: RemoteTranscriptionApiFormat,
    /// API root without the `/audio/transcriptions` suffix.
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// `None` asks the provider to auto-detect.
    pub language: Option<String>,
    /// Vocabulary hint sent as OpenAI's `prompt`. Providers that ignore the
    /// field (OpenRouter) simply drop it.
    pub prompt: Option<String>,
    /// Vocabulary hint as discrete terms, for ElevenLabs' `keyterms`.
    pub keyterms: Vec<String>,
    /// Use `/audio/translations` (English output) instead of transcription.
    pub translate: bool,
    pub timeout: Duration,
}

impl RemoteSttConfig {
    fn url(&self) -> String {
        let base = self.base_url.trim().trim_end_matches('/');
        if self.api_format == RemoteTranscriptionApiFormat::ElevenLabs {
            return format!("{}/speech-to-text", base);
        }
        if self.translate {
            format!("{}/audio/translations", base)
        } else {
            format!("{}/audio/transcriptions", base)
        }
    }
}

/// Whether remote transcription should handle the next recording at all.
pub fn is_enabled(settings: &AppSettings) -> bool {
    settings.remote_transcription_enabled
}

/// Build a request config from settings, or explain why it can't be built.
///
/// The error strings are user-facing: they are what the settings UI shows when
/// "Test connection" fails and what the fallback path logs.
pub fn resolve_config(settings: &AppSettings) -> Result<RemoteSttConfig, String> {
    resolve_config_for_provider(settings, &settings.remote_transcription_provider_id)
}

pub fn resolve_config_for_provider(
    settings: &AppSettings,
    provider_id: &str,
) -> Result<RemoteSttConfig, String> {
    let provider = settings
        .remote_transcription_provider(provider_id)
        .ok_or_else(|| format!("Remote transcription provider '{}' not found", provider_id))?;

    let base_url = provider.base_url.trim().to_string();
    if base_url.is_empty() {
        return Err(format!("{} has no base URL configured", provider.label));
    }

    let api_key = settings.remote_transcription_api_key(provider_id);
    if provider.requires_api_key && api_key.trim().is_empty() {
        return Err(format!("An API key is required for {}", provider.label));
    }

    let model = settings.remote_transcription_model(provider_id);
    if model.trim().is_empty() {
        return Err(format!("No remote model selected for {}", provider.label));
    }

    // "auto" is Handy's own sentinel for language detection; the OpenAI dialect
    // expresses that by omitting `language` entirely.
    let language = match settings.selected_language.trim() {
        "" | "auto" => None,
        code => Some(code.to_string()),
    };

    // Custom words double as a decode-time vocabulary hint, the same role the
    // local whisper path uses them for. Fuzzy post-correction still runs on the
    // returned text, so a provider that ignores `prompt` loses nothing.
    let mut hints: Vec<String> = Vec::new();
    let configured = settings.remote_transcription_prompt.trim();
    if !configured.is_empty() {
        hints.push(configured.to_string());
    }
    let custom_words: Vec<String> = settings
        .custom_words
        .iter()
        .map(|w| w.trim().to_string())
        .filter(|w| !w.is_empty())
        .collect();
    if !custom_words.is_empty() {
        hints.push(custom_words.join(", "));
    }
    let prompt = if hints.is_empty() {
        None
    } else {
        Some(hints.join(" "))
    };

    // ElevenLabs takes terms, not a sentence: split the hint on commas (the
    // format the settings placeholder suggests) and add each custom word.
    // Keyterms carry a per-request surcharge, so nothing is sent unless the
    // user configured some vocabulary.
    let keyterms = if provider.api_format == RemoteTranscriptionApiFormat::ElevenLabs {
        elevenlabs_keyterms(
            configured
                .split(',')
                .chain(custom_words.iter().map(String::as_str)),
        )
    } else {
        Vec::new()
    };

    let timeout_seconds = if settings.remote_transcription_timeout_seconds == 0 {
        30
    } else {
        settings.remote_transcription_timeout_seconds
    };

    Ok(RemoteSttConfig {
        provider_label: provider.label.clone(),
        api_format: provider.api_format,
        base_url,
        api_key,
        model: model.trim().to_string(),
        language,
        prompt,
        keyterms,
        // Only honored where the provider actually implements the separate
        // translations route; elsewhere the request would 404.
        translate: settings.translate_to_english && provider.supports_translations,
        timeout: Duration::from_secs(timeout_seconds),
    })
}

/// Keep only terms ElevenLabs will accept, deduplicated case-insensitively.
fn elevenlabs_keyterms<'a>(candidates: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut terms = Vec::new();
    for candidate in candidates {
        let term = candidate.trim();
        if term.is_empty()
            || term.chars().count() >= ELEVENLABS_MAX_KEYTERM_CHARS
            || term.split_whitespace().count() > ELEVENLABS_MAX_KEYTERM_WORDS
            || term.contains(ELEVENLABS_KEYTERM_FORBIDDEN)
        {
            continue;
        }
        if seen.insert(term.to_lowercase()) {
            terms.push(term.to_string());
        }
        if terms.len() == ELEVENLABS_MAX_KEYTERMS {
            break;
        }
    }
    terms
}

fn auth_headers(api_key: &str, api_format: RemoteTranscriptionApiFormat) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(
        REFERER,
        HeaderValue::from_static("https://github.com/cjpais/Handy"),
    );
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("Handy/1.0 (+https://github.com/cjpais/Handy)"),
    );
    headers.insert("X-Title", HeaderValue::from_static("Handy"));

    if !api_key.trim().is_empty() {
        let (name, value) = match api_format {
            RemoteTranscriptionApiFormat::OpenAi => {
                (AUTHORIZATION, format!("Bearer {}", api_key.trim()))
            }
            RemoteTranscriptionApiFormat::ElevenLabs => (
                HeaderName::from_static("xi-api-key"),
                api_key.trim().to_string(),
            ),
        };
        let mut value =
            HeaderValue::from_str(&value).map_err(|e| anyhow!("Invalid API key value: {}", e))?;
        value.set_sensitive(true);
        headers.insert(name, value);
    }
    Ok(headers)
}

fn f32_to_i16(sample: f32) -> i16 {
    // Clamp before casting: an out-of-range f32 would wrap instead of
    // saturating and turn a loud frame into noise.
    (sample * 32767.0).clamp(-32768.0, 32767.0) as i16
}

/// Encode 16 kHz mono f32 PCM as headerless 16-bit little-endian PCM, the
/// `pcm_s16le_16` upload ElevenLabs accepts without server-side decoding.
pub fn encode_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&sample| f32_to_i16(sample).to_le_bytes())
        .collect()
}

/// Encode 16 kHz mono f32 PCM as a 16-bit WAV in memory.
///
/// The whole audio pipeline hands out 16 kHz mono f32, which is also what every
/// STT provider wants, so no resampling is involved.
pub fn encode_wav_16k_mono(samples: &[f32]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| anyhow!("Failed to start WAV encode: {}", e))?;
        for &sample in samples {
            writer
                .write_sample(f32_to_i16(sample))
                .map_err(|e| anyhow!("Failed to encode WAV sample: {}", e))?;
        }
        writer
            .finalize()
            .map_err(|e| anyhow!("Failed to finalize WAV: {}", e))?;
    }

    Ok(cursor.into_inner())
}

/// Pull the transcript out of a provider response.
///
/// `response_format=json` yields `{"text": "..."}`, and `verbose_json` adds
/// fields around the same `text`. Some self-hosted servers answer with bare
/// text regardless of the requested format, so that is accepted too.
fn extract_text(body: &str) -> Result<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
            return Ok(text.trim().to_string());
        }
        // A JSON body without `text` is an error shape, not a transcript —
        // returning the raw JSON would paste `{"error":...}` into the user's
        // document.
        // OpenAI-style `error`, or ElevenLabs/FastAPI-style `detail`.
        if let Some(error) = value.get("error").or_else(|| value.get("detail")) {
            return Err(anyhow!(
                "Provider returned an error: {}",
                error_message(error)
            ));
        }
        if value.is_object() {
            return Err(anyhow!(
                "Provider response had no 'text' field: {}",
                trimmed.chars().take(200).collect::<String>()
            ));
        }
    }

    Ok(trimmed.to_string())
}

/// Human-readable message from a provider error value: a string, an object
/// with `message`, or FastAPI's validation list of `{msg}` objects.
fn error_message(error: &serde_json::Value) -> String {
    if let Some(text) = error.as_str() {
        return text.to_string();
    }
    if let Some(message) = error.get("message").and_then(|m| m.as_str()) {
        return message.to_string();
    }
    if let Some(items) = error.as_array() {
        let messages: Vec<&str> = items
            .iter()
            .filter_map(|item| item.get("msg").and_then(|m| m.as_str()))
            .collect();
        if !messages.is_empty() {
            return messages.join("; ");
        }
    }
    error.to_string()
}

/// Pull a readable message out of a non-2xx body, falling back to the raw
/// (truncated) body when it isn't a recognizable error shape.
fn http_error_detail(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .or_else(|| value.get("detail"))
                .map(error_message)
        })
        .unwrap_or_else(|| body.chars().take(400).collect())
}

fn form_fields(cfg: &RemoteSttConfig) -> Vec<(&'static str, String)> {
    if cfg.api_format == RemoteTranscriptionApiFormat::ElevenLabs {
        let mut fields = vec![
            ("model_id", cfg.model.clone()),
            ("file_format", "pcm_s16le_16".to_string()),
            // Dictation output is pasted as-is; "(laughter)" tags don't belong.
            ("tag_audio_events", "false".to_string()),
        ];
        if let Some(language) = &cfg.language {
            fields.push(("language_code", language.clone()));
        }
        // A list field is sent as one form part per item.
        for term in &cfg.keyterms {
            fields.push(("keyterms", term.clone()));
        }
        return fields;
    }

    let mut fields = vec![
        ("model", cfg.model.clone()),
        ("response_format", "json".to_string()),
    ];
    // The translations route always outputs English and rejects `language`.
    if !cfg.translate {
        if let Some(language) = &cfg.language {
            fields.push(("language", language.clone()));
        }
    }
    if let Some(prompt) = &cfg.prompt {
        fields.push(("prompt", prompt.clone()));
    }
    fields
}

/// Transcribe synchronously, for the blocking transcription path.
///
/// The work runs on a dedicated OS thread because `reqwest::blocking` panics
/// when constructed on a Tokio worker, and `transcribe()` is reached from
/// async command handlers.
pub fn transcribe_blocking(cfg: &RemoteSttConfig, samples: &[f32]) -> Result<String> {
    let (audio, file_name, mime) = match cfg.api_format {
        RemoteTranscriptionApiFormat::OpenAi => {
            (encode_wav_16k_mono(samples)?, "audio.wav", "audio/wav")
        }
        RemoteTranscriptionApiFormat::ElevenLabs => (
            encode_pcm_s16le(samples),
            "audio.pcm",
            "application/octet-stream",
        ),
    };
    let cfg = cfg.clone();
    let url = cfg.url();
    debug!(
        "Remote transcription: POST {} (model={}, {} bytes {})",
        url,
        cfg.model,
        audio.len(),
        file_name
    );

    std::thread::spawn(move || -> Result<String> {
        let client = reqwest::blocking::Client::builder()
            .default_headers(auth_headers(&cfg.api_key, cfg.api_format)?)
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?;

        let part = reqwest::blocking::multipart::Part::bytes(audio)
            .file_name(file_name)
            .mime_str(mime)
            .map_err(|e| anyhow!("Failed to attach audio: {}", e))?;

        let mut form = reqwest::blocking::multipart::Form::new().part("file", part);
        for (key, value) in form_fields(&cfg) {
            form = form.text(key, value);
        }

        let response = client
            .post(cfg.url())
            .multipart(form)
            .send()
            .map_err(|e| anyhow!("Request to {} failed: {}", cfg.provider_label, e))?;

        let status = response.status();
        let body = response.text().unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!(
                "{} returned {}: {}",
                cfg.provider_label,
                status,
                http_error_detail(&body)
            ));
        }

        extract_text(&body)
    })
    .join()
    .map_err(|_| anyhow!("Remote transcription thread panicked"))?
}

/// List the models a provider offers, for the model picker.
///
/// OpenRouter supports narrowing `/models` to speech-to-text via
/// `output_modalities=transcription`; other providers return everything and are
/// filtered client-side only by what the user recognizes.
pub async fn list_models(cfg: &RemoteSttConfig, models_query: Option<&str>) -> Result<Vec<String>> {
    if cfg.api_format == RemoteTranscriptionApiFormat::ElevenLabs {
        return Ok(ELEVENLABS_STT_MODELS
            .iter()
            .map(|m| m.to_string())
            .collect());
    }

    let base = cfg.base_url.trim().trim_end_matches('/');
    let url = match models_query {
        Some(query) if !query.is_empty() => format!("{}/models?{}", base, query),
        _ => format!("{}/models", base),
    };

    let client = reqwest::Client::builder()
        .default_headers(auth_headers(&cfg.api_key, cfg.api_format)?)
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to fetch models: {}", e))?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!(
            "Model list request failed ({}): {}",
            status,
            body.chars().take(400).collect::<String>()
        ));
    }

    let parsed: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| anyhow!("Failed to parse model list: {}", e))?;

    let mut models = Vec::new();
    if let Some(data) = parsed.get("data").and_then(|d| d.as_array()) {
        for entry in data {
            if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                models.push(id.to_string());
            }
        }
    } else if let Some(array) = parsed.as_array() {
        for entry in array {
            if let Some(id) = entry.as_str() {
                models.push(id.to_string());
            } else if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                models.push(id.to_string());
            }
        }
    }

    models.sort();
    models.dedup();
    Ok(models)
}

/// End-to-end check of URL, credentials, and model by transcribing a short
/// silent clip. Cheap (a fraction of a second of audio) but exercises exactly
/// the path a real recording takes.
pub async fn test_connection(cfg: RemoteSttConfig) -> Result<String> {
    // 0.4 s of silence: long enough that providers accept it, short enough to be
    // effectively free.
    let silence = vec![0.0f32; 6_400];
    tauri::async_runtime::spawn_blocking(move || transcribe_blocking(&cfg, &silence))
        .await
        .map_err(|e| anyhow!("Connection test task failed: {}", e))?
        .map(|text| text.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RemoteSttConfig {
        RemoteSttConfig {
            provider_label: "OpenRouter".into(),
            api_format: RemoteTranscriptionApiFormat::OpenAi,
            base_url: "https://openrouter.ai/api/v1/".into(),
            api_key: "test-key".into(),
            model: "microsoft/mai-transcribe-2".into(),
            language: Some("en".into()),
            prompt: Some("Handy".into()),
            keyterms: Vec::new(),
            translate: false,
            timeout: Duration::from_secs(30),
        }
    }

    #[test]
    fn url_appends_transcriptions_and_normalizes_trailing_slash() {
        assert_eq!(
            cfg().url(),
            "https://openrouter.ai/api/v1/audio/transcriptions"
        );
    }

    #[test]
    fn url_uses_translations_route_when_translating() {
        let mut c = cfg();
        c.translate = true;
        assert_eq!(c.url(), "https://openrouter.ai/api/v1/audio/translations");
    }

    #[test]
    fn form_fields_include_model_language_and_prompt() {
        let fields = form_fields(&cfg());
        assert!(fields.contains(&("model", "microsoft/mai-transcribe-2".to_string())));
        assert!(fields.contains(&("language", "en".to_string())));
        assert!(fields.contains(&("prompt", "Handy".to_string())));
        assert!(fields.contains(&("response_format", "json".to_string())));
    }

    #[test]
    fn form_fields_omit_language_when_translating() {
        let mut c = cfg();
        c.translate = true;
        let fields = form_fields(&c);
        assert!(!fields.iter().any(|(key, _)| *key == "language"));
    }

    #[test]
    fn extract_text_reads_json_text_field() {
        assert_eq!(
            extract_text(r#"{"text":"  hello world  ","usage":{"seconds":1}}"#).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn extract_text_accepts_plain_text_bodies() {
        assert_eq!(
            extract_text("hello from whisper.cpp\n").unwrap(),
            "hello from whisper.cpp"
        );
    }

    #[test]
    fn extract_text_surfaces_provider_errors() {
        let err = extract_text(r#"{"error":{"message":"No auth credentials found"}}"#)
            .expect_err("error body must not be treated as a transcript");
        assert!(err.to_string().contains("No auth credentials found"));
    }

    #[test]
    fn extract_text_rejects_json_without_text() {
        assert!(extract_text(r#"{"segments":[]}"#).is_err());
    }

    #[test]
    fn encode_wav_produces_riff_header_and_pcm_payload() {
        let wav = encode_wav_16k_mono(&[0.0, 0.5, -0.5]).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");

        let mut reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.spec().channels, 1);
        let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0], 0);
        assert!(samples[1] > 16_000);
        assert!(samples[2] < -16_000);
    }

    #[test]
    fn resolve_config_requires_api_key_for_hosted_providers() {
        let mut settings = crate::settings::get_default_settings();
        settings.remote_transcription_enabled = true;
        settings.remote_transcription_provider_id = "openrouter".into();

        let err = resolve_config(&settings).expect_err("missing key must be reported");
        assert!(err.contains("API key"));

        settings
            .remote_transcription_api_keys
            .insert("openrouter".into(), "sk-or-test".into());
        let resolved = resolve_config(&settings).expect("config should resolve once keyed");
        assert_eq!(resolved.model, "microsoft/mai-transcribe-2");
        // Default language intent is "auto", which must not be sent verbatim.
        assert_eq!(resolved.language, None);
    }

    #[test]
    fn resolve_config_allows_self_hosted_without_key_and_uses_selected_language() {
        let mut settings = crate::settings::get_default_settings();
        settings.remote_transcription_enabled = true;
        settings.remote_transcription_provider_id = "custom".into();
        settings.selected_language = "no".into();
        settings.custom_words = vec!["Handy".into(), "OpenRouter".into()];

        let resolved = resolve_config(&settings).expect("self-hosted needs no key");
        assert_eq!(resolved.language.as_deref(), Some("no"));
        assert_eq!(resolved.prompt.as_deref(), Some("Handy, OpenRouter"));
        assert_eq!(resolved.timeout, Duration::from_secs(30));
    }

    /// End-to-end exercise of the real HTTP path against a one-shot stub
    /// server: multipart encoding, auth header, and response parsing. Nothing
    /// leaves the loopback interface.
    #[test]
    fn transcribe_blocking_posts_multipart_and_parses_transcript() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub server");
        let port = listener.local_addr().unwrap().port();

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            // Read until the body is complete: headers, then Content-Length bytes.
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).expect("read request");
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&raw).to_string();
                if let Some(header_end) = text.find("\r\n\r\n") {
                    let length = text
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            if name.eq_ignore_ascii_case("content-length") {
                                value.trim().parse::<usize>().ok()
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    if raw.len() >= header_end + 4 + length {
                        break;
                    }
                }
            }

            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 30\r\n\r\n{\"text\":\"hello from the stub\"}",
                )
                .expect("write response");
            stream.flush().ok();
            String::from_utf8_lossy(&raw).to_string()
        });

        let config = RemoteSttConfig {
            provider_label: "Stub".into(),
            api_format: RemoteTranscriptionApiFormat::OpenAi,
            base_url: format!("http://127.0.0.1:{}/v1", port),
            api_key: "sk-stub".into(),
            model: "microsoft/mai-transcribe-2".into(),
            language: Some("en".into()),
            prompt: Some("Handy".into()),
            keyterms: Vec::new(),
            translate: false,
            timeout: Duration::from_secs(10),
        };

        let transcript = transcribe_blocking(&config, &[0.0, 0.25, -0.25])
            .expect("stub transcription should succeed");
        assert_eq!(transcript, "hello from the stub");

        let request = server.join().expect("stub server thread");
        assert!(request.starts_with("POST /v1/audio/transcriptions HTTP/1.1"));
        assert!(request.contains("authorization: Bearer sk-stub"));
        assert!(request.contains("multipart/form-data"));
        assert!(request.contains("name=\"file\"; filename=\"audio.wav\""));
        assert!(request.contains("name=\"model\""));
        assert!(request.contains("microsoft/mai-transcribe-2"));
        assert!(request.contains("name=\"language\""));
        assert!(request.contains("name=\"prompt\""));
        assert!(request.contains("RIFF"));
    }

    #[test]
    fn transcribe_blocking_reports_provider_http_errors() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub server");
        let port = listener.local_addr().unwrap().port();

        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(
                b"HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: 47\r\n\r\n{\"error\":{\"message\":\"No auth credentials found\"}}",
            );
        });

        let config = RemoteSttConfig {
            provider_label: "Stub".into(),
            api_format: RemoteTranscriptionApiFormat::OpenAi,
            base_url: format!("http://127.0.0.1:{}/v1", port),
            api_key: String::new(),
            model: "whisper-1".into(),
            language: None,
            prompt: None,
            keyterms: Vec::new(),
            translate: false,
            timeout: Duration::from_secs(10),
        };

        let err = transcribe_blocking(&config, &[0.0; 16])
            .expect_err("a 401 must not be treated as a transcript");
        let message = err.to_string();
        assert!(message.contains("401"), "unexpected error: {}", message);
        assert!(message.contains("Stub"), "unexpected error: {}", message);
    }

    #[test]
    fn resolve_config_ignores_translation_for_providers_without_the_route() {
        let mut settings = crate::settings::get_default_settings();
        settings.remote_transcription_enabled = true;
        settings.translate_to_english = true;

        settings.remote_transcription_provider_id = "openrouter".into();
        settings
            .remote_transcription_api_keys
            .insert("openrouter".into(), "sk-or-test".into());
        assert!(!resolve_config(&settings).unwrap().translate);

        settings.remote_transcription_provider_id = "openai".into();
        settings
            .remote_transcription_api_keys
            .insert("openai".into(), "sk-test".into());
        assert!(resolve_config(&settings).unwrap().translate);
    }

    fn elevenlabs_cfg() -> RemoteSttConfig {
        RemoteSttConfig {
            provider_label: "ElevenLabs".into(),
            api_format: RemoteTranscriptionApiFormat::ElevenLabs,
            base_url: "https://api.elevenlabs.io/v1/".into(),
            api_key: "xi-test".into(),
            model: "scribe_v2".into(),
            language: Some("en".into()),
            prompt: Some("Handy, Tauri".into()),
            keyterms: vec!["Handy".into(), "Tauri".into()],
            translate: false,
            timeout: Duration::from_secs(30),
        }
    }

    #[test]
    fn elevenlabs_url_targets_speech_to_text_even_when_translating() {
        let mut c = elevenlabs_cfg();
        assert_eq!(c.url(), "https://api.elevenlabs.io/v1/speech-to-text");
        c.translate = true;
        assert_eq!(c.url(), "https://api.elevenlabs.io/v1/speech-to-text");
    }

    #[test]
    fn elevenlabs_form_fields_use_scribe_names_and_repeat_keyterms() {
        let fields = form_fields(&elevenlabs_cfg());
        assert!(fields.contains(&("model_id", "scribe_v2".to_string())));
        assert!(fields.contains(&("language_code", "en".to_string())));
        assert!(fields.contains(&("file_format", "pcm_s16le_16".to_string())));
        assert!(fields.contains(&("tag_audio_events", "false".to_string())));
        let keyterms: Vec<&str> = fields
            .iter()
            .filter(|(key, _)| *key == "keyterms")
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(keyterms, vec!["Handy", "Tauri"]);
        // OpenAI field names must not leak into the Scribe request.
        assert!(!fields
            .iter()
            .any(|(key, _)| matches!(*key, "model" | "prompt" | "response_format")));
    }

    #[test]
    fn elevenlabs_keyterms_drop_terms_the_api_would_reject() {
        let long = "x".repeat(ELEVENLABS_MAX_KEYTERM_CHARS);
        let terms = elevenlabs_keyterms(
            [
                " Handy ",
                "",
                "handy",
                "one two three four five six",
                "C[++]",
                long.as_str(),
                "Tauri",
            ]
            .into_iter(),
        );
        assert_eq!(terms, vec!["Handy", "Tauri"]);
    }

    #[test]
    fn resolve_config_builds_elevenlabs_keyterms_from_hint_and_custom_words() {
        let mut settings = crate::settings::get_default_settings();
        settings.remote_transcription_enabled = true;
        settings.remote_transcription_provider_id = "elevenlabs".into();
        settings
            .remote_transcription_api_keys
            .insert("elevenlabs".into(), "xi-test".into());
        settings.remote_transcription_prompt = "Handy, Tauri".into();
        settings.custom_words = vec!["Kubernetes".into(), "tauri".into()];
        settings.translate_to_english = true;

        let resolved = resolve_config(&settings).expect("keyed ElevenLabs resolves");
        assert_eq!(
            resolved.api_format,
            RemoteTranscriptionApiFormat::ElevenLabs
        );
        assert_eq!(resolved.model, "scribe_v2");
        assert_eq!(resolved.keyterms, vec!["Handy", "Tauri", "Kubernetes"]);
        assert!(!resolved.translate, "Scribe has no translation mode");
    }

    #[test]
    fn resolve_config_sends_no_elevenlabs_keyterms_without_vocabulary() {
        let mut settings = crate::settings::get_default_settings();
        settings.remote_transcription_provider_id = "elevenlabs".into();
        settings
            .remote_transcription_api_keys
            .insert("elevenlabs".into(), "xi-test".into());
        assert!(resolve_config(&settings).unwrap().keyterms.is_empty());
    }

    #[test]
    fn encode_pcm_s16le_is_headerless_little_endian() {
        assert_eq!(
            encode_pcm_s16le(&[0.0, 1.0, -1.0, 2.0]),
            vec![0x00, 0x00, 0xff, 0x7f, 0x01, 0x80, 0xff, 0x7f]
        );
    }

    #[test]
    fn extract_text_surfaces_elevenlabs_detail_errors() {
        let err =
            extract_text(r#"{"detail":{"status":"invalid_api_key","message":"Invalid API key"}}"#)
                .expect_err("detail body must not be treated as a transcript");
        assert!(err.to_string().contains("Invalid API key"));

        let err =
            extract_text(r#"{"detail":[{"loc":["body","model_id"],"msg":"field required"}]}"#)
                .expect_err("validation errors must surface");
        assert!(err.to_string().contains("field required"));
    }

    #[test]
    fn extract_text_reads_elevenlabs_transcript() {
        assert_eq!(
            extract_text(
                r#"{"language_code":"eng","language_probability":0.9,"text":"Hi there","words":[]}"#
            )
            .unwrap(),
            "Hi there"
        );
    }

    #[tokio::test]
    async fn elevenlabs_model_list_is_static() {
        let models = list_models(&elevenlabs_cfg(), None).await.unwrap();
        assert_eq!(models, vec!["scribe_v2", "scribe_v1"]);
    }

    /// Same loopback stub as the OpenAI test, checking the Scribe dialect:
    /// route, `xi-api-key` auth, raw PCM part, and an error `detail`.
    #[test]
    fn transcribe_blocking_speaks_elevenlabs_dialect() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub server");
        let port = listener.local_addr().unwrap().port();

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).expect("read request");
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&raw).to_string();
                if let Some(header_end) = text.find("\r\n\r\n") {
                    let length = text
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            if name.eq_ignore_ascii_case("content-length") {
                                value.trim().parse::<usize>().ok()
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    if raw.len() >= header_end + 4 + length {
                        break;
                    }
                }
            }
            let body = br#"{"language_code":"eng","text":"hello scribe","words":[]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).expect("write head");
            stream.write_all(body).expect("write body");
            stream.flush().ok();
            String::from_utf8_lossy(&raw).to_string()
        });

        let mut config = elevenlabs_cfg();
        config.base_url = format!("http://127.0.0.1:{}/v1", port);
        let transcript =
            transcribe_blocking(&config, &[0.0, 0.25, -0.25]).expect("stub transcription");
        assert_eq!(transcript, "hello scribe");

        let request = server.join().expect("stub server thread");
        assert!(request.starts_with("POST /v1/speech-to-text HTTP/1.1"));
        assert!(request.contains("xi-api-key: xi-test"));
        assert!(!request.to_lowercase().contains("authorization:"));
        assert!(request.contains("name=\"file\"; filename=\"audio.pcm\""));
        assert!(request.contains("name=\"model_id\""));
        assert!(request.contains("name=\"keyterms\""));
        assert!(!request.contains("RIFF"), "Scribe gets raw PCM, not WAV");
    }
}
