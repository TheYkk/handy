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

use std::io::Cursor;
use std::time::Duration;

use anyhow::{anyhow, Result};
use log::debug;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, REFERER, USER_AGENT};

use crate::settings::AppSettings;

/// A fully-resolved remote endpoint: everything needed to make one request,
/// with no further settings lookups (and therefore no surprises if settings
/// change mid-transcription).
#[derive(Debug, Clone)]
pub struct RemoteSttConfig {
    pub provider_label: String,
    /// API root without the `/audio/transcriptions` suffix.
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// `None` asks the provider to auto-detect.
    pub language: Option<String>,
    /// Vocabulary hint sent as OpenAI's `prompt`. Providers that ignore the
    /// field (OpenRouter) simply drop it.
    pub prompt: Option<String>,
    /// Use `/audio/translations` (English output) instead of transcription.
    pub translate: bool,
    pub timeout: Duration,
}

impl RemoteSttConfig {
    fn url(&self) -> String {
        let base = self.base_url.trim().trim_end_matches('/');
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

    let timeout_seconds = if settings.remote_transcription_timeout_seconds == 0 {
        30
    } else {
        settings.remote_transcription_timeout_seconds
    };

    Ok(RemoteSttConfig {
        provider_label: provider.label.clone(),
        base_url,
        api_key,
        model: model.trim().to_string(),
        language,
        prompt,
        // Only honored where the provider actually implements the separate
        // translations route; elsewhere the request would 404.
        translate: settings.translate_to_english && provider.supports_translations,
        timeout: Duration::from_secs(timeout_seconds),
    })
}

fn auth_headers(api_key: &str) -> Result<HeaderMap> {
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
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key.trim()))
                .map_err(|e| anyhow!("Invalid API key value: {}", e))?,
        );
    }
    Ok(headers)
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
            // Clamp before casting: an out-of-range f32 would wrap instead of
            // saturating and turn a loud frame into noise.
            let scaled = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(scaled)
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
        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(|m| m.as_str())
                .map(|m| m.to_string())
                .unwrap_or_else(|| error.to_string());
            return Err(anyhow!("Provider returned an error: {}", message));
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

fn form_fields(cfg: &RemoteSttConfig) -> Vec<(&'static str, String)> {
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
    let wav = encode_wav_16k_mono(samples)?;
    let cfg = cfg.clone();
    let url = cfg.url();
    debug!(
        "Remote transcription: POST {} (model={}, {} bytes wav)",
        url,
        cfg.model,
        wav.len()
    );

    std::thread::spawn(move || -> Result<String> {
        let client = reqwest::blocking::Client::builder()
            .default_headers(auth_headers(&cfg.api_key)?)
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?;

        let part = reqwest::blocking::multipart::Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")
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
                body.chars().take(400).collect::<String>()
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
    let base = cfg.base_url.trim().trim_end_matches('/');
    let url = match models_query {
        Some(query) if !query.is_empty() => format!("{}/models?{}", base, query),
        _ => format!("{}/models", base),
    };

    let client = reqwest::Client::builder()
        .default_headers(auth_headers(&cfg.api_key)?)
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
            base_url: "https://openrouter.ai/api/v1/".into(),
            api_key: "test-key".into(),
            model: "microsoft/mai-transcribe-2".into(),
            language: Some("en".into()),
            prompt: Some("Handy".into()),
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
            base_url: format!("http://127.0.0.1:{}/v1", port),
            api_key: "sk-stub".into(),
            model: "microsoft/mai-transcribe-2".into(),
            language: Some("en".into()),
            prompt: Some("Handy".into()),
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
            base_url: format!("http://127.0.0.1:{}/v1", port),
            api_key: String::new(),
            model: "whisper-1".into(),
            language: None,
            prompt: None,
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
}
