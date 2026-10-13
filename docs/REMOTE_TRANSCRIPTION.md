# Remote transcription

Handy transcribes locally by default. This fork adds an option to send recordings
to a remote speech-to-text API instead — a cloud provider such as OpenRouter or
ElevenLabs, or a server you host yourself.

The motivating case is [`microsoft/mai-transcribe-2`](https://openrouter.ai/microsoft/mai-transcribe-2)
on OpenRouter: 60 languages with automatic language identification, ranked #1 on
the FLEURS multilingual benchmark, at $0.10 per hour of audio. Nothing about the
implementation is specific to it.

## What it works with

Two wire formats are supported, both `multipart/form-data`:

- **OpenAI-compatible** `POST /audio/transcriptions` with `Authorization: Bearer`.
  Every provider below except ElevenLabs.
- **ElevenLabs Scribe** `POST /v1/speech-to-text` with an `xi-api-key` header.

Providers ship preconfigured; `Custom` is for any other OpenAI-compatible server.

| Provider             | Base URL                                  | Default model                   | Key required |
| -------------------- | ----------------------------------------- | ------------------------------- | ------------ |
| OpenRouter           | `https://openrouter.ai/api/v1`            | `microsoft/mai-transcribe-2`    | yes          |
| OpenAI               | `https://api.openai.com/v1`               | `gpt-4o-mini-transcribe`        | yes          |
| Groq                 | `https://api.groq.com/openai/v1`          | `whisper-large-v3-turbo`        | yes          |
| DeepInfra            | `https://api.deepinfra.com/v1/openai`     | `openai/whisper-large-v3-turbo` | yes          |
| ElevenLabs           | `https://api.elevenlabs.io/v1`            | `scribe_v2`                     | yes          |
| Custom (self-hosted) | editable, e.g. `http://localhost:8000/v1` | `whisper-1`                     | no           |

`Custom` has been used against whisper.cpp's `server`, speaches,
faster-whisper-server, and vLLM. Any server that copies OpenAI's route should
work; if it returns bare text instead of JSON, that is handled too.

### ElevenLabs Scribe

[Scribe v2](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) is
ElevenLabs' batch speech-to-text model: 90+ languages with automatic detection.
How Handy's settings map onto the request:

| Handy                         | Scribe request                                               |
| ----------------------------- | ------------------------------------------------------------ |
| Model                         | `model_id`: `scribe_v2` (default) or `scribe_v1`             |
| Language (not auto)           | `language_code`; auto-detect omits it                        |
| Vocabulary hint, custom words | `keyterms`, one form part per term                           |
| Audio                         | raw 16 kHz mono PCM with `file_format=pcm_s16le_16`, not WAV |
| (fixed)                       | `tag_audio_events=false`, so "(laughter)" never gets pasted  |

- **Keyterms cost extra.** ElevenLabs bills keyterm requests at a 20%
  surcharge, so keyterms are sent only when you have a vocabulary hint or custom
  words. The hint is split on commas; terms ElevenLabs would reject (50+
  characters, more than 5 words, or containing `< > { } [ ] \`) are dropped
  rather than failing the recording.
- **Raw PCM** skips server-side decoding. In a quick test it came back about
  0.3 s faster than the same clip as WAV.
- **The model list is fixed** to `scribe_v2` and `scribe_v1`. ElevenLabs'
  `/models` lists only text-to-speech models, and scoped API keys often can't
  read it. You can still type any other model id.
- **No translation.** Scribe only transcribes, so "Translate to English" is
  ignored for this provider.
- An API key restricted to the **Speech to Text** permission is enough.

## Setup

1. Open **Models** in settings. The **Remote transcription** group is at the top.
2. Toggle **Use a remote model**.
3. Pick a provider, paste an API key, and choose a model — the refresh button
   lists what the provider offers (OpenRouter is filtered to speech-to-text
   models via `output_modalities=transcription`).
4. Press **Test connection**. It pushes 0.4 s of silence through the real
   transcription path, so it exercises the URL, the key, and the model together.
   Cost is a rounding error.

The tray menu shows which engine will handle the next recording:
`Remote: OpenRouter · microsoft/mai-transcribe-2` or `Local: <model>`. The same
text is in the tray tooltip.

## Settings

| Setting                  | Default                                   | Notes                                                                                                 |
| ------------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Use a remote model       | off                                       | When on, audio leaves your device.                                                                    |
| Provider / model         | OpenRouter · `microsoft/mai-transcribe-2` | Model is per provider.                                                                                |
| API key                  | empty                                     | Stored in Handy's settings store, redacted from logs.                                                 |
| Vocabulary hint          | empty                                     | Sent as OpenAI's `prompt`, or as ElevenLabs `keyterms`. Your custom words are appended automatically. |
| Timeout                  | 30 s                                      | Upstream providers time out around 60 s per request anyway.                                           |
| Fall back to local model | on                                        | On failure, transcribe locally instead.                                                               |

## Behavior worth knowing

**The local model is not loaded while remote is on.** Enabling remote unloads
it, and the pre-load on hotkey press is skipped. With fallback enabled the local
engine is loaded on demand the _first time_ a remote call fails, not upfront. So
remote-only users pay no memory for an ASR model, and Handy can run without one
installed at all.

**Voice activity detection still runs locally.** Silero VAD filters every 16 kHz
frame before anything is sent, and non-speech frames are dropped. The provider
receives speech plus hangover tails, not your silences. It is one request at the
end of the recording, not a chunked stream.

**Live preview is off in remote mode.** The OpenAI transcription route returns
one finished transcript; there are no partial results to stream, and
MAI-Transcribe-2 on Azure is a whole-file route as well. You get the compact
overlay instead of the live-text panel.

**Custom words and filler-word removal still apply.** The returned text goes
through the same fuzzy correction and cleanup as a local transcript. Custom words
are additionally sent as a decode hint; providers that ignore `prompt`
(OpenRouter does) simply drop it.

**"Translate to English" is only honored where the provider implements it.**
OpenAI and Groq expose `/audio/translations`; OpenRouter, DeepInfra and
ElevenLabs do not, so the setting is ignored there rather than producing a 404.

**Long recordings.** Multipart uploads cap at 25 MB, about 13 minutes of 16 kHz
mono WAV, and providers time out near 60 s of processing. Dictation-length clips
are far inside both limits; hour-long recordings would need chunking, which is
not implemented.

## Privacy

With remote enabled, your audio is uploaded to the provider you selected. That is
the whole point of the feature, but it is worth stating plainly, because it
inverts Handy's default promise of staying offline.

Independent of remote mode, Handy keeps a local history: a WAV of each recording
in `recordings/` plus a row in `history.db`, inside the app data directory. See
[History and retention](#history-and-retention).

API keys are stored in the settings store on disk (`settings_store.json`), the
same place post-processing keys live. They are redacted from debug output, but
they are not in the system keychain.

## History and retention

Both the transcript and the audio are kept locally.

- **Audio**: 16 kHz mono WAV per recording, in `<app data>/recordings/`.
- **Text**: SQLite at `<app data>/history.db`, table `transcription_history`
  (`file_name`, `timestamp`, `saved`, `title`, `transcription_text`,
  `post_processed_text`, `post_process_prompt`, `post_process_requested`).

App data directory: `~/Library/Application Support/<bundle id>/` on macOS,
`%APPDATA%\<bundle id>\` on Windows, `~/.local/share/<bundle id>/` on Linux.

Cleanup runs every time an entry is saved, and is driven by the
**recording retention** setting:

| Setting                     | Effect                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------- |
| Preserve limit (default)    | Keeps the most recent N unsaved entries, where N is the history limit (default 5). |
| 3 days / 2 weeks / 3 months | Deletes unsaved entries older than that.                                           |
| Never                       | Deletes nothing. Grows forever.                                                    |

Deleting an entry removes the database row _and_ the WAV file. Entries you mark
as saved are exempt from every cleanup rule.

## Before you publish releases

The updater is **disabled by default in this fork**, and this is deliberate.
Upstream's config pointed at `cjpais/Handy` releases, signed with upstream's
minisign key. Left as-is, a Handy Remote install would happily "update" itself
into an upstream Handy build — losing this feature while keeping the fork's data
directory.

To ship your own releases:

1. Generate a signing keypair: `bunx @tauri-apps/cli signer generate`.
2. Put the public key in `tauri.conf.json` under `plugins.updater.pubkey`, and
   set `endpoints` to your own repository.
3. Add the private key and its password to CI as
   `TAURI_SIGNING_PRIVATE_KEY` / `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.
4. Flip `default_update_checks_enabled()` in `src-tauri/src/settings.rs` back to
   `true`.

The `endpoints` value currently in the config is a placeholder pointing at this
fork's presumed repository — correct it to yours.

## Implementation

- `src-tauri/src/remote_transcription.rs` — the client: WAV/PCM encoding, the
  OpenAI and ElevenLabs request formats, response parsing, model listing,
  connection test.
- `src-tauri/src/managers/transcription.rs` — `try_remote_transcribe` runs before
  the local engine; `ensure_local_engine_loaded` is the lazy fallback.
- `src-tauri/src/settings.rs` — `RemoteTranscriptionProvider` (with its
  `api_format`), the `remote_transcription_*` settings, and
  `ensure_remote_transcription_defaults`, which adds new built-in providers to
  existing settings stores.
- `src-tauri/src/shortcut/mod.rs` — Tauri commands.
- `src/components/settings/remote-transcription/` — the settings UI.

Tests live next to the client in `remote_transcription.rs`, including three that
run a real HTTP request against a loopback stub server to check the multipart
body, the auth header, and error handling for both formats.
