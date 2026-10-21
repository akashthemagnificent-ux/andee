# This fork

Upstream: https://github.com/kuafuai/andee

One change: the Google voice provider speaks the **Gemini Developer API**
(`generativelanguage.googleapis.com`) instead of **Vertex AI**
(`aiplatform.googleapis.com`).

## Why

Upstream's Google path is Vertex-only. An ordinary AI Studio key is refused
there with `403 SERVICE_DISABLED` (the Agent Platform API is not enabled on the
project), and Google's own docs state that the dedicated Gemini Live API models
are unsupported in Vertex **Express mode** — so recognition could not connect at
all, whatever was typed into the settings card.

The key works on the Developer API, and both halves were verified against it
end-to-end before this patch: TTS returns `audio/l16; rate=24000; channels=1`
over SSE, and that audio streamed into the Live API transcribes back verbatim.

## What changed

| File | Change |
|---|---|
| `config/VoiceConfig.kt` | base URL → `https://generativelanguage.googleapis.com/v1beta`; ASR model default → `gemini-3.5-transcribe-live` (the Vertex `-preview` spelling is a 404 here) |
| `asr/GoogleAsr.kt` | Live endpoint → `google.ai.generativelanguage.v1beta.GenerativeService`; model named `models/<id>` |
| `tts/GoogleTts.kt` | dropped the `/publishers/google` segment, which is a 404 on this surface |
| `device/SelfCheck.kt` | health probe targets the host actually in use |
| `app/build.gradle` | signing config from env/`local.properties`, so every CI build shares one package identity |
| `.github/workflows/andee-signed.yml` | builds an installable arm64 debug APK |

## Settings card, if you configure by hand

| Field | Value |
|---|---|
| Voice provider | Google |
| API key | your AI Studio key |
| Endpoint | `https://generativelanguage.googleapis.com/v1beta` |
| Project / Location | leave **blank** — not used on this surface |
| Transcription model | `gemini-3.5-transcribe-live` |
| TTS model | `gemini-3.1-flash-tts-preview` |
| Language | blank (auto) |

## Installing

The artifact is built with `-PdbgSuffix=.fork`, so its package id is
`net.kuafuai.andee.fork` and it installs **beside** the official app rather than
over it. The official APK is signed with the maintainer's throwaway debug key,
so replacing it in place would require an uninstall — and that takes the chat
history, notebook and vault with it.

To move data across, open the old app first and note anything worth keeping;
there is no export path in the app for the notebook or the vault.

## Rotating the key

`GOOGLE_VOICE_KEY` is a repository secret and is baked into
`BuildConfig.GOOGLE_VOICE_KEY` at package time. Update the secret and re-run the
workflow; no source change needed.
