# Changelog

All notable changes to this project will be documented in this file.

## v0.1.0 - 2025-10-11

### Added
- Next.js UI (`services/ui/`) as the primary frontend, replacing embedded HTML in `demo.py`.
- Chat input that posts utterances to ingestion and triggers agent suggestions (suggest-on-send) in `services/ui/pages/index.tsx`.
- RAG flows in UI: document upload (`POST /api/rag/upload`) and search (`GET /api/rag/search`).

### Changed
- `demo.py` root (`get_demo`) is now API-only and returns a JSON message pointing to the Next.js UI.
- Next.js `next.config.js` rewrites: `/api/agent/:path*` now proxies to `http://agent:8000/agent/:path*`.
- Meeting ID is generated client-side post-mount to avoid hydration mismatch in `index.tsx`.

### Fixed
- Agent service compatibility by pinning `httpx==0.27.0` with `openai==1.3.0`.
- UI hydration error caused by `Date.now()` during SSR.

### DevOps
- Dockerized build for UI with Next.js 14.
- Multiple keepalive commits for activity tracking.

### Notes
- Speech-to-Text (mic) not yet wired in the Next.js UI; current flow is text-first.
- Optional next items: mic streaming to ingestion WS, meeting summarization button, finalize `GET /agent/meetings/{id}/suggestions` endpoint.
