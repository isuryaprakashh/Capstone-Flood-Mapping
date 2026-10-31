# Orchestrator status

- **Phase:** train
- **Sessions started:** 14
- **Running:** sn8-train-a
- **Code commit:** 7d4177d07c6e323169f785d081d010c2126e1f48
- **Waiting for quota until:** —
- **Updated:** 2026-10-31T11:53:30+00:00

Per-run progress: [PROGRESS.md](PROGRESS.md)

## Recent events

- 2026-10-31T11:53:30+00:00 — Started train session 14 on sn8-train-a (code 7d4177d, 11.5 h)
- 2026-10-31T11:52:08+00:00 — sn8-train-b (train) ended: complete after 698 min
- 2026-10-31T00:14:18+00:00 — Started train session 13 on sn8-train-b (code 7d4177d, 11.5 h)
- 2026-10-26T22:44:48+00:00 — Weekly GPU quota used up (0.0 GPU h left); waiting until 2026-10-31T00:10:00+00:00
- 2026-10-26T22:43:43+00:00 — sn8-train-a (train) ended: complete after 688 min
- 2026-10-26T16:47:15+00:00 — Command: training code re-pinned to 7d4177d
- 2026-10-26T16:42:56+00:00 — Command: training code re-pinned to 793cc61
- 2026-10-26T11:15:59+00:00 — Started train session 12 on sn8-train-a (code 2637979)
- 2026-10-26T11:15:13+00:00 — sn8-train-b (train) ended: complete after 696 min
- 2026-10-25T23:39:25+00:00 — Started train session 11 on sn8-train-b (code 2637979)
- 2026-10-25T23:39:01+00:00 — sn8-train-a (train) ended: complete after 678 min
- 2026-10-25T12:20:31+00:00 — Started train session 10 on sn8-train-a (code 2637979)
- 2026-10-25T12:20:29+00:00 — Dataset prepared; training starts
- 2026-10-25T12:20:21+00:00 — sn8-prepare (prepare) ended: complete after 20 min
- 2026-10-25T12:00:02+00:00 — Started prepare session 9 on sn8-prepare (code 2637979)
- 2026-10-25T12:00:00+00:00 — Smoke test passed; preparing the full dataset
- 2026-10-25T11:59:46+00:00 — sn8-smoke (smoke) ended: complete after 33 min
- 2026-10-25T11:39:25+00:00 — Command: training code re-pinned to 2637979
- 2026-10-25T11:26:45+00:00 — Started smoke session 8 on sn8-smoke (code b4a70bb)
- 2026-10-25T11:26:43+00:00 — Smoke test failed: models do not fit their batch size: ["ra_fusion {'backbone': 'convnext_tiny.dinov3_lvd1689m'}"] (see smoke/gpu_probe.json) — retry 2/2
- 2026-10-25T11:26:29+00:00 — sn8-smoke (smoke) ended: complete after 30 min
- 2026-10-25T10:56:02+00:00 — Started smoke session 7 on sn8-smoke (code b4a70bb)
- 2026-10-25T10:37:38+00:00 — Smoke test failed: runs not completed: ['smoke_siamese_unet'] — retry 1/2
- 2026-10-25T10:37:33+00:00 — sn8-smoke (smoke) ended: error after 13 min
- 2026-10-25T10:24:48+00:00 — Started smoke session 6 on sn8-smoke (code 6e97ab0)
- 2026-10-25T10:24:44+00:00 — Manually resumed at phase 'smoke'
- 2026-10-25T10:07:42+00:00 — Smoke test failed: runs not completed: ['smoke_siamese_unet'] — retry 1/2
- 2026-10-25T10:07:36+00:00 — sn8-smoke (smoke) ended: error after 19 min
- 2026-10-25T09:48:37+00:00 — Started smoke session 5 on sn8-smoke (code 2dcf24d)
- 2026-10-25T09:48:33+00:00 — Manually resumed at phase 'smoke'
