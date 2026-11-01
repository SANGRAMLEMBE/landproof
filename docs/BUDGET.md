# Spending plan: hard cap $130 of the $150 credit

Two independent guards:

1. **In code:** every paid call is checked against `cache/spend.json` before it is sent. The cap is `LANDPROOF_BUDGET_USD=130` in `.env`. Every script also prints a cost estimate first and refuses if it would cross the cap. Run `python scripts/spend.py` at any time.
2. **On Meta's side:** on the Billing page at dev.meta.ai, set a **Spend Limit of $130**. This is the real guarantee: the code's prices are list prices and could drift.

Prices used: SAM 3.1 $0.0025 per image call, $0.0002 per video frame (min 50 frames per request); Muse Spark 1.3 $1.25 / $4.25 per M tokens; Muse Image $0.01 per image; Muse Voice Transcribe $0.18 per audio hour. Cached results (`cache/sam/`) are free to re-use, so never delete `cache/` mid-project.

| # | step | command | est. cost |
|---|---|---|---|
| 0 | access check | `00_check_access.py --image` | $0.05 |
| 1 | vocabulary validation: FloodNet 20 + ICG 15 frames | `01_validate_vocab.py` (×2, second with `--merge`) | $4–5 |
| 2a | video-vs-image A/B check, 10 images | `02_autolabel.py --ab-check 10` | $0.70 |
| 2b | auto-label 300 FloodNet images, 20 phrases | `02_autolabel.py --mode auto --limit 300` | $4 (all video) – $15 (all image) |
| 3 | train student A (SAM labels) and B (GT labels) | local GPU or Modal | $0 API |
| 4 | Muse Spark atlas runs for the video, 5–10 | `04_build_atlas.py --planner spark` | $1–2 |
| 5 | real-models simulator demo, 3 runs | `05_descent_demo.py --evidence meta-api --planner spark` | ~$1 |
| 6 | HazardForge, 2 rounds × 8 traps | `06_hazardforge.py` | $3–5 |
| 7 | LandBench: FloodNet 80 + ICG 60 frames (+20 with Spark) | `07_landbench.py` | $9–11 |
| 8 | shadow mode, 3 one-minute takes | `08_shadow_mode.py` | ~$6 |
| 9 | voice override clip | `05_descent_demo.py --voice-wav` | <$0.01 |
| | **planned total** | | **≈ $30–45** |
| | buffer for re-runs and mistakes | | ≈ $85 |

## Rules of thumb

- Run the A/B check before bulk labelling. If most phrases pass, bulk labelling costs about $4 instead of $15.
- Start every experiment with `--limit` or `--n` small (5–10). Look at the output, then scale up.
- Use the scripted planner for bulk benchmarks. Use Muse Spark where its judgement is the point: the atlas demo, and a 20-frame planner comparison.
- Never point anything with people in it at a `-contributor` model. It is cheaper, but Meta may train on that traffic.
