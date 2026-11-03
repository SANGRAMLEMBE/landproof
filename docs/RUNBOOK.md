# LandProof runbook: Sat Oct 31 → submission

Deadline: **Wed Nov 4, 9:00 PM ET = Thu Nov 5, 7:30 AM IST.** Plan to submit by
**Wed 10 PM IST**. Office hours run 10–11 AM ET (7:30–8:30 PM IST) on weekdays. Tue
Nov 3 is the last troubleshooting day.

## Running from VS Code (Windows, Mac or Linux)

1. Install Python 3.11 or 3.12 and Git. Open the `landproof` folder in VS Code and open a terminal (Ctrl+`).
2. Create the environment:
   - Windows: `py -3.12 -m venv .venv` then `.venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv .venv && source .venv/bin/activate`
3. If you have an NVIDIA GPU, install CUDA PyTorch **first**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`. On Windows, plain `pip install torch` gives you a CPU-only build.
4. `pip install -e ".[student,dev]"`
5. Copy `.env.example` to `.env` and paste `MODEL_API_KEY=...` (keep `LANDPROOF_BUDGET_USD=130`). `.env` is git-ignored; never commit it and never paste the key into code or chat. Also set a $130 Spend Limit on the Billing page.
6. VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → `.venv`. The Run and Debug panel (Ctrl+Shift+D) has ready configurations: *Check Meta API access*, *Show spend*, *Offline demo*, *Atlas: Muse Spark + SAM 3.1*, *Demo: real Meta models in the simulator*, *Tests*.
7. Always run commands from the repo root (caches go to `cache/`, outputs to `runs/`).

If something fails on a live API call, copy the traceback (it never contains your key) and share it with the team.

## Saturday Oct 31 (tonight): go/no-go

```bash
# 0. repo: create a PUBLIC GitHub repo named "landproof", then
git remote add origin https://github.com/SANGRAMLEMBE/landproof.git && git push -u origin main

# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[student,dev]"
pytest                                    # all offline tests should pass
python scripts/05_descent_demo.py --evidence oracle --student oracle --glimmer oracle   # sanity video

# 2. keys + access (claim the $150 credit form first; a card is required)
cp .env.example .env                      # MODEL_API_KEY=...
python scripts/00_check_access.py         # must print ALL GOOD
python scripts/00_check_access.py --image # one Muse Image edit
```

- Request access to `facebook/sam3.1` on Hugging Face (fallback if the API fails). Do it now: requests can be rejected.
- Request Modal credits ($100) and Baseten credits.
- Download **FloodNet Track 1** (supervised) and the **ICG Semantic Drone Dataset** into `data/`. They are git-ignored.
- **Kill criterion:** if `00_check_access` fails from India by Sunday noon, post in `#questions`, then use `--evidence local` (SAM open weights on a GPU) for labelling.

```bash
# 3. vocabulary validation (~$4-5): FloodNet has no people/animals/wires, ICG does
python scripts/01_validate_vocab.py --dataset floodnet --root data/FloodNet --split val --n 20
python scripts/01_validate_vocab.py --dataset icg --root data/semantic_drone --n 15 --merge
# 4. cheap labelling: check packed-video mode against per-image mode (~$0.70), then bulk (~$4-15)
python scripts/02_autolabel.py --dataset floodnet --root data/FloodNet --split train --ab-check 10
python scripts/02_autolabel.py --dataset floodnet --root data/FloodNet --split train --out data/pseudo/floodnet_train --limit 300 --mode auto
python scripts/spend.py                   # spent vs the $130 cap (see docs/BUDGET.md)
```

## Sunday Nov 1: student, Glimmer, Spark atlas, footage

```bash
# student trained on SAM labels only, and the GT-trained reference (same val set)
python scripts/03_train_student.py --train-dir data/pseudo/floodnet_train --val-dataset floodnet --val-root data/FloodNet --out checkpoints/student_sam.pt
python scripts/03_train_student.py --train-dataset floodnet --train-root data/FloodNet --limit 300 --val-dataset floodnet --val-root data/FloodNet --out checkpoints/student_gt.pt
#   no local GPU? -> modal_apps/train_student.py (A10G)

# Glimmer on Modal (L40S, FP8)
modal secret create glimmer-api-key GLIMMER_API_KEY=$(openssl rand -hex 16)
modal deploy modal_apps/glimmer_server.py && modal run modal_apps/glimmer_server.py
python scripts/00_check_access.py --glimmer

# watch Muse Spark plan an atlas on a real frame (screen-record this for the video)
python scripts/04_build_atlas.py --image data/FloodNet/.../7577.jpg --gsd 0.015 --planner spark --evidence meta-api
```

**Footage (if you can fly safely and legally):**
- Drone of 250 g or less with its UIN. Check the Digital Sky map for a green zone that day. Private land, with the owner's consent.
- Turn on the camera's SRT/subtitle telemetry. Point the gimbal straight down (−90°). Hover at 30–40 m over open ground, then descend slowly.
- A teammate walks into the area below at ~15–20 m altitude. Do three takes, plus one with no intruder.
- If it has rained, film waterlogged ground too: that is your hero "runway" shot.
- If you can't fly: use a phone on a pole or a rooftop overhead shot, labelled honestly as replayed real footage.

Post the X teaser: a 10 s clip of the HUD hold/abort from the sim, with "building in public for @AIatMeta's hackathon".

## Monday Nov 2: experiments

```bash
# operator's spoken override, transcribed by Muse Voice Transcribe (record 2-3 s on your phone, convert to 16 kHz WAV)
#   ffmpeg -i stop.m4a -ac 1 -ar 16000 -sample_fmt s16 data/voice/stop_ta.wav ;  pip install -e ".[voice]"
python scripts/05_descent_demo.py --evidence oracle --student oracle --glimmer oracle --voice-at 31.5 --voice-wav data/voice/stop_ta.wav --voice-lang ta --out runs/demo_voice

# real footage in shadow mode (SAM 3.1 video tracks for movers + Spark atlas on frame 0)
python scripts/08_shadow_mode.py --video data/footage/DJI_0042.MP4 --srt data/footage/DJI_0042.SRT --planner spark --stride 2 --out runs/shadow_take1

# HazardForge: Muse Image traps -> SAM-verified -> who gets fooled? round 2 designed by Spark
python scripts/06_hazardforge.py --frames "data/forge_frames/*.jpg" --gsd 0.015 --rounds 2 --student checkpoints/student_sam.pt --spark
# student v2 = SAM labels + verified trap frames
python scripts/03_train_student.py --train-dir data/pseudo/floodnet_train --extra-dir runs/forge/train_pairs --val-dataset floodnet --val-root data/FloodNet --out checkpoints/student_v2.pt

# LandBench on real data with SAM 3.1 evidence (~$0.15 per frame for the full stack)
python scripts/07_landbench.py --source floodnet --root data/FloodNet --split test --n 80 --evidence meta-api --student checkpoints/student_sam.pt --out runs/landbench_floodnet
python scripts/07_landbench.py --source icg --root data/semantic_drone --n 60 --evidence meta-api --student checkpoints/student_sam.pt --out runs/landbench_icg

# the full stack in the closed-loop simulator
python scripts/05_descent_demo.py --evidence meta-api --planner spark --student checkpoints/student_sam.pt --glimmer remote --out runs/demo_real
```

## Tuesday Nov 3: freeze and record

- **Feature freeze at 6 PM IST.** Copy every number into the README Results table.
- Ask anything blocking in office hours (7:30–8:30 PM IST, last troubleshooting day).
- Record the takes (see the shot list below). **Hero X post at about 8 PM IST.** Reply to every reply.

## Wednesday Nov 4: submit

- Edit to **2:55**. Upload to YouTube (unlisted or public) and check the link in a private window.
- Make sure the repo is public, the README renders, `pytest` passes on a fresh clone, and both team members are listed.
- Submit by **10 PM IST**. Use `docs/SUBMISSION.md` for the form text.

## Cut order (if you fall behind)

1. PX4 / anything not listed here (never started).
2. HazardForge round 2 (keep round 1 + the failure table).
3. Student v2 retrain.
4. Glimmer on Modal → use `--glimmer none` and say "edge veto: future work".

**Never cut:** certificates + verifier refutation on screen, the walk-in hold/abort, the
flood comparison, and real numbers.

## 3-minute video shot list

| time | shot | source |
|---|---|---|
| 0:00–0:06 | Overhead waterlogged street. "BATTERY 4%, MUST LAND NOW. Flattest area: 98% smooth → LANDING" | intro of `05_descent_demo` (or a real frame) |
| 0:06–0:12 | "CLAIM REFUTED: standing water" → the pad turns red, jumps to dry ground. Title: *To a drone, a flooded street looks like the perfect runway.* | same |
| 0:12–0:25 | You on camera: drone-vision researcher in Chennai; why fixed-class models fail in the monsoon | phone |
| 0:25–0:50 | Screen recording: Muse Spark calling `segment_concepts("waterlogged road", "power line"…)` → zones → claims turning green or red | `04_build_atlas --planner spark` |
| 0:50–1:35 | One continuous take: descent, person walks in, **"predicted entry in 3.0 s" → HOLD → ABORT → DIVERT (skipped zone on intruder's path)** | real footage in shadow mode, else sim |
| 1:35–1:55 | LINK LOST → Muse Glimmer edge check → touchdown | sim |
| 1:55–2:25 | HazardForge grid: Muse Image traps, SAM-verified; fooled-rate round 1 → round 2 | `06_hazardforge` outputs |
| 2:25–2:45 | Two numbers: false-safe rate (flattest / fixed-vocab / LandProof) and student-on-SAM-labels % of GT | README table |
| 2:45–3:00 | "Built Oct 27–Nov 4. The LLM never flies the drone." Repo URL. Hero split frame | |

Label every clip SIMULATION or REPLAY OF REAL FOOTAGE (the HUD already does this).
