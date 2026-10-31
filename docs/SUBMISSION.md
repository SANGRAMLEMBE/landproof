# Submission form drafts

Replace every `[bracket]` with your real result before you paste. Do not submit
numbers you have not measured.

**Team name:** LandProof

**Team members:** Sangram Lembe, [Wizardian's real name]

**Track:** Indie Developer

**Problem statement:** Push the Frontier

**X post link (Most Viral):** [link]

**Public GitHub repository:** https://github.com/SANGRAMLEMBE/landproof

**3-minute demo video:** [YouTube link]

---

### Project description

LandProof gives drones a capability they lack today: a provably safe emergency landing
in an open world.

When a drone must land now (battery critical, motor fault, GPS loss), today's failsafe
descends blind. Research systems pick a site, but only from a fixed list of about 20
classes. They have no word for floodwater, which in a Chennai monsoon is the flattest
patch in view, or for a power line, a cow or a tarpaulin.

LandProof splits the job so that no model is trusted blindly.

- **Before the emergency, while the link is up.** Muse Spark 1.3 studies the scene, picks
  the hazards that matter here from a vetted vocabulary, and makes falsifiable claims
  about candidate zones ("no standing water within 5.4 m of Z2"). SAM 3.1 supplies the
  pixel evidence. A deterministic verifier certifies a zone only if every claim survives.
- **During the descent.** A student model distilled from SAM 3.1 labels, plus ego-motion
  compensation and ground-frame tracking, predicts when a person will walk into the zone.
  The drone then holds, aborts and diverts to the next certified zone, skipping any zone
  on the intruder's path.
- **If the link drops.** Muse Glimmer (open weights) re-checks the zone and can only
  veto.
- **Offline red-teaming.** HazardForge uses Muse Image to invent traps, and SAM 3.1 must
  confirm each one before it counts.

Every decision lands in a hash-chained ledger. The LLM never flies the drone.

[Headline result: e.g. "On FloodNet test frames, false-safe landings fell from X% (flattest patch) and Y% (fixed vocabulary) to Z% with LandProof."]

### Describe your use of Meta products

- **SAM 3.1 (Meta Model API, Responses API + `meta-sam-parser`):** used four ways.
  - Evidence for every safety claim, including zoomed re-checks on zone crops.
  - Zero-annotator auto-labelling of training data: [N] images → student at [x]% of the GT-trained mIoU.
  - Identity-preserving video tracks of people and vehicles on real footage in shadow mode.
  - Verifying every Muse Image trap before it is used.
- **Muse Spark 1.3:** a multimodal tool-calling agent (`segment_concepts`, `propose_zones`,
  `submit_claims`, `finalize_atlas`).
  - Phrases are JSON-schema enums of a vocabulary we validated against labelled frames.
  - It sees overlay images of its own candidate zones and certificates.
  - It also designs HazardForge's next round from the failure table, using structured output.
- **Muse Image 1.0:** `/images/edits` inserts realistic hazards (floodwater, a person lying
  down, livestock, debris) into real nadir frames to red-team the landing system. Traps
  become training data for student v2.
- **Muse Glimmer (open weights, Apache-2.0):** served with vLLM on Modal (L40S, FP8) as the
  edge model. When the cloud link is lost it answers a structured yes/no per hazard on the
  zone crop, and can only veto.
- **Muse Code:** [describe honestly if you used it, e.g. "scaffolded the Modal/vLLM
  deployment and the ledger tests", or delete this line].

### Why are Meta's models important to your solution?

The core idea, an open vocabulary checked by evidence rather than trusted, needs a
promptable segmenter that understands concepts. SAM 3.1 turns any phrase into
instance masks in one call, so the safety vocabulary can grow to "waterlogged road" or
"cattle" without collecting a dataset.

- **SAM 3.1:** besides open-vocabulary evidence, its video tracks follow people through
  moving drone footage, and its labels let us distil an onboard model with no human
  annotation.
- **Muse Spark:** gives the open vocabulary judgement. It picks which concepts matter in
  this scene, and its tool calling and structured outputs let it act inside a verifiable
  loop instead of issuing opinions.
- **Muse Image:** lets us manufacture the rare, dangerous scenes we cannot wait for a flood
  to film. SAM 3.1 makes those synthetic labels trustworthy.
- **Muse Glimmer:** gives an open-weight model that can run off-cloud for the moment
  connectivity disappears.

No other single family covers evidence, reasoning, generation and edge deployment.

### Feedback on Meta products

- **SAM 3.1 API:** the special-token output plus the official parser worked first time.
  "Empty output = no match" is exactly right for safety checks.
  - Requests: document max image size.
  - Return per-instance confidence by default.
  - Allow a batch of phrases per request: we make one call per concept.
- **Muse Spark 1.3:** strong tool use and vision.
  - Requests: a note in the tool-calling guide that reasoning tokens share `max_tokens` would have saved us empty replies.
  - Document the latency at `low` effort.
- **Muse Image:** convincing edits. A mask or region input for `/images/edits` would make
  targeted insertion deterministic; we had to centre crops.
- **Muse Glimmer:** easy to serve with the vLLM recipe and its parsers. A published
  image-input example and FP8 memory guidance per GPU would help.
- [Add anything else you actually hit: access from India, rate limits, Muse Code.]

### Feedback on partner technologies

- **Modal:** [e.g. "vLLM web_server + Volumes made Glimmer deployable in under an hour;
  cold starts of N minutes - keep a warm container for demos."]
- **Baseten / Fireworks:** [only if used].
