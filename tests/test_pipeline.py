"""End-to-end offline pipeline (oracle evidence): atlas -> intruder -> hold/abort -> divert -> link loss -> touchdown."""

import numpy as np

from landproof.bench.landbench import Frame, run_bench
from landproof.bench.strategies import BlindLand, FlattestPatch, LandProofStrategy, MaskScorer
from landproof.agents.atlas_agent import AtlasAgent
from landproof.demo import Scenario, run_scenario
from landproof.geometry import Airframe, Camera
from landproof.meta.glimmer import OracleGlimmer
from landproof.meta.sam import OracleSam
from landproof.perception.student import OracleStudent
from landproof.sim.replay import Actor, Pose, ReplaySim
from landproof.verifier import Verifier


def test_closed_loop_landing_with_intruder(world, vocab, small_cfg, tmp_path):
    sc = Scenario(
        name="test",
        start=Pose(116.0, 30.0, 30.0, 0.0),   # already over the park zone
        actors=[Actor("person", [(126.0, 22.0), (116.0, 30.0), (106.0, 38.0)], speed_mps=1.5, start_t=5.0, name="walker")],
        link_down_t=14.0,
        max_t=90.0,
        fps=10.0,
    )
    s = run_scenario(world, sc, small_cfg, vocab, sam=OracleSam(vocab), student=OracleStudent(min_area_px=2), glimmer=OracleGlimmer(vocab),
                     out_dir=tmp_path, render=False, intro=False, verbose=False)
    assert s["landed"], s
    assert s["touchdown_gt"]["safe"], s
    assert s["ledger_chain_ok"]
    assert s["baseline_flattest_patch"]["verdict"] in ("PASS", "FAIL")
    assert s["first_hold_or_abort_t"] is not None, "intruder should trigger HOLD/ABORT"


def test_landbench_flood_trap(world, vocab, small_cfg):
    cam, air = Camera.from_cfg(small_cfg), Airframe.from_cfg(small_cfg)
    sim = ReplaySim(world, cam, vocab, Pose(115, 78, 30, 0), [], seed=0)  # looking at the flooded field
    frames = []
    for k, (x, y) in enumerate([(115, 78), (112, 85), (45, 85)]):
        rgb, label, _ = sim.render(Pose(x, y, 30, 0), 0)
        frames.append(Frame(f"f{k}", rgb, label, cam.gsd(30), 30.0, "synthetic"))

    def agent_factory(rgb, ctx):
        sam = OracleSam(vocab)
        sam.register(rgb, ctx["gt"])
        return AtlasAgent(vocab, small_cfg, Verifier(vocab, sam, small_cfg), cam, air, planner="scripted")

    fixed = [c.name for c in vocab.hazard_classes if c.fixed_vocab]
    strategies = [
        BlindLand(),
        FlattestPatch(air),
        MaskScorer("fixed-vocab (UAVid classes, GT)", vocab, small_cfg, air, lambda rgb, ctx: ctx["gt"], classes=fixed),
        LandProofStrategy(agent_factory),
    ]
    summary = run_bench(frames, strategies, vocab, footprint_m=air.diameter_m / 2 + air.touchdown_margin_m)
    assert summary["flattest-patch"]["false_safe_rate"] > 0      # smooth floodwater fools it
    assert summary["landproof"]["false_safe_rate"] == 0.0
