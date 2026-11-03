import numpy as np
import pytest

from landproof.budget import Budget, BudgetExceeded
from landproof.data.autolabel import estimate_usd
from landproof.evidence import Instance
from landproof.meta.voice import parse_command


def test_budget_blocks_before_overspend(tmp_path):
    b = Budget(tmp_path / "spend.json", cap_usd=1.0)
    for _ in range(300):
        b.reserve("sam-3.1/image")
        b.add("sam-3.1/image")
    assert b.total == pytest.approx(0.75)
    with pytest.raises(BudgetExceeded):
        b.reserve("sam-3.1/image", 100)          # $0.25 x 1.25 safety > $0.25 remaining
    with pytest.raises(BudgetExceeded):
        b.preflight({"sam-3.1/image": 200}, "too big")
    # persists across processes / restarts
    assert Budget(tmp_path / "spend.json", cap_usd=1.0).total == pytest.approx(0.75)


def test_retries_dropped_connection_but_not_client_errors(monkeypatch):
    from landproof.meta import client

    monkeypatch.setattr(client.time, "sleep", lambda s: None)

    class APIConnectionError(Exception):  # openai's name; no status_code, the request never got a response
        pass

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise APIConnectionError("Connection error.")
        return "ok"

    assert client.with_retries(flaky) == "ok" and len(calls) == 3

    class BadRequest(Exception):
        status_code = 400

    with pytest.raises(BadRequest):
        client.with_retries(lambda: (_ for _ in ()).throw(BadRequest()))


def test_video_mode_is_cheaper():
    img = estimate_usd(300, ["a"] * 20, [], 50)
    vid = estimate_usd(300, [], ["a"] * 20, 50)
    b = Budget(path="/nonexistent/x.json", cap_usd=1000)
    assert b.cost("sam-3.1/image", img["sam-3.1/image"]) == pytest.approx(15.0)
    assert b.cost("sam-3.1/video_frame", vid["sam-3.1/video_frame"]) < 4.0


@pytest.mark.parametrize("text,cmd", [
    ("Stop! there are kids", "abort"), ("abort abort", "abort"), ("நிறுத்து!", "abort"), ("இறங்காதே", "abort"),
    ("रुको", "abort"), ("ruko bhai", "abort"), ("looks good, continue", None), ("", None),
])
def test_voice_commands(text, cmd):
    assert parse_command(text) == cmd


def test_video_batch_maps_back_to_image_coords(tmp_path):
    from landproof.meta.sam import MetaApiSam

    class FakeVideoSam(MetaApiSam):
        def segment_video(self, video_path, phrase):
            # image 1 (300x600 portrait) is letterboxed to 480x960 at x-offset 400 in a 1280x960 canvas
            raster = np.ones((160, 160), bool)
            inst = Instance("p#1", phrase, (400 + 160.0, 320.0, 400 + 320.0, 480.0), raster, 0.9, 3, "fake")
            return {3: [inst]}

    sam = FakeVideoSam(cache_dir=None)
    imgs = [np.zeros((480, 640, 3), np.uint8), np.zeros((600, 300, 3), np.uint8)]
    out = sam.segment_images_as_video(imgs, ["person"], work_dir=tmp_path)
    assert out["person"][0] == []
    (inst,) = out["person"][1]
    assert inst.bounds == pytest.approx((100.0, 200.0, 200.0, 300.0))


def test_voice_override_aborts_immediately(world, vocab, small_cfg, tmp_path):
    from landproof.demo import Scenario, run_scenario
    from landproof.meta.glimmer import OracleGlimmer
    from landproof.meta.sam import OracleSam
    from landproof.perception.student import OracleStudent
    from landproof.sim.replay import Pose

    sc = Scenario(name="voice", start=Pose(116.0, 30.0, 30.0, 0.0), human_abort_t=8.0, voice_text="நிறுத்து!", max_t=40.0, fps=10.0)
    s = run_scenario(world, sc, small_cfg, vocab, sam=OracleSam(vocab), student=OracleStudent(), glimmer=OracleGlimmer(vocab),
                     out_dir=tmp_path, render=False, intro=False, verbose=False)
    import json

    rows = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    aborts = [r for r in rows if r["kind"] == "transition" and r["payload"]["to"] == "ABORT"]
    assert aborts and "human override" in aborts[0]["payload"]["why"] and "நிறுத்து" in aborts[0]["payload"]["why"]
    assert any(r["kind"] == "human_voice" for r in rows)
    assert s["ledger_chain_ok"]
