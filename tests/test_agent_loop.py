"""Exercise the Muse Spark tool-calling loop with a scripted stand-in for the model."""

import json
from types import SimpleNamespace as NS

from landproof.agents.atlas_agent import AtlasAgent
from landproof.geometry import Airframe, Camera
from landproof.meta.sam import OracleSam
from landproof.sim.replay import Pose, ReplaySim
from landproof.verifier import Verifier


class FakeSpark:
    """Returns OpenAI-shaped messages: plans scene-specific concepts, then claims, then finalizes."""

    def __init__(self):
        self.turn = 0
        self.seen = []

    def _call(self, i, name, args):
        return NS(id=f"call_{self.turn}_{i}", type="function", function=NS(name=name, arguments=json.dumps(args)))

    def chat(self, messages, tools=None, **kw):
        self.seen.append(messages)
        names = {t["function"]["name"] for t in tools}
        assert names == {"segment_concepts", "propose_zones", "submit_claims", "finalize_atlas"}
        enum = tools[0]["function"]["parameters"]["properties"]["phrases"]["items"]["enum"]
        assert "waterlogged road" in enum and "power line" in enum
        self.turn += 1
        if self.turn == 1:
            calls = [self._call(0, "segment_concepts", {"phrases": ["flood water", "waterlogged road", "power line", "cattle", "tarpaulin"]})]
        elif self.turn == 2:
            calls = [self._call(0, "propose_zones", {})]
        elif self.turn == 3:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            zones = json.loads(tool_msgs[-1]["content"])["zones"]
            calls = [self._call(i, "submit_claims", {"zone_id": z["zone_id"], "absent_phrases": ["flood water", "power line", "person"], "rationale": "dry grass"})
                     for i, z in enumerate(zones[:3])]
        else:
            certified = [json.loads(m["content"]) for m in messages if m.get("role") == "tool" and "verdict" in m["content"]]
            ok = [c["zone_id"] for c in certified if c["verdict"] == "PASS"]
            calls = [self._call(0, "finalize_atlas", {"ranked_zone_ids": ok + ["Z99"], "summary": "dry zones away from flood and wires"})]
        return NS(content="", tool_calls=calls)


def test_spark_tool_loop(world, vocab, small_cfg):
    cam, air = Camera.from_cfg(small_cfg), Airframe.from_cfg(small_cfg)
    sim = ReplaySim(world, cam, vocab, Pose(80, 60, 60, 0))
    obs = sim.observe()
    sam = OracleSam(vocab)
    sam.register(obs.rgb, obs.label)
    spark = FakeSpark()
    agent = AtlasAgent(vocab, small_cfg, Verifier(vocab, sam, small_cfg), cam, air, planner="spark", spark=spark)
    atlas = agent.build(obs.rgb, obs.pose_est.as_tuple())
    assert atlas.entries, atlas.rejected
    assert all(e.cert.passed for e in atlas.entries)
    assert "Z99" not in [e.zone_id for e in atlas.entries]
    # mandatory classes were auto-segmented even though the planner skipped e.g. 'vehicle'
    prop = next(t for t in atlas.trace if t["tool"] == "propose_zones")
    assert prop["result"]["auto_segmented_mandatory"]
    # after propose/submit the harness shows the overlay image back to the model
    last = spark.seen[-1]
    assert any(m["role"] == "user" and isinstance(m["content"], list) and m["content"][0]["text"].startswith("Overlay") for m in last)
    # every tool message answers a call id from the preceding assistant message
    ids = {c["id"] for m in last if m["role"] == "assistant" for c in m.get("tool_calls", [])}
    assert all(m["tool_call_id"] in ids for m in last if m["role"] == "tool")
