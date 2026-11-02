import math

import numpy as np
import pytest

from landproof.claims import ClaimStatus, HazardClaim
from landproof.evidence import Instance
from landproof.geometry import Airframe, Camera, apply_homography
from landproof.ledger import Ledger, verify_chain
from landproof.meta.sam import OracleSam, SamBackend, parse_sam_output
from landproof.perception.tracker import GroundTracker, Detection, Track
from landproof.scorer import Zone, ZoneScorer, clearance_map
from landproof.verifier import Verifier


def test_gsd_roundtrip():
    cam = Camera(70.0, 960, 720)
    g = cam.gsd(80.0)
    assert math.isclose(cam.altitude_for_gsd(g), 80.0, rel_tol=1e-9)
    assert math.isclose(cam.ground_extent(80.0)[0], 2 * 80 * math.tan(math.radians(35)), rel_tol=1e-9)


def test_vocab_is_closed(vocab):
    with pytest.raises(KeyError):
        vocab.class_of("definitely safe spot")
    assert {c.name for c in vocab.mandatory_classes} == {"person", "vehicle", "animal", "water", "wire_pole"}
    assert not vocab.by_name["water"].fixed_vocab  # UAVid has no water class


def _scene(vocab, h=400, w=400):
    masks = {c.id: np.zeros((h, w), bool) for c in vocab.hazard_classes}
    return masks


def test_clearance_uses_per_class_buffers(vocab):
    masks = _scene(vocab)
    pid, wid = vocab.by_name["person"].id, vocab.by_name["water"].id
    masks[pid][200, 100] = True     # person: 5 m buffer
    masks[wid][200, 300] = True     # water: 1.5 m buffer
    cr = clearance_map(masks, vocab.buffers_m(), gsd_m=0.1, work_gsd_m=0.1, include_border=False)
    # 6 m from both: person-limited clearance = 6 - 5 = 1, water = 6 - 1.5 = 4.5
    assert cr.clearance_m[200, 160] == pytest.approx(1.0, abs=0.11)
    assert cr.clearance_m[200, 240] == pytest.approx(4.5, abs=0.11)


def test_scorer_avoids_hazards_and_abstains(vocab, cfg):
    masks = _scene(vocab)
    masks[vocab.by_name["water"].id][:, :200] = True
    sc = ZoneScorer(vocab.buffers_m(), cfg)
    zones = sc.score(masks, 0.1, 2.0)
    assert zones and all(z.center_px[0] - 200 >= (2.0 + 1.5) / 0.1 - 2 for z in zones)
    masks[vocab.by_name["water"].id][:] = True
    assert sc.score(masks, 0.1, 2.0) == []


def _oracle_frame(vocab, person_at=None):
    rgb = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    label = np.zeros((300, 300), np.uint8)
    if person_at:
        y, x = person_at
        label[y - 2 : y + 3, x - 2 : x + 3] = vocab.by_name["person"].id
    return rgb, label


def test_verifier_refutes_and_enforces_mandatory(vocab, cfg):
    rgb, label = _oracle_frame(vocab, person_at=(150, 190))  # 4 m from centre at 0.1 m/px
    sam = OracleSam(vocab)
    sam.register(rgb, label)
    v = Verifier(vocab, sam, cfg | {"verifier": cfg["verifier"] | {"crop_zoom": False}}, Ledger())
    z = Zone("Z1", (150.0, 150.0), 2.0, 20.0, 5.0, 1.0, True)
    claims = [HazardClaim(claim_id="c0", zone_id="Z1", phrase="tree", hazard_class="tree")]
    cert = v.verify_zone(rgb, z, claims, gsd_m=0.1, altitude_m=30.0, t=0.0)
    assert cert.verdict == "FAIL"
    assert set(cert.mandatory_added) == {"person", "vehicle", "animal", "water", "wire_pole"}
    refuted = [r for r in cert.results if r.status == ClaimStatus.REFUTED]
    assert [r.claim.hazard_class for r in refuted] == ["person"]
    assert refuted[0].nearest_m == pytest.approx(3.8, abs=0.3)


def test_verifier_pass_and_backend_error_is_not_absence(vocab, cfg):
    rgb, label = _oracle_frame(vocab, person_at=(150, 290))  # 14 m away
    sam = OracleSam(vocab)
    sam.register(rgb, label)
    v = Verifier(vocab, sam, cfg, None)
    z = Zone("Z1", (150.0, 150.0), 2.0, 20.0, 5.0, 1.0, True)
    assert v.verify_zone(rgb, z, [], 0.1, 30.0, t=0.0).passed

    class Broken(SamBackend):
        name = "broken"

        def _segment_array(self, image, phrase, parent):
            raise RuntimeError("HTTP 503")

    v2 = Verifier(vocab, Broken(), cfg, None)
    cert = v2.verify_zone(rgb, z, [], 0.1, 30.0, t=0.0)
    assert cert.verdict == "FAIL" and all(r.status == ClaimStatus.UNVERIFIABLE for r in cert.results)


def test_mandatory_class_checked_with_best_phrase(vocab, cfg):
    rgb, label = _oracle_frame(vocab, person_at=(150, 290))
    sam = OracleSam(vocab)
    sam.register(rgb, label)
    water = vocab.by_name["water"]
    weak = next(p for p in water.phrases if p != water.default_phrase)
    claims = [HazardClaim(claim_id="c0", zone_id="Z1", phrase=weak, hazard_class="water")]
    cert = Verifier(vocab, sam, cfg, None).verify_zone(rgb, Zone("Z1", (150.0, 150.0), 2.0, 20.0, 5.0, 1.0, True), claims, 0.1, 30.0, t=0.0)
    assert "water" in cert.mandatory_added
    assert {r.claim.phrase for r in cert.results if r.claim.hazard_class == "water"} == {weak, water.default_phrase}


def test_default_phrase_with_unvalidated_stats():
    from landproof.vocab import HazardClass

    c = HazardClass("water", 4, 5.0, True, False, True, False, ["flood water", "pond", "puddle"],
                    stats={"flood water": {"iou": None}, "pond": {"iou": 0.4}, "puddle": {"iou": None}})
    assert c.default_phrase == "pond"
    c.stats = {p: {"iou": None, "unvalidated": True} for p in c.phrases}  # class absent from the dataset
    assert c.default_phrase == "flood water"


def test_phrase_quality_is_yaml_safe(vocab):
    import yaml

    from landproof.data.autolabel import phrase_quality

    rgb, label = _oracle_frame(vocab, person_at=(150, 190))
    sam = OracleSam(vocab)
    sam.register(rgb, label)
    q = phrase_quality(rgb, label, "person", sam, vocab)
    assert all(type(q[k]) is float for k in ("iou", "precision", "recall"))
    yaml.safe_dump(q)  # numpy scalars raise RepresenterError (crashed 01_validate_vocab)


def test_ledger_detects_tampering(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    for i in range(5):
        led.append("event", {"i": i})
    assert verify_chain(led.entries) == (True, None)
    led.entries[2]["payload"]["i"] = 99
    ok, bad = verify_chain(led.entries)
    assert not ok and bad == 2


def test_instance_distance_and_paint():
    m = np.zeros((10, 10), bool)
    m[4:6, 4:6] = True
    inst = Instance.from_full_mask(m, "p#1", "person", "test")
    assert inst.distance_px((5.0, 5.0)) == 0.0
    assert inst.distance_px((15.0, 5.0)) == pytest.approx(9.5, abs=0.6)
    canvas = np.zeros((10, 10), bool)
    inst.paint(canvas)
    assert (canvas == m).all()
    moved = inst.transformed((100.0, 50.0), 2.0)
    assert moved.bounds == (102.0, 52.0, 103.0, 53.0)


def test_parse_empty_sam_output_is_no_match():
    assert parse_sam_output("") == []


def test_time_to_enter():
    tr = Track(1, 1, np.array([10.0, 0.0]), vel=np.array([-2.0, 0.0]))
    assert tr.time_to_enter(4.0) == pytest.approx(3.0)
    tr.vel = np.array([0.0, 2.0])
    assert math.isinf(tr.time_to_enter(4.0))


def test_tracker_velocity_ignores_jitter():
    rng = np.random.default_rng(0)
    gt = GroundTracker()
    for k in range(30):
        t = k / 15
        walker = (10 - 1.4 * t + rng.normal(0, 0.05), 0.0)
        parked = (5 + rng.normal(0, 0.15), 5 + rng.normal(0, 0.15))
        gt.update([Detection(1, walker, 0.3, (0, 0)), Detection(2, parked, 1.0, (0, 0))], t)
    speeds = {tr.cls: tr.speed for tr in gt.tracks}
    assert speeds[1] == pytest.approx(1.4, abs=0.15)
    assert speeds[2] < 0.4


def test_egomotion_recovers_camera_motion(world, vocab, small_cfg):
    from landproof.perception.egomotion import EgoMotion, telemetry_homography
    from landproof.sim.replay import Pose, ReplaySim

    cam = Camera.from_cfg(small_cfg)
    sim = ReplaySim(world, cam, vocab, Pose(40, 40, 30, 0), [], seed=0)
    a = sim.render(Pose(40, 40, 30, 0), 0)[0]
    b = sim.render(Pose(41.5, 39.0, 27, 4.0), 0)[0]
    reg = EgoMotion().register(a, b)
    assert reg.H is not None
    H_true = telemetry_homography((40, 40, 30, 0), (41.5, 39.0, 27, 4.0), cam, cam.width, cam.height)
    pts = np.array([[100, 100], [240, 180], [400, 300]], float)
    err = np.linalg.norm(apply_homography(reg.H, pts) - apply_homography(H_true, pts), axis=1)
    assert err.max() < 2.0
