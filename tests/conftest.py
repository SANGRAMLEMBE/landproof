import pytest

from landproof.config import load_config
from landproof.sim.scene import make_world
from landproof.vocab import Vocabulary


@pytest.fixture(scope="session")
def vocab():
    return Vocabulary.load()


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_cfg():
    return load_config(overrides={"camera": {"width": 480, "height": 360}, "verifier": {"crop_size_px": 512}})


@pytest.fixture(scope="session")
def world(vocab):
    return make_world(vocab, seed=7, gsd=0.12)
