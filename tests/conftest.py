import os

# Two learners sharing a few cores with OpenMP spin-waiting is very slow.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest

from gmethods import ChatSimulator, Scenario

FAST = {"max_iter": 20, "learning_rate": 0.2}


@pytest.fixture(scope="session")
def small_sim():
    return ChatSimulator(Scenario.preset("baseline", n_chats=3_000, seed=7))


@pytest.fixture(scope="session")
def small_data(small_sim):
    return small_sim.simulate()
