import pytest

from gmethods import Scenario


class TestScenario:
    @pytest.mark.parametrize("name", Scenario.PRESETS)
    def test_presets_build(self, name):
        assert isinstance(Scenario.preset(name), Scenario)

    def test_overrides(self):
        sc = Scenario.preset("baseline", n_chats=10, p_stay=0.5)
        assert sc.n_chats == 10 and sc.p_stay == 0.5

    def test_null_switches_off_effects(self):
        sc = Scenario.preset("null")
        assert set(sc.delta.values()) == {0.0}
        assert set(sc.gamma.values()) == {0.0}
        assert set(sc.progress_keep.values()) == {1.0}

    def test_unknown_preset(self):
        with pytest.raises(ValueError):
            Scenario.preset("nope")

    def test_validation(self):
        with pytest.raises(ValueError):
            Scenario(task_mix=(1.0,))
