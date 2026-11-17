import numpy as np
import pytest

from gmethods import interventions as iv
from gmethods.interventions import HANDOFF_TYPES


class TestApply:
    a = np.array([0, 1, 2, 3, 4, 2])
    u = np.array([0.1, 0.9, 0.2, 0.7, 0.4, 0.8])

    def test_natural_is_identity(self):
        np.testing.assert_array_equal(
            iv.natural().apply(self.a, self.u, HANDOFF_TYPES), self.a
        )

    def test_never_removes_only_listed(self):
        out = iv.never("escalation").apply(self.a, self.u, HANDOFF_TYPES)
        np.testing.assert_array_equal(out, [0, 1, 0, 3, 4, 0])

    def test_never_any(self):
        out = iv.never_any().apply(self.a, self.u, HANDOFF_TYPES)
        assert (out == 0).all()

    def test_scale_keeps_draws_below_factor(self):
        out = iv.scale("escalation", factor=0.5).apply(self.a, self.u, HANDOFF_TYPES)
        np.testing.assert_array_equal(out, [0, 1, 2, 3, 4, 0])

    def test_does_not_mutate_input(self):
        a = self.a.copy()
        iv.never_any().apply(a, self.u, HANDOFF_TYPES)
        np.testing.assert_array_equal(a, self.a)


class TestTransformProbs:
    p = np.array([[0.5, 0.2, 0.2, 0.05, 0.05], [0.9, 0.0, 0.1, 0.0, 0.0]])

    @pytest.mark.parametrize(
        "intervention",
        [
            iv.natural(),
            iv.never("transfer"),
            iv.never_any(),
            iv.scale("escalation", factor=0.3),
        ],
    )
    def test_rows_sum_to_one(self, intervention):
        out = intervention.transform_probs(self.p, HANDOFF_TYPES)
        np.testing.assert_allclose(out.sum(axis=1), 1.0)

    def test_never_moves_mass_to_none(self):
        out = iv.never("transfer").transform_probs(self.p, HANDOFF_TYPES)
        np.testing.assert_allclose(out[0], [0.7, 0.0, 0.2, 0.05, 0.05])

    def test_matches_apply_by_simulation(self):
        rng = np.random.default_rng(0)
        p = np.array([0.4, 0.3, 0.2, 0.06, 0.04])
        a = rng.choice(5, size=200_000, p=p)
        u = rng.random(len(a))
        intervention = iv.scale("transfer", "requeue", factor=0.25)
        out = intervention.apply(a, u, HANDOFF_TYPES)
        empirical = np.bincount(out, minlength=5) / len(out)
        expected = intervention.transform_probs(p[None, :], HANDOFF_TYPES)[0]
        np.testing.assert_allclose(empirical, expected, atol=0.005)


class TestConstructors:
    def test_names(self):
        assert iv.never("transfer").name == "never_transfer"
        assert iv.scale("requeue", factor=0.5).name == "scale_requeue_0.5"

    def test_standard_set(self):
        names = [x.name for x in iv.standard_set()]
        assert names[0] == "natural_course" and names[-1] == "never_any"
        assert len(names) == 6

    def test_scale_set(self):
        regimes = iv.scale_set(factor=0.5)
        names = [x.name for x in regimes]
        assert names[0] == "natural_course" and names[-1] == "scale_any_0.5"
        p = np.full((1, 5), 0.2)
        out = regimes[-1].transform_probs(p, HANDOFF_TYPES)
        np.testing.assert_allclose(out[0], [0.6, 0.1, 0.1, 0.1, 0.1])

    def test_rejects_unknown_level(self):
        with pytest.raises(ValueError):
            iv.never("teleport").apply(np.array([0]), np.array([0.5]), HANDOFF_TYPES)

    def test_rejects_reference_level(self):
        with pytest.raises(ValueError):
            iv.never("none").apply(np.array([0]), np.array([0.5]), HANDOFF_TYPES)

    def test_scale_factor_range(self):
        with pytest.raises(ValueError):
            iv.scale("transfer", factor=1.5)
