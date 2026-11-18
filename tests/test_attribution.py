import numpy as np
import pytest

from gmethods import ChatSimulator, Scenario
from gmethods.attribution import attribute, attribution_regimes

BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 10_000))


@pytest.fixture(scope="module")
def truth_attribution():
    sim = ChatSimulator(Scenario.preset("baseline"))
    regimes = attribution_regimes(BUCKETS)
    risks = sim.true_risks(regimes, n_mc=20_000).risks
    never_any = sim.true_risks([regimes[0], regimes[-1]], n_mc=20_000).risks[
        regimes[-1].name
    ]
    return attribute(dict(risks), BUCKETS), never_any


class TestAttribution:
    def test_regime_count(self):
        assert len(attribution_regimes(BUCKETS)) == 1 + 4 * 5

    def test_bad_buckets(self):
        with pytest.raises(ValueError):
            attribution_regimes(((1, 2), (4, 10_000)))
        with pytest.raises(ValueError):
            attribution_regimes(((1, 2), (3, 5)))

    def test_adds_up_to_attributable_risk(self, truth_attribution):
        att, _ = truth_attribution
        total = att.by_bucket.to_numpy().sum()
        assert total == pytest.approx(att.attributable_risk, abs=1e-12)
        s = att.summary()
        types = s.drop(index=["all handoffs", "not caused by handoffs"])
        assert types["share_of_handoff_complaints"].sum() == pytest.approx(1.0)

    def test_last_base_is_no_handoffs(self, truth_attribution):
        att, never_any = truth_attribution
        assert att.risk_no_handoffs == pytest.approx(never_any, abs=1e-12)

    def test_single_turn_buckets_are_exact(self, truth_attribution):
        att, _ = truth_attribution
        np.testing.assert_allclose(att.remainder.iloc[:3], 0.0, atol=1e-12)

    def test_null_scenario_attributes_nothing(self):
        sim = ChatSimulator(Scenario.preset("null"))
        risks = sim.true_risks(attribution_regimes(BUCKETS), n_mc=5_000).risks
        att = attribute(dict(risks), BUCKETS)
        np.testing.assert_allclose(att.by_bucket.to_numpy(), 0.0, atol=1e-12)


def test_ltmle_attribution(small_data):
    from gmethods import LTMLE
    from gmethods.attribution import attribute_ltmle

    from .conftest import FAST

    model = LTMLE(learner_params=FAST).fit(small_data.panel)
    att = attribute_ltmle(model, BUCKETS)
    s = att.summary()
    assert att.risk_natural == pytest.approx(
        small_data.chats["complained"].mean(), abs=2e-3
    )
    assert s.loc["all handoffs", "se_per_1000"] > 0
    assert set(att.se) == {
        "transfer",
        "escalation",
        "requeue",
        "shift_change",
        "all handoffs",
    }


class TestMarginal:
    def test_shares_add_up_and_interaction_small(self):
        from gmethods.attribution import marginal_from_risks
        from gmethods.interventions import scale_set

        sim = ChatSimulator(Scenario.preset("baseline"))
        risks = sim.true_risks(scale_set(factor=0.9), n_mc=50_000).risks
        out = marginal_from_risks(dict(risks), factor=0.9)
        types = out.drop(index=["all types together", "interaction"])
        assert types["share"].sum() == pytest.approx(1.0)
        joint = out.loc["all types together", "per_1000"]
        assert abs(out.loc["interaction", "per_1000"]) < 0.15 * abs(joint)

    def test_with_ltmle(self, small_data):
        from gmethods import LTMLE
        from gmethods.attribution import attribute_marginal

        from .conftest import FAST

        model = LTMLE(learner_params=FAST).fit(small_data.panel)
        out = attribute_marginal(model)
        assert list(out.index[:4]) == [
            "transfer",
            "escalation",
            "requeue",
            "shift_change",
        ]
        assert (out["se_per_1000"].iloc[:4] > 0).all()
