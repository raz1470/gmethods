import numpy as np
import pandas as pd
import pytest

from gmethods import (
    LTMLE,
    CampaignScenario,
    CampaignSimulator,
    CountRegression,
    FrictionScenario,
    FrictionSimulator,
    PooledHazard,
)
from gmethods import interventions as iv

from .conftest import FAST

SIMS = [
    (FrictionSimulator, FrictionScenario),
    (CampaignSimulator, CampaignScenario),
]


@pytest.fixture(scope="module", params=SIMS, ids=["friction", "campaign"])
def sim_and_data(request):
    sim_cls, sc_cls = request.param
    sim = sim_cls(sc_cls.preset("baseline", n_customers=3_000, seed=4))
    return sim, sim.simulate()


class TestWeeklyPanel:
    def test_columns(self, sim_and_data):
        sim, data = sim_and_data
        spec = sim.panel_spec()
        spec.check(data.panel)

    def test_weeks_consecutive_and_stop_at_event(self, sim_and_data):
        sim, data = sim_and_data
        p = data.panel
        g = p.groupby("customer_id")["week"]
        assert (g.min() == 1).all() and (g.max() == g.count()).all()
        stop = p[sim.OUTCOME] + (p[sim.COMPETING] if sim.COMPETING else 0)
        after = stop.groupby(p["customer_id"]).cumsum() - stop
        assert (after == 0).all()

    def test_summary_counts_match_panel(self, sim_and_data):
        sim, data = sim_and_data
        c = data.customers.set_index("customer_id")
        for lv in sim.LEVELS[1:]:
            in_panel = (
                (data.panel["treatment"] == lv).groupby(data.panel["customer_id"]).sum()
            )
            pd.testing.assert_series_equal(
                c[f"n_{lv}_before_event"],
                in_panel.reindex(c.index),
                check_names=False,
                check_dtype=False,
            )

    def test_reproducible(self, sim_and_data):
        sim, _ = sim_and_data
        a, b = sim.simulate(n=300, seed=9), sim.simulate(n=300, seed=9)
        pd.testing.assert_frame_equal(a.panel, b.panel)


@pytest.mark.parametrize("sim_cls,sc_cls", SIMS, ids=["friction", "campaign"])
def test_null_preset_has_no_effects(sim_cls, sc_cls):
    sim = sim_cls(sc_cls.preset("null"))
    truth = sim.true_risks(iv.standard_set(sim.LEVELS), n_mc=5_000)
    np.testing.assert_allclose(truth.to_frame()["rd"], 0.0, atol=1e-12)


@pytest.mark.parametrize("sim_cls,sc_cls", SIMS, ids=["friction", "campaign"])
def test_removing_all_treatment_changes_risk_in_expected_direction(sim_cls, sc_cls):
    sim = sim_cls(sc_cls.preset("baseline"))
    truth = sim.true_risks([iv.natural(), iv.never_any(sim.LEVELS)], n_mc=10_000)
    # Friction raises churn; campaigns raise take-up. Both lower the risk
    # when removed.
    assert truth.rd("never_any") < 0


def test_estimators_run_on_weekly_panel(sim_and_data):
    sim, data = sim_and_data
    spec = sim.panel_spec()
    regimes = [iv.natural(), iv.never(sim.LEVELS[1]), iv.never_any(sim.LEVELS)]
    table = LTMLE(spec, learner_params=FAST).fit(data.panel).estimate(regimes)
    assert set(table.ci) == {r.name for r in regimes}
    PooledHazard(spec).fit(data.panel).estimate(regimes)
    est = CountRegression.for_simulator(sim, "before_complaint_or_all", True)
    assert est.fit(data.customers).estimate(regimes).regimes == [
        r.name for r in regimes
    ]


def test_shap_drivers(sim_and_data):
    pytest.importorskip("shap")
    from gmethods.estimators import ShapDrivers

    sim, data = sim_and_data
    out = ShapDrivers.for_simulator(sim).fit(data.customers).drivers(sample=500)
    assert list(out.index) == list(sim.LEVELS[1:])
    assert out["share"].sum() == pytest.approx(1.0)
