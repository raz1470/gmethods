import numpy as np
import pandas as pd
import pytest

from gmethods import (
    IPW,
    ChatSimulator,
    CountRegression,
    GFormula,
    PooledHazard,
    Scenario,
)
from gmethods import interventions as iv
from gmethods.estimators import (
    PanelSpec,
    bootstrap,
    natural_course_check,
    positivity_table,
    resample_clusters,
    weight_summary,
)
from gmethods.estimators._features import panel_history

from .conftest import FAST

REGIMES = [iv.natural(), iv.never("escalation"), iv.never_any()]


@pytest.fixture(scope="module")
def fitted_gf(small_data):
    gf = GFormula(learner_params=FAST).fit(small_data.panel)
    table = gf.estimate(REGIMES, n_mc=1_000)
    return gf, table


class TestHistory:
    def test_matches_simulator_columns(self, small_data):
        p = small_data.panel
        spec = PanelSpec()
        h = panel_history(p, spec, spec.treatment_codes(p))
        np.testing.assert_array_equal(h["n_treated"] + 1, p["agents_so_far"])
        np.testing.assert_array_equal(h["since_treated"], p["turns_since_handoff"])
        for lv in ("transfer", "escalation"):
            np.testing.assert_array_equal(h[f"cum_{lv}"], p[f"cum_{lv}"])

    def test_spec_check(self, small_data):
        with pytest.raises(KeyError):
            PanelSpec().check(small_data.panel.drop(columns=["wait"]))


class TestGFormula:
    def test_returns_all_regimes(self, fitted_gf):
        _, table = fitted_gf
        assert table.regimes == [r.name for r in REGIMES]
        assert all(0 <= v <= 1 for v in table.risks.values())

    def test_natural_course_close_to_observed(self, fitted_gf, small_data):
        _, table = fitted_gf
        observed = small_data.chats["complained"].mean()
        assert abs(table.risks["natural_course"] - observed) < 0.015

    def test_never_any_lowers_risk(self, fitted_gf):
        _, table = fitted_gf
        assert table.rd("never_any") < 0

    def test_common_random_numbers(self, small_data):
        gf = GFormula(learner_params=FAST).fit(small_data.panel)
        a = gf.estimate([iv.natural()], n_mc=500, seed=1)
        b = gf.estimate([iv.natural(), iv.never("requeue")], n_mc=500, seed=1)
        assert a.risks["natural_course"] == b.risks["natural_course"]

    def test_natural_course_check(self, fitted_gf):
        gf, _ = fitted_gf
        out = natural_course_check(gf, n_turns=5)
        assert len(out) == 5 and "risk_obs" in out and "sentiment_sim" in out

    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            GFormula().estimate(REGIMES)

    def test_learner_per_role(self, small_data):
        gf = GFormula(learner={"outcome": "linear"}, learner_params=FAST)
        gf.fit(small_data.panel)
        assert type(gf.models_["_outcome"]).__name__ == "Pipeline"
        assert type(gf.models_["_treatment"]).__name__.startswith("HistGradient")

    def test_linear_learner_runs(self, small_data):
        table = (
            GFormula(learner="linear").fit(small_data.panel).estimate(REGIMES, n_mc=300)
        )
        assert set(table.regimes) == {r.name for r in REGIMES}


@pytest.fixture(scope="module")
def ipw(small_data):
    model = IPW(learner_params=FAST).fit(small_data.panel)
    return model, model.estimate(REGIMES)


class TestIPW:
    def test_natural_equals_observed_curve(self, ipw, small_data):
        _, table = ipw
        observed = small_data.chats["complained"].mean()
        # Equal up to chats still open at the last turn.
        assert table.risks["natural_course"] == pytest.approx(observed, abs=1e-3)

    def test_never_censors_treated(self, ipw):
        model, _ = ipw
        w = model.weights(iv.never("escalation"))
        a = model.work_["handoff"].astype(str).to_numpy()
        assert (w[a == "escalation"] == 0).all()

    def test_weight_summary(self, ipw):
        model, _ = ipw
        summary = weight_summary(model)
        assert summary.loc["natural_course", "mean"] == pytest.approx(1.0)

    def test_truncation_caps(self, small_data):
        model = IPW(learner_params=FAST, truncate=0.9).fit(small_data.panel)
        w = model.weights(iv.never_any())
        raw = IPW(learner_params=FAST).fit(small_data.panel).weights(iv.never_any())
        assert w.max() <= raw.max()

    def test_positivity_table(self, ipw):
        model, _ = ipw
        table = positivity_table(model)
        assert "share_p_none_lt_0.01" in table.columns
        assert len(table) == 8


class TestNaive:
    def test_count_regression(self, small_data):
        est = CountRegression("all").fit(small_data.chats)
        table = est.estimate(REGIMES)
        assert table.rd("never_any") < 0
        assert (est.odds_ratios() > 0).all()

    def test_window_and_sentiment_options(self, small_data):
        est = CountRegression("before_complaint_or_all", adjust_sentiment=True)
        assert est.fit(small_data.chats).estimate(REGIMES).label == (
            "count_before_complaint_or_all_sentiment"
        )

    def test_bad_window(self):
        with pytest.raises(ValueError):
            CountRegression("sometimes")

    def test_pooled_hazard(self, small_data):
        table = PooledHazard().fit(small_data.panel).estimate(REGIMES)
        observed = small_data.chats["complained"].mean()
        assert table.risks["natural_course"] == pytest.approx(observed, abs=0.01)

    def test_reverse_causality_inflates_escalation(self):
        sim = ChatSimulator(Scenario.preset("baseline", n_chats=10_000, p_post=0.6))
        data = sim.simulate(seed=2)
        all_counts = CountRegression("all").fit(data.chats).odds_ratios()
        window = (
            CountRegression("before_complaint_or_all").fit(data.chats).odds_ratios()
        )
        assert all_counts["n_escalation"] > window["n_escalation"]


class TestBootstrap:
    def test_resample_keeps_clusters_whole(self, small_data):
        out = resample_clusters(small_data.panel, np.random.default_rng(0))
        assert out["customer_id"].nunique() <= small_data.panel["customer_id"].nunique()
        sizes = out.groupby("chat_id")["turn"].agg(["min", "count", "max"])
        assert (sizes["min"] == 1).all() and (sizes["count"] == sizes["max"]).all()

    def test_intervals(self, small_data):
        def fit_estimate(panel):
            return CountRegression("all").fit(_chats_from(panel)).estimate(REGIMES)

        point, reps = bootstrap(fit_estimate, small_data.panel, n_boot=5)
        low, high = point.ci["never_any"]
        assert low <= high and len(reps) == 5


def _chats_from(panel: pd.DataFrame) -> pd.DataFrame:
    """A minimal chat summary rebuilt from a panel (for the bootstrap test)."""
    g = panel.groupby("chat_id")
    first = g.head(1).set_index("chat_id")
    out = first[
        [
            "customer_id",
            "task_type",
            "first_agent_team",
            "friction_history",
            "start_hour",
            "n_products",
            "age_band",
            "income_band",
        ]
    ].copy()
    out["complained"] = g["complaint"].max()
    out["mean_sentiment"] = g["sentiment"].mean()
    for lv in ("transfer", "escalation", "requeue", "shift_change"):
        out[f"n_{lv}"] = g["handoff"].apply(lambda s, lv=lv: (s == lv).sum())
        out[f"n_{lv}_before_complaint"] = out[f"n_{lv}"]
    return out.reset_index()


@pytest.fixture(scope="module")
def fitted(small_data):
    from gmethods.estimators import LTMLE

    model = LTMLE(learner="linear", treatment_learner="hgb", learner_params=FAST)
    return model.fit(small_data.panel)


class TestLTMLE:
    def test_natural_course_equals_observed(self, fitted, small_data):
        table = fitted.estimate([iv.natural()])
        observed = small_data.chats["complained"].mean()
        assert table.risks["natural_course"] == pytest.approx(observed, abs=2e-3)

    def test_intervals_cover_estimate(self, fitted):
        table = fitted.estimate(REGIMES)
        for name in ("never_escalation", "never_any"):
            low, high = table.ci[name]
            assert low <= table.rd(name) <= high
        assert fitted.se_["never_any"] > 0
        assert fitted.se_["natural_course"] == 0.0

    def test_ice_has_no_intervals(self, small_data):
        from gmethods.estimators import ICE

        table = ICE(learner="linear").fit(small_data.panel).estimate(REGIMES)
        assert table.label == "ice" and not table.ci
        assert table.rd("never_any") < 0

    def test_null_scenario_near_zero(self):
        from gmethods.estimators import LTMLE

        sim = ChatSimulator(Scenario.preset("null", n_chats=4_000, seed=3))
        data = sim.simulate()
        table = (
            LTMLE(learner="linear", treatment_learner="hgb", learner_params=FAST)
            .fit(data.panel)
            .estimate(REGIMES)
        )
        for name in ("never_escalation", "never_any"):
            low, high = table.ci[name]
            assert low < 0.002 and high > -0.002

    def test_fluctuation_guard(self):
        from gmethods.estimators.ltmle import _fluctuate

        z = np.zeros(50)
        assert _fluctuate(z, np.full(50, 0.01), np.ones(50)) == 0.0
        z = np.r_[np.ones(20), np.zeros(80)]
        eps = _fluctuate(z, np.full(100, 0.1), np.ones(100))
        assert eps > 0


class TestStack:
    def test_stacked_classifier(self, small_data):
        from gmethods.estimators._features import make_learner

        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 3))
        y = (X[:, 0] + rng.normal(size=600) > 0).astype(int)
        est = make_learner("stack", "classify", [False] * 3, params=FAST).fit(X, y)
        p = est.predict_proba(X)
        assert p.shape == (600, 2) and np.allclose(p.sum(axis=1), 1)

    def test_stacked_regressor_weights_non_negative(self):
        from gmethods.estimators._features import make_learner

        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 2))
        y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=500)
        est = make_learner("stack", "regress", [False] * 2, params=FAST).fit(X, y)
        assert (est.weights_ >= 0).all()
        assert np.corrcoef(est.predict(X), y)[0, 1] > 0.8


def test_ltmle_weight_summary(fitted):
    from gmethods.estimators import ltmle_weight_summary

    out = ltmle_weight_summary(fitted, [iv.never("escalation"), iv.never_any()])
    assert (out["max"] >= out["p999"]).all()
    assert out.loc["never_any", "ess_share"] <= 1.0
    assert fitted.truncate == 0.999


def test_shap_drivers_for_chats(small_data):
    pytest.importorskip("shap")
    from gmethods.estimators import ShapDrivers

    out = ShapDrivers.for_chats().fit(small_data.chats).drivers(sample=300)
    assert out["share"].sum() == pytest.approx(1.0)
    assert list(out.index) == ["transfer", "escalation", "requeue", "shift_change"]
