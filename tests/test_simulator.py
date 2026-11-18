import numpy as np
import pandas as pd
import pytest

from gmethods import ChatSimulator, Scenario
from gmethods import interventions as iv
from gmethods.sim import PANEL_BASELINE, PANEL_TIME_VARYING, cumulative_incidence


class TestPanel:
    def test_columns(self, small_data):
        for col in [
            *PANEL_BASELINE,
            *PANEL_TIME_VARYING,
            "handoff",
            "complaint",
            "ended",
        ]:
            assert col in small_data.panel.columns

    def test_turns_are_consecutive(self, small_data):
        p = small_data.panel
        g = p.groupby("chat_id")["turn"]
        assert (g.min() == 1).all()
        assert (g.max() == g.count()).all()

    def test_no_rows_after_complaint_or_end(self, small_data):
        p = small_data.panel
        stop = p["complaint"] + p["ended"]
        assert (stop <= 1).all()
        after = stop.groupby(p["chat_id"]).cumsum() - stop
        assert (after == 0).all()

    def test_every_chat_ends_or_complains(self, small_data):
        last = small_data.panel.groupby("chat_id").tail(1)
        reached_max = last["turn"] == small_data.scenario.t_max
        assert ((last["complaint"] + last["ended"] == 1) | reached_max).all()

    def test_no_end_at_a_handoff_turn(self, small_data):
        p = small_data.panel
        assert not ((p["ended"] == 1) & (p["handoff"] != "none")).any()

    def test_history_columns_match_handoffs(self, small_data):
        p = small_data.panel
        moved = (p["handoff"] != "none").astype(int)
        before = moved.groupby(p["chat_id"]).cumsum() - moved
        assert (p["agents_so_far"] == 1 + before).all()

    def test_specialist_after_escalation(self, small_data):
        p = small_data.panel
        assert (p.loc[p["prev_handoff"] == "escalation", "with_specialist"] == 1).all()
        assert (p.loc[p["prev_handoff"] == "transfer", "with_specialist"] == 0).all()

    def test_waits_and_idle_non_negative(self, small_data):
        p = small_data.panel
        assert (p[["wait", "idle", "queue_length"]] >= 0).all().all()


class TestReproducibility:
    def test_same_seed_same_data(self):
        sim = ChatSimulator(Scenario.preset("baseline", n_chats=500))
        a, b = sim.simulate(seed=3), sim.simulate(seed=3)
        pd.testing.assert_frame_equal(a.panel, b.panel)
        pd.testing.assert_frame_equal(a.event_log, b.event_log)

    def test_different_seed_differs(self):
        sim = ChatSimulator(Scenario.preset("baseline", n_chats=500))
        assert not sim.simulate(seed=3).panel.equals(sim.simulate(seed=4).panel)

    def test_golden_values(self):
        data = ChatSimulator(
            Scenario.preset("baseline", n_chats=2_000, seed=11)
        ).simulate()
        assert len(data.panel) == GOLDEN["rows"]
        assert int(data.chats["complained"].sum()) == GOLDEN["complaints"]
        assert int((data.panel["handoff"] != "none").sum()) == GOLDEN["handoffs"]


GOLDEN = {"rows": 13_558, "complaints": 36, "handoffs": 1_046}


class TestInterventions:
    def test_only_downstream_changes(self):
        sim = ChatSimulator(Scenario.preset("baseline", n_chats=1_000))
        nat = sim.simulate(seed=5).panel
        none = sim.simulate(seed=5, intervention=iv.never_any()).panel
        first = (
            nat[nat["handoff"] != "none"]
            .groupby("chat_id")["turn"]
            .min()
            .reindex(nat["chat_id"].unique(), fill_value=10**6)
        )
        merged = nat.merge(none, on=["chat_id", "turn"], suffixes=("", "_iv"))
        first_turn = merged["chat_id"].map(first)
        # Confounders match up to and including the first handoff turn.
        # Outcomes match strictly before it: the handoff itself moves them.
        checks = [
            (PANEL_TIME_VARYING, merged["turn"] <= first_turn),
            (["complaint", "ended"], merged["turn"] < first_turn),
        ]
        for cols, rows in checks:
            for c in cols:
                np.testing.assert_allclose(
                    merged.loc[rows, c].to_numpy(float),
                    merged.loc[rows, f"{c}_iv"].to_numpy(float),
                )
        assert (none["handoff"] == "none").all()

    def test_never_any_has_no_handoffs_in_log(self):
        sim = ChatSimulator(Scenario.preset("baseline", n_chats=500, p_post=0.0))
        log = sim.simulate(intervention=iv.never_any()).event_log
        assert (log["event"] != "handoff").all()


class TestTruth:
    def test_null_scenario_has_no_effects(self):
        sim = ChatSimulator(Scenario.preset("null"))
        truth = sim.true_risks(iv.standard_set(), n_mc=5_000).to_frame()
        assert np.allclose(truth["rd"], 0.0, atol=1e-12)

    def test_natural_matches_observed_rate(self):
        sim = ChatSimulator(Scenario.preset("baseline"))
        truth = sim.true_risks([iv.natural()], n_mc=30_000)
        data = sim.simulate(n_chats=30_000, seed=99)
        rate = data.chats["complained"].mean()
        se = np.sqrt(rate * (1 - rate) / 30_000 + rate * (1 - rate) / 30_000)
        assert abs(truth.risks["natural_course"] - rate) < 4 * se

    def test_removing_handoffs_lowers_risk(self):
        sim = ChatSimulator(Scenario.preset("baseline"))
        truth = sim.true_risks([iv.natural(), iv.never_any()], n_mc=10_000)
        assert truth.rd("never_any") < 0

    def test_curves_are_monotone(self):
        sim = ChatSimulator(Scenario.preset("baseline"))
        truth = sim.true_risks([iv.natural()], n_mc=5_000)
        assert (np.diff(truth.curves["natural_course"]) >= 0).all()

    def test_unique_names(self):
        sim = ChatSimulator(Scenario.preset("baseline"))
        with pytest.raises(ValueError):
            sim.true_risks([iv.natural(), iv.natural()], n_mc=10)


class TestCumulativeIncidence:
    def test_single_turn(self):
        out = cumulative_incidence(np.array([[0.2]]), np.array([[0.5]]))
        assert out[0, 0] == pytest.approx(0.2)

    def test_competing_event_removes_risk(self):
        h_y = np.array([[0.1], [0.1]])
        h_d = np.array([[0.5], [0.0]])
        out = cumulative_incidence(h_y, h_d)
        assert out[-1, 0] == pytest.approx(0.1 + 0.9 * 0.5 * 0.1)


class TestEventLog:
    def test_rebuilt_from_log(self, small_data):
        log, panel = small_data.event_log, small_data.panel
        last_turn = panel.groupby("chat_id")["turn"].max()
        replies = log[log["event"] == "agent_reply"].groupby("chat_id").size()
        assert (replies.reindex(last_turn.index) == last_turn).all()
        comp = log[log["event"] == "complaint"].set_index("chat_id")["turn"]
        comp_panel = panel[panel["complaint"] == 1].set_index("chat_id")["turn"]
        pd.testing.assert_series_equal(
            comp.sort_index(), comp_panel.sort_index(), check_names=False
        )

    def test_handoffs_before_complaint_match_panel(self, small_data):
        log, panel = small_data.event_log, small_data.panel
        pre = log[(log["event"] == "handoff")]
        comp_time = log[log["event"] == "complaint"].set_index("chat_id")["timestamp"]
        after = pre["timestamp"] > pre["chat_id"].map(comp_time)
        in_log = pre[~after].groupby("chat_id").size()
        in_panel = panel[panel["handoff"] != "none"].groupby("chat_id").size()
        pd.testing.assert_series_equal(
            in_log.sort_index(), in_panel.sort_index(), check_names=False
        )

    def test_timestamps_increase_within_chat(self, small_data):
        log = small_data.event_log
        diffs = log.groupby("chat_id")["timestamp"].diff().dropna()
        assert (diffs >= pd.Timedelta(0)).all()


class TestChatSummary:
    def test_one_row_per_chat(self, small_data):
        assert small_data.chats["chat_id"].is_unique
        assert len(small_data.chats) == small_data.panel["chat_id"].nunique()

    def test_total_agents_includes_post_complaint(self, small_data):
        c = small_data.chats
        assert (c["total_agents"] >= c["agents_before_complaint_or_all"]).all()
        assert (
            c.loc[c["complained"] == 0, "total_agents"]
            == c.loc[c["complained"] == 0, "agents_before_complaint_or_all"]
        ).all()

    def test_customers_share_traits(self, small_data):
        c = small_data.chats
        assert (c.groupby("customer_id")["age_band"].nunique() == 1).all()
        assert c["customer_id"].nunique() < len(c)
