"""Chat handoff simulator with known causal effects.

A chat is a sequence of turns. Within each turn, in this order:

1. Confounders update: queue length, the wait going into the turn (which
   includes queue time from the previous turn's handoff), the agent's shift
   time left, frustration and its sentiment proxy, and customer idle time.
2. A handoff may happen: none, transfer, escalation, requeue or shift change.
3. A complaint may happen.
4. The chat may end (resolved or abandoned). It cannot end at a turn with a
   handoff, because a new agent has just picked it up.

Handoffs feed back into later confounders: they add queue time, raise
frustration and reset resolution progress. Those confounders then drive
later handoffs and complaints. That feedback is time-varying confounding
affected by earlier treatment.

The same step code produces observed data and counterfactual data. All
random numbers are drawn up front for each batch of chats in a fixed order,
so every intervention sees the same draws (common random numbers).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..interventions import HANDOFF_TYPES, NONE, Intervention, natural
from ..results import RiskTable
from .scenario import TASK_TYPES, TEAMS, Scenario

TRANSFER, ESCALATION, REQUEUE, SHIFT = 1, 2, 3, 4
SPECIALIST = TEAMS.index("specialist")
FRAUD = TASK_TYPES.index("fraud")
N_LEVELS = len(HANDOFF_TYPES)
BASE_DATE = np.datetime64("2026-01-05T00:00")  # a Monday

PANEL_BASELINE = [
    "task_type",
    "first_agent_team",
    "friction_history",
    "start_hour",
    "weekday",
    "n_products",
    "age_band",
    "income_band",
]
"""Baseline (fixed) confounder columns in the panel."""

PANEL_TIME_VARYING = [
    "queue_length",
    "wait",
    "shift_time_left",
    "skill_mismatch",
    "with_specialist",
    "sentiment",
    "idle",
]
"""Time-varying confounder columns, in their order within a turn."""


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _by_level(d: dict) -> np.ndarray:
    return np.array([d[k] for k in HANDOFF_TYPES], dtype=float)


@dataclass
class SimData:
    """Output of :meth:`ChatSimulator.simulate`.

    Attributes
    ----------
    panel:
        One row per chat-turn while the chat is open and before any
        complaint. The row of a complaint or chat end is the last row.
    event_log:
        Raw events with timestamps, as a support platform would log them.
        Includes handoffs logged after a complaint.
    chats:
        One row per chat with counts. What a naive analysis would use.
    latent:
        Unobserved variables by chat-turn (frustration, resolution progress,
        temperament). For teaching and checks only.
    scenario:
        The scenario that generated the data.
    """

    panel: pd.DataFrame
    event_log: pd.DataFrame
    chats: pd.DataFrame
    latent: pd.DataFrame
    scenario: Scenario


class _Baseline:
    """Baseline covariates and per-chat constants for one batch."""

    def __init__(self, sc: Scenario, rng: np.random.Generator, n: int, id_offset: int):
        # Customers, each with one or more chats sharing customer traits.
        n_cust_guess = int(n / (1 + sc.repeat_rate)) + 10
        per_cust = 1 + rng.poisson(sc.repeat_rate, size=n_cust_guess)
        while per_cust.sum() < n:
            per_cust = np.concatenate([per_cust, 1 + rng.poisson(sc.repeat_rate, 100)])
        cust = np.repeat(np.arange(len(per_cust)), per_cust)[:n]
        n_cust = cust.max() + 1
        self.customer_id = cust + id_offset
        self.chat_id = np.arange(n) + id_offset

        u_cust = rng.standard_normal(n_cust)
        age_cust = rng.choice(5, size=n_cust, p=[0.2, 0.3, 0.25, 0.15, 0.1])
        income_cust = rng.choice(4, size=n_cust, p=[0.25, 0.35, 0.25, 0.15])
        prod_cust = 1 + rng.binomial(3, 0.3 + 0.08 * income_cust)
        self.U = u_cust[cust]
        self.age_band = age_cust[cust]
        self.income_band = income_cust[cust]
        self.n_products = prod_cust[cust]

        self.task = rng.choice(len(TASK_TYPES), size=n, p=np.array(sc.task_mix))
        self.complexity = np.array(sc.task_complexity)[self.task]
        self.home_team = np.array(sc.task_home_team)[self.task]
        misroute = rng.random(n) < np.array(sc.misroute_rate)[self.task]
        self.first_team = np.where(
            misroute, _other_team(self.home_team, rng.random(n)), self.home_team
        )
        rate = np.array(sc.friction_rate)[self.task] * (1 + 0.2 * (self.n_products - 1))
        self.friction = rng.poisson(rate)

        hour_w = np.array(
            [0.2] * 7
            + [1, 2, 3, 3.5, 3.5, 3, 3, 3, 3, 2.8, 2.5]
            + [2, 1.6, 1.2, 0.8, 0.5, 0.3]
        )
        self.start_hour = rng.choice(24, size=n, p=hour_w / hour_w.sum())
        self.start_minute = rng.random(n) * 60.0
        self.weekday = rng.choice(7, size=n, p=[0.17, 0.16, 0.16, 0.16, 0.15, 0.1, 0.1])

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "chat_id": self.chat_id,
                "customer_id": self.customer_id,
                "task_type": pd.Categorical.from_codes(self.task, TASK_TYPES),
                "first_agent_team": pd.Categorical.from_codes(self.first_team, TEAMS),
                "friction_history": self.friction,
                "start_hour": self.start_hour,
                "weekday": self.weekday,
                "n_products": self.n_products,
                "age_band": self.age_band,
                "income_band": self.income_band,
            }
        )


def _other_team(home: np.ndarray, u: np.ndarray) -> np.ndarray:
    """A team among the first four that is not ``home``, chosen by ``u``."""
    shift = 1 + np.floor(u * 3).astype(int)
    return (home + shift) % 4


class _Draws:
    """All per-turn random numbers for one batch, drawn in a fixed order."""

    FIELDS = (
        "z_queue",
        "g_wait",
        "z_frust",
        "z_sent",
        "g_idle",
        "g_reply",
        "u_handoff",
        "u_interv",
        "u_shift_new",
        "u_route",
        "u_route_team",
        "u_y",
        "u_d",
    )

    def __init__(self, sc: Scenario, rng: np.random.Generator, n: int):
        t = sc.t_max
        self.z_queue = rng.standard_normal((t, n))
        self.g_wait = rng.gamma(sc.wait_shape, 1.0 / sc.wait_shape, (t, n))
        self.z_frust = rng.standard_normal((t, n))
        self.z_sent = rng.standard_normal((t, n))
        self.g_idle = rng.gamma(sc.idle_shape, 1.0 / sc.idle_shape, (t, n))
        self.g_reply = rng.gamma(2.0, 0.5, (t, n))
        self.u_handoff = rng.random((t, n))
        self.u_interv = rng.random((t, n))
        self.u_shift_new = rng.random((t + 1, n))
        self.u_route = rng.random((t, n))
        self.u_route_team = rng.random((t, n))
        self.u_y = rng.random((t, n))
        self.u_d = rng.random((t, n))


def _queue_mu(sc: Scenario, hour: np.ndarray) -> np.ndarray:
    day = np.clip(np.sin(np.pi * (hour - 6.0) / 16.0), 0.0, None)
    return sc.queue_mean * (0.6 + sc.queue_peak * day)


class ChatSimulator:
    """Simulate chats under the natural course or any intervention.

    Parameters
    ----------
    scenario:
        The data-generating process. Defaults to the ``baseline`` preset.

    Examples
    --------
    >>> from gmethods import ChatSimulator, Scenario, interventions as iv
    >>> sim = ChatSimulator(Scenario.preset("baseline", n_chats=2_000))
    >>> data = sim.simulate()
    >>> truth = sim.true_risks([iv.natural(), iv.never("escalation")], n_mc=20_000)
    """

    def __init__(self, scenario: Scenario | None = None):
        self.scenario = scenario or Scenario()
        sc = self.scenario
        self._gamma = _by_level(sc.gamma)
        self._delta = _by_level(sc.delta)
        self._queue_time = _by_level(sc.queue_time)
        self._keep = _by_level(sc.progress_keep)

    # ------------------------------------------------------------------ core
    def _run(
        self,
        base: _Baseline,
        draws: _Draws,
        intervention: Intervention,
        record: bool,
    ):
        """Advance every chat in the batch through all turns.

        The path of confounders and handoffs never depends on whether a
        complaint or chat end has happened, because those only stop the chat.
        So the full path is generated for every chat. Returns the hazard
        arrays (and per-turn records when ``record`` is true).
        """
        sc = self.scenario
        n = len(base.chat_id)
        T = sc.t_max
        mu_q = _queue_mu(sc, base.start_hour)
        sd_q = 0.25 * sc.queue_mean

        queue = mu_q + sd_q * draws.z_queue[0]
        F = np.zeros(n)
        progress = np.zeros(n)
        prev_a = np.zeros(n, dtype=int)
        team = base.first_team.copy()
        shift_left = sc.shift_left_range[0] + draws.u_shift_new[0] * (
            sc.shift_left_range[1] - sc.shift_left_range[0]
        )
        cum = np.zeros((n, N_LEVELS), dtype=int)
        turns_since = np.zeros(n, dtype=int)
        last_reply = np.zeros(n)
        last_idle = np.zeros(n)
        elapsed = np.zeros(n)

        h_y = np.empty((T, n))
        h_d = np.empty((T, n))
        recs = [] if record else None
        lo, hi = sc.shift_left_range

        for t in range(T):
            # 1. Time-varying confounders, going into the turn.
            if t > 0:
                queue = (
                    mu_q
                    + sc.queue_ar * (queue - mu_q)
                    + sd_q * np.sqrt(1 - sc.queue_ar**2) * draws.z_queue[t]
                )
            queue = np.clip(queue, 0.0, None)
            queue_scale = queue / sc.queue_mean
            wait_mean = 0.3 + sc.base_wait_per_queue * queue
            wait_mean = wait_mean + self._queue_time[prev_a] * queue_scale
            wait = wait_mean * draws.g_wait[t]

            new_agent = prev_a != NONE
            fresh = lo + draws.u_shift_new[t] * (hi - lo)
            if t > 0:
                kept = shift_left - last_reply - last_idle - wait
                shift_left = np.where(new_agent & (prev_a != SHIFT), fresh, kept)
                # A shift change hands over to an agent starting a full window.
                shift_left = np.where(prev_a == SHIFT, fresh, shift_left)
            mismatch = (team != base.home_team) & (team != SPECIALIST)

            F = (
                sc.rho * F
                + sc.beta_wait * np.log1p(wait)
                + self._gamma[prev_a]
                + sc.beta_complexity_f * base.complexity
                + sc.beta_friction_f * base.friction
                + sc.lambda_f * base.U
                + sc.frustration_sd * draws.z_frust[t]
            )
            sentiment = F + sc.sentiment_noise * draws.z_sent[t]
            idle_mean = sc.idle_mean * np.exp(
                sc.idle_frustration * (F - 2.0) + sc.idle_age * (base.age_band - 2)
            )
            idle = idle_mean * draws.g_idle[t]
            reply = sc.reply_mean * draws.g_reply[t]

            progress = self._keep[prev_a] * progress + np.where(
                mismatch, sc.progress_gain_mismatch, sc.progress_gain_match
            )

            # 2. Handoff: shift change first, then a softmax over the rest.
            p_shift = (1 - sc.p_stay) * _sigmoid(
                -(shift_left - idle) / sc.shift_softness
            )
            eta_tr = (
                sc.c_transfer + sc.b_mismatch * mismatch + sc.b_transfer_early * (t < 3)
            )
            eta_esc = (
                sc.c_escalation
                + sc.b_frustration_esc * F
                + sc.b_complexity_esc * base.complexity
                + sc.b_friction_esc * base.friction
                + sc.escalation_determinism * (base.task == FRAUD)
                + sc.b_specialist_esc * (team == SPECIALIST)
            )
            eta_rq = (
                sc.c_requeue
                + sc.b_idle_requeue * np.log(np.maximum(idle, 1e-6) / sc.idle_threshold)
                + sc.b_queue_requeue * (queue - sc.queue_mean)
            )
            eta = np.stack([np.zeros(n), eta_tr, eta_esc, eta_rq])
            eta -= eta.max(axis=0)
            soft = np.exp(eta)
            soft /= soft.sum(axis=0)
            probs = np.empty((N_LEVELS, n))
            probs[:4] = soft * (1 - p_shift)
            probs[SHIFT] = p_shift
            cdf = np.cumsum(probs, axis=0)
            a_nat = (draws.u_handoff[t][None, :] > cdf).sum(axis=0)
            a_nat = np.minimum(a_nat, N_LEVELS - 1)
            a = intervention.apply(a_nat, draws.u_interv[t], HANDOFF_TYPES, t=t + 1)

            # 3. Complaint and 4. chat end hazards.
            agents_so_far = 1 + cum[:, 1:].sum(axis=1)
            eta_y = (
                sc.alpha_0
                + sc.alpha_frustration * F
                + sc.alpha_friction * base.friction
                + sc.alpha_products * (base.n_products - 1)
                + sc.alpha_income * base.income_band
                + self._delta[a]
                + sc.delta_agents * (agents_so_far - 1)
                + sc.lambda_y * base.U
            )
            h_y[t] = _sigmoid(eta_y)
            eta_d = sc.d_0 + sc.d_progress * progress + sc.d_frustration * F
            blocked = (a != NONE) & sc.handoff_blocks_end
            h_d[t] = np.where(blocked, 0.0, _sigmoid(eta_d))

            if record:
                recs.append(
                    {
                        "queue_length": queue,
                        "wait": wait,
                        "shift_time_left": shift_left,
                        "skill_mismatch": mismatch.astype(int),
                        "with_specialist": (team == SPECIALIST).astype(int),
                        "sentiment": sentiment,
                        "idle": idle,
                        "agents_so_far": agents_so_far,
                        "turns_since_handoff": turns_since.copy(),
                        "prev_handoff": prev_a.copy(),
                        "cum": cum.copy(),
                        "handoff": a.copy(),
                        "frustration": F.copy(),
                        "progress": progress.copy(),
                        "reply": reply,
                        "elapsed_start": elapsed.copy(),
                    }
                )

            # Next-turn state.
            moved = a != NONE
            route_home = draws.u_route[t] >= np.array(sc.misroute_rate)[base.task]
            requeue_team = np.where(
                route_home,
                base.home_team,
                _other_team(base.home_team, draws.u_route_team[t]),
            )
            team = np.where(a == TRANSFER, base.home_team, team)
            team = np.where(a == ESCALATION, SPECIALIST, team)
            team = np.where(a == REQUEUE, requeue_team, team)
            cum[np.arange(n), a] += 1
            turns_since = np.where(moved, 0, turns_since + 1)
            prev_a = a
            last_reply, last_idle = reply, idle
            elapsed = elapsed + wait + reply + idle

        return h_y, h_d, recs

    def _batch(self, n: int, batch: int, seed: int, id_offset: int = 0):
        ss = np.random.SeedSequence([seed, batch])
        base_rng, draw_rng, event_rng = (np.random.default_rng(s) for s in ss.spawn(3))
        base = _Baseline(self.scenario, base_rng, n, id_offset)
        draws = _Draws(self.scenario, draw_rng, n)
        return base, draws, event_rng

    # ----------------------------------------------------------- simulate
    def simulate(
        self,
        n_chats: int | None = None,
        seed: int | None = None,
        intervention: Intervention | None = None,
    ) -> SimData:
        """Simulate observed data (or data under an intervention).

        Parameters
        ----------
        n_chats, seed:
            Override the scenario's values.
        intervention:
            Defaults to the natural course, which gives observed data.
        """
        sc = self.scenario
        n = n_chats or sc.n_chats
        seed = sc.seed if seed is None else seed
        intervention = intervention or natural()
        base, draws, event_rng = self._batch(n, 0, seed)
        h_y, h_d, recs = self._run(base, draws, intervention, record=True)
        y = draws.u_y < h_y
        d = (~y) & (draws.u_d < h_d)
        return _assemble(self.scenario, base, recs, y, d, event_rng)

    # -------------------------------------------------------------- truth
    def true_risks(
        self,
        interventions: Sequence[Intervention],
        n_mc: int = 200_000,
        batch_size: int = 50_000,
        seed: int = 12345,
    ) -> RiskTable:
        """Ground-truth risk of a complaint by chat end under each regime.

        Runs the structural equations with the treatment set by each
        intervention. Every regime sees the same chats and random numbers,
        so differences between regimes carry little Monte Carlo noise. Risks
        are accumulated from the hazards rather than from simulated events.
        """
        sc = self.scenario
        names = [iv.name for iv in interventions]
        if len(set(names)) != len(names):
            raise ValueError("intervention names must be unique")
        totals = {nm: np.zeros(sc.t_max) for nm in names}
        n_done = 0
        batch = 0
        while n_done < n_mc:
            n = min(batch_size, n_mc - n_done)
            base, draws, _ = self._batch(n, batch, seed)
            for iv in interventions:
                h_y, h_d, _ = self._run(base, draws, iv, record=False)
                totals[iv.name] += cumulative_incidence(h_y, h_d).sum(axis=1)
            n_done += n
            batch += 1
        curves = {nm: tot / n_done for nm, tot in totals.items()}
        return RiskTable(
            risks={nm: float(c[-1]) for nm, c in curves.items()},
            curves=curves,
            label="truth",
        )

    def with_scenario(self, **changes) -> ChatSimulator:
        """A new simulator with some scenario values changed."""
        return ChatSimulator(replace(self.scenario, **changes))


def cumulative_incidence(h_y: np.ndarray, h_d: np.ndarray) -> np.ndarray:
    """Cumulative incidence of the outcome by turn, per unit.

    ``h_y`` and ``h_d`` have shape (turns, units). The outcome is checked
    before the competing event within a turn. Returns shape (turns, units).
    """
    surv = np.cumprod((1 - h_y) * (1 - h_d), axis=0)
    prior = np.vstack([np.ones((1, h_y.shape[1])), surv[:-1]])
    return np.cumsum(prior * h_y, axis=0)


def _assemble(sc, base, recs, y, d, event_rng) -> SimData:
    """Turn per-turn arrays into the panel, event log, chat summary and latents."""
    T, n = y.shape
    stop = y | d
    # A chat's last row is its first complaint or end, else turn T.
    any_stop = stop.any(axis=0)
    last = np.where(any_stop, stop.argmax(axis=0), T - 1)
    complained = y[last, np.arange(n)] & any_stop
    ended = d[last, np.arange(n)] & any_stop

    keep = np.arange(T)[:, None] <= last[None, :]
    tt, ii = np.nonzero(keep)
    bframe = base.frame()

    def col(name):
        return np.stack([r[name] for r in recs])[tt, ii]

    cum = np.stack([r["cum"] for r in recs])[tt, ii]
    handoff = col("handoff")
    panel = bframe.iloc[ii].reset_index(drop=True)
    panel.insert(2, "turn", tt + 1)
    for name in PANEL_TIME_VARYING:
        panel[name] = col(name)
    panel["agents_so_far"] = col("agents_so_far")
    panel["turns_since_handoff"] = col("turns_since_handoff")
    panel["prev_handoff"] = pd.Categorical.from_codes(
        col("prev_handoff"), HANDOFF_TYPES
    )
    for k, level in enumerate(HANDOFF_TYPES[1:], start=1):
        panel[f"cum_{level}"] = cum[:, k]
    panel["handoff"] = pd.Categorical.from_codes(handoff, HANDOFF_TYPES)
    panel["complaint"] = y[tt, ii].astype(int)
    panel["ended"] = (d[tt, ii] & ~y[tt, ii]).astype(int)
    panel = panel.sort_values(["chat_id", "turn"], kind="stable").reset_index(drop=True)

    latent = (
        pd.DataFrame(
            {
                "chat_id": base.chat_id[ii],
                "turn": tt + 1,
                "frustration": col("frustration"),
                "progress": col("progress"),
                "temperament": base.U[ii],
            }
        )
        .sort_values(["chat_id", "turn"], kind="stable")
        .reset_index(drop=True)
    )

    event_log = _event_log(sc, base, recs, last, complained, ended, event_rng)
    chats = _chat_summary(bframe, panel, event_log, last, complained, ended)
    return SimData(
        panel=panel, event_log=event_log, chats=chats, latent=latent, scenario=sc
    )


def _event_log(sc, base, recs, last, complained, ended, rng) -> pd.DataFrame:
    """Raw timestamped events, including the post-complaint artefact."""
    n = len(base.chat_id)
    start = BASE_DATE + (
        base.weekday * 1440 + base.start_hour * 60 + base.start_minute
    ).astype("timedelta64[m]")
    rows_id, rows_ev, rows_type, rows_turn, rows_min = [], [], [], [], []

    def add(mask, event, kind, turn, minutes):
        idx = np.nonzero(mask)[0]
        rows_id.append(base.chat_id[idx])
        rows_ev.append(np.full(len(idx), event, dtype=object))
        rows_type.append(
            np.asarray(kind, dtype=object)[idx]
            if np.ndim(kind)
            else np.full(len(idx), kind, dtype=object)
        )
        rows_turn.append(np.broadcast_to(turn, n)[idx])
        rows_min.append(np.broadcast_to(minutes, n)[idx])

    add(np.ones(n, bool), "chat_start", None, 0, np.zeros(n))
    end_minutes = np.zeros(n)
    for t, r in enumerate(recs):
        alive = t <= last
        t_reply = r["elapsed_start"] + r["wait"] + r["reply"]
        t_close = t_reply + r["idle"]
        add(alive, "agent_reply", None, t + 1, t_reply)
        moved = alive & (r["handoff"] != NONE)
        add(
            moved,
            "handoff",
            np.array(HANDOFF_TYPES, dtype=object)[r["handoff"]],
            t + 1,
            t_close,
        )
        at_last = t == last
        add(at_last & complained, "complaint", None, t + 1, t_close + 0.5)
        end_minutes = np.where(at_last, t_close, end_minutes)

    # Raw-log artefact: escalations logged after a complaint.
    post = complained & (rng.random(n) < sc.p_post)
    add(post, "handoff", "escalation", last + 1, end_minutes + 5.0)
    end_at = end_minutes + np.where(complained, 10.0, 0.0)
    add(np.ones(n, bool), "chat_end", None, last + 1, end_at)

    minutes = np.concatenate(rows_min)
    log = pd.DataFrame(
        {
            "chat_id": np.concatenate(rows_id),
            "event": np.concatenate(rows_ev),
            "handoff_type": np.concatenate(rows_type),
            "turn": np.concatenate(rows_turn).astype(int),
            "minutes": minutes,
        }
    )
    cust = pd.Series(base.customer_id, index=base.chat_id)
    log.insert(1, "customer_id", cust.loc[log["chat_id"]].to_numpy())
    st = pd.Series(start, index=base.chat_id)
    log["timestamp"] = st.loc[log["chat_id"]].to_numpy() + pd.to_timedelta(
        log["minutes"], unit="min"
    )
    order = {
        "chat_start": 0,
        "agent_reply": 1,
        "handoff": 2,
        "complaint": 3,
        "chat_end": 4,
    }
    log["_o"] = log["event"].map(order)
    log = log.sort_values(["chat_id", "minutes", "_o"], kind="stable")
    return log.drop(columns=["_o", "minutes"]).reset_index(drop=True)


def _chat_summary(bframe, panel, event_log, last, complained, ended) -> pd.DataFrame:
    """One row per chat, with handoff counts from the raw log."""
    chats = bframe.copy()
    chats["n_turns"] = last + 1
    chats["complained"] = complained.astype(int)
    chats["resolved_or_abandoned"] = ended.astype(int)
    ho = event_log[event_log["event"] == "handoff"]
    comp_min = event_log[event_log["event"] == "complaint"].set_index("chat_id")[
        "timestamp"
    ]
    ho = ho.assign(after_complaint=ho["timestamp"] > ho["chat_id"].map(comp_min))
    for level in HANDOFF_TYPES[1:]:
        sub = ho[ho["handoff_type"] == level]
        chats[f"n_{level}"] = (
            sub.groupby("chat_id")
            .size()
            .reindex(chats["chat_id"], fill_value=0)
            .to_numpy()
        )
        chats[f"n_{level}_before_complaint"] = (
            sub[~sub["after_complaint"]]
            .groupby("chat_id")
            .size()
            .reindex(chats["chat_id"], fill_value=0)
            .to_numpy()
        )
    n_cols = [f"n_{lv}" for lv in HANDOFF_TYPES[1:]]
    pre_cols = [f"n_{lv}_before_complaint" for lv in HANDOFF_TYPES[1:]]
    chats["total_agents"] = 1 + chats[n_cols].sum(axis=1)
    chats["agents_before_complaint_or_all"] = 1 + chats[pre_cols].sum(axis=1)
    chats["mean_sentiment"] = (
        panel.groupby("chat_id")["sentiment"]
        .mean()
        .reindex(chats["chat_id"])
        .to_numpy()
    )
    return chats
