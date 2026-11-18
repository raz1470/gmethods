"""Tiny SVG helpers that match the article's house style (viewBox 340 x 200 panels)."""

from __future__ import annotations

import numpy as np

MUTED, GRID, INK = "#6b7280", "#e5e7eb", "#111827"


def _nice_ticks(lo, hi, n=4):
    span = hi - lo
    step = 10 ** np.floor(np.log10(span / n))
    for m in [1, 2, 2.5, 5, 10]:
        if span / (step * m) <= n + 1:
            step *= m
            break
    start = np.ceil(lo / step) * step
    return [round(v, 6) for v in np.arange(start, hi + 1e-9, step)]


def _fmt(v):
    if abs(v) < 1e-9:
        return "0"
    return f"{v:g}" if abs(v) >= 1 or v == 0 else f"{v:.2g}"


def path(xs, ys):
    return "M" + "L".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def line_panel(
    x,
    series,
    title,
    sub="",
    ylim=None,
    xlabel="Treatment a",
    w=340,
    h=200,
    xticks=None,
    vlines=(),
):
    """series: list of dicts with y, color, and optional dash, width, lo, hi, opacity."""
    x = np.asarray(x, float)
    L, R, T, B = 36, w - 14, 10, h - 34
    ys = [np.asarray(s["y"], float) for s in series]
    ys += [np.asarray(s[k], float) for s in series for k in ("lo", "hi") if k in s]
    ymin, ymax = ylim if ylim else (min(v.min() for v in ys), max(v.max() for v in ys))
    pad = 0.05 * (ymax - ymin)
    ymin, ymax = ymin - pad, ymax + pad
    sx = lambda v: L + (v - x.min()) / (x.max() - x.min()) * (R - L)  # noqa: E731
    sy = lambda v: B - (np.clip(v, ymin, ymax) - ymin) / (ymax - ymin) * (B - T)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(ymin, ymax):
        out.append(
            f'<line x1="{L}" x2="{R}" y1="{sy(t):.2f}" y2="{sy(t):.2f}" stroke="{GRID}"/>'
            f'<text x="{L - 6}" y="{sy(t) + 4:.2f}" text-anchor="end" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    for t in xticks or _nice_ticks(x.min(), x.max(), 4):
        if x.min() - 1e-9 <= t <= x.max() + 1e-9:
            out.append(
                f'<text x="{sx(t):.2f}" y="{B + 16}" text-anchor="middle" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
            )
    for v, lab in vlines:
        out.append(
            f'<line x1="{sx(v):.2f}" x2="{sx(v):.2f}" y1="{T}" y2="{B}" stroke="{MUTED}" stroke-dasharray="3 3"/>'
            f'<text x="{sx(v) + 4:.2f}" y="{T + 10}" font-size="10" fill="{MUTED}">{lab}</text>'
        )
    out.append(
        f'<text x="{(L + R) / 2}" y="{h - 4}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text>'
    )
    for s in series:
        if "lo" in s:
            lo, hi = np.asarray(s["lo"]), np.asarray(s["hi"])
            pts = (
                path(sx(x), sy(hi))
                + "L"
                + "L".join(
                    f"{a:.2f},{b:.2f}" for a, b in zip(sx(x[::-1]), sy(lo[::-1]))
                )
                + "Z"
            )
            out.append(f'<path d="{pts}" fill="{s["color"]}" opacity="0.15"/>')
    for s in series:
        out.append(
            f'<path d="{path(sx(x), sy(np.asarray(s["y"])))}" fill="none" stroke="{s["color"]}" '
            f'stroke-width="{s.get("width", 2.2)}" stroke-dasharray="{s.get("dash", "")}" '
            f'opacity="{s.get("opacity", 1)}" stroke-linejoin="round"/>'
        )
    out.append("</svg>")
    head = f'<div class="panel-title">{title}</div>' + (
        f'<div class="panel-sub">{sub}</div>' if sub else ""
    )
    return f"<div>{head}{''.join(out)}</div>"


def legend(items):
    """items: (label, color, dash)."""
    spans = []
    for lab, col, dash in items:
        spans.append(
            f'<span class="li"><svg width="20" height="4"><line x1="0" y1="2" x2="20" y2="2" '
            f'stroke="{col}" stroke-width="2.5" stroke-dasharray="{dash}"/></svg>{lab}</span>'
        )
    return f'<div class="legend">{"".join(spans)}</div>'


def grouped_bars(
    categories, groups, w=680, h=260, ylabel="", ref=None, fmt=None, vmax=None
):
    """groups: list of (label, color, values). ref: (value, label) dashed reference line."""
    L, R, T, B = 48, w - 10, 12, h - 46
    fmt = fmt or (lambda v: f"{v:.2f}")
    vmax = vmax or max(max(v) for _, _, v in groups) * 1.12
    sy = lambda v: B - v / vmax * (B - T)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(0, vmax):
        out.append(
            f'<line x1="{L}" x2="{R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{GRID}"/>'
            f'<text x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    nc, ng = len(categories), len(groups)
    cw = (R - L) / nc
    bw = min(34, cw * 0.8 / ng)
    for i, cat in enumerate(categories):
        cx = L + cw * (i + 0.5)
        for j, (lab, col, vals) in enumerate(groups):
            x0 = cx + (j - ng / 2) * (bw + 2)
            y0 = sy(vals[i])
            out.append(
                f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{B - y0:.1f}" rx="3" fill="{col}">'
                f"<title>{lab}: {vals[i]:.2f}</title></rect>"
                f'<text x="{x0 + bw / 2:.1f}" y="{y0 - 4:.1f}" text-anchor="middle" font-size="10" fill="{MUTED}">{fmt(vals[i])}</text>'
            )
        for k, part in enumerate(cat.split("\n")):
            out.append(
                f'<text x="{cx:.1f}" y="{B + 16 + 13 * k}" text-anchor="middle" font-size="11" fill="{INK if k == 0 else MUTED}">{part}</text>'
            )
    out.append(
        f'<text x="14" y="{(T + B) / 2}" text-anchor="middle" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {(T + B) / 2})">{ylabel}</text>'
    )
    if ref is not None:
        out.append(
            f'<line x1="{L}" x2="{R}" y1="{sy(ref[0]):.1f}" y2="{sy(ref[0]):.1f}" stroke="{INK}" stroke-dasharray="5 4"/>'
            f'<text x="{R - 2}" y="{sy(ref[0]) - 5:.1f}" text-anchor="end" font-size="11" fill="{INK}">{ref[1]}</text>'
        )
    out.append(f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{MUTED}"/></svg>')
    return "".join(out)


def scatter(
    xs,
    ys,
    w=680,
    h=280,
    xlabel="",
    ylabel="",
    logx=False,
    ylim=None,
    refs=None,
    color="#eb6834",
):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    L, R, T, B = 48, w - 12, 12, h - 40
    fx = np.log10 if logx else (lambda v: v)
    xl, xh = fx(xs.min()) - 0.05, fx(xs.max()) + 0.05
    yl, yh = ylim if ylim else (ys.min(), ys.max())
    sx = lambda v: L + (fx(v) - xl) / (xh - xl) * (R - L)  # noqa: E731
    sy = lambda v: B - (np.clip(v, yl, yh) - yl) / (yh - yl) * (B - T)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(yl, yh, 5):
        out.append(
            f'<line x1="{L}" x2="{R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{GRID}"/>'
            f'<text x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    for t in [10, 20, 50, 100, 200, 500] if logx else _nice_ticks(xs.min(), xs.max()):
        if xs.min() <= t <= xs.max() * 1.05:
            out.append(
                f'<text x="{sx(t):.1f}" y="{B + 16}" text-anchor="middle" font-size="11" fill="{MUTED}">{t:g}</text>'
            )
    if refs is not None:
        lo, hi = refs
        out.append(
            f'<rect x="{L}" y="{sy(hi):.1f}" width="{R - L}" height="{sy(lo) - sy(hi):.1f}" fill="#111827" opacity="0.07"/>'
        )
    for x, y in zip(xs, ys):
        clipped = (ylim is not None) and (y < yl or y > yh)
        out.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{color}" opacity="{0.45 if clipped else 0.75}"/>'
        )
    out.append(
        f'<text x="{(L + R) / 2}" y="{h - 6}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text>'
    )
    out.append(
        f'<text x="14" y="{(T + B) / 2}" text-anchor="middle" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {(T + B) / 2})">{ylabel}</text>'
    )
    out.append("</svg>")
    return "".join(out)


def funnel_panel(
    n,
    err,
    title,
    sub="",
    color="#2a78d6",
    ylim=(-2, 2),
    w=340,
    h=210,
    xlabel="Observations in store (log scale)",
):
    """Error of a per-group estimate against group size (log x), one dot per group."""
    n, err = np.asarray(n, float), np.asarray(err, float)
    ok = np.isfinite(err)
    n, err = n[ok], err[ok]
    L, R, T, B = 36, w - 10, 10, h - 36
    xl, xh = np.log10(max(n.min() * 0.85, 1)), np.log10(n.max() * 1.15)
    yl, yh = ylim
    sx = lambda v: L + (np.log10(v) - xl) / (xh - xl) * (R - L)  # noqa: E731
    sy = lambda v: B - (np.clip(v, yl, yh) - yl) / (yh - yl) * (B - T)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(yl, yh, 4):
        col = MUTED if abs(t) < 1e-9 else GRID
        out.append(
            f'<line x1="{L}" x2="{R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{col}"'
            f"{' stroke-dasharray="3 3"' if abs(t) < 1e-9 else ''}/>"
            f'<text x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    for t in [10, 20, 50, 100, 200, 500]:
        if 10**xl <= t <= 10**xh:
            out.append(
                f'<text x="{sx(t):.1f}" y="{B + 16}" text-anchor="middle" font-size="11" fill="{MUTED}">{t}</text>'
            )
    for x, y in zip(n, err):
        clipped = y < yl or y > yh
        out.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6" fill="{color}" '
            f'opacity="{0.35 if clipped else 0.7}"/>'
        )
    out.append(
        f'<text x="{(L + R) / 2}" y="{h - 4}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text>'
    )
    out.append("</svg>")
    head = f'<div class="panel-title">{title}</div>' + (
        f'<div class="panel-sub">{sub}</div>' if sub else ""
    )
    return f"<div>{head}{''.join(out)}</div>"


def hist_panel(
    values,
    title,
    sub="",
    bins=30,
    color="#2a78d6",
    marks=(),
    shade=None,
    xlabel="Dose",
    w=340,
    h=200,
):
    """Histogram with optional vertical marks [(x, label, dash)] and a shaded x-range."""
    v = np.asarray(values, float)
    lo, hi = np.quantile(v, [0.001, 0.999])
    pad = 0.15 * (hi - lo + 1e-9)
    xs = [m[0] for m in marks] + ([shade[0], shade[1]] if shade else [])
    lo, hi = min([lo - pad] + xs), max([hi + pad] + xs)
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    L, R, T, B = 20, w - 10, 14, h - 34
    sx = lambda x: L + (x - lo) / (hi - lo) * (R - L)  # noqa: E731
    sy = lambda c: B - c / counts.max() * (B - T)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    if shade:
        out.append(
            f'<rect x="{sx(shade[0]):.1f}" y="{T}" width="{sx(shade[1]) - sx(shade[0]):.1f}" '
            f'height="{B - T}" fill="{color}" opacity="0.1"/>'
        )
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        if c:
            out.append(
                f'<rect x="{sx(e0) + 0.5:.1f}" y="{sy(c):.1f}" width="{max(sx(e1) - sx(e0) - 1, 0.5):.1f}" '
                f'height="{B - sy(c):.1f}" fill="{color}" opacity="0.75"/>'
            )
    for x, lab, dash in marks:
        out.append(
            f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{B}" stroke="{INK}" '
            f'stroke-width="1.6" stroke-dasharray="{dash}"/>'
            f'<text x="{sx(x) - 4:.1f}" y="{T + 10}" text-anchor="end" font-size="10" fill="{INK}">{lab}</text>'
        )
    for t in _nice_ticks(lo, hi, 4):
        out.append(
            f'<text x="{sx(t):.1f}" y="{B + 16}" text-anchor="middle" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    out.append(
        f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{MUTED}"/>'
        f'<text x="{(L + R) / 2}" y="{h - 4}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text></svg>'
    )
    head = f'<div class="panel-title">{title}</div>' + (
        f'<div class="panel-sub">{sub}</div>' if sub else ""
    )
    return f"<div>{head}{''.join(out)}</div>"


def shrink_panel(
    n,
    before,
    after,
    c_before,
    c_after,
    ylim,
    w=680,
    h=300,
    xlabel="Observations in store (log scale)",
    ylabel="",
):
    """Shrinkage plot: per group, a line from its no-pooling to its partial-pooling estimate."""
    n, before, after = (np.asarray(v, float) for v in (n, before, after))
    L, R, T, B = 44, w - 12, 10, h - 36
    xl, xh = np.log10(max(n.min() * 0.85, 1)), np.log10(n.max() * 1.15)
    yl, yh = ylim
    sx = lambda v: L + (np.log10(v) - xl) / (xh - xl) * (R - L)  # noqa: E731
    sy = lambda v: B - (np.clip(v, yl, yh) - yl) / (yh - yl) * (B - T)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(yl, yh, 5):
        out.append(
            f'<line x1="{L}" x2="{R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{GRID}"/>'
            f'<text x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    for t in [10, 20, 50, 100, 200, 500]:
        if 10**xl <= t <= 10**xh:
            out.append(
                f'<text x="{sx(t):.1f}" y="{B + 16}" text-anchor="middle" font-size="11" fill="{MUTED}">{t}</text>'
            )
    for x, a, b in zip(n, before, after):
        out.append(
            f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(a):.1f}" y2="{sy(b):.1f}" stroke="{MUTED}" '
            f'stroke-width="1" opacity="0.5"/>'
        )
    for x, a in zip(n, before):
        clipped = a < yl or a > yh
        out.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(a):.1f}" r="2.8" fill="{c_before}" opacity="{0.35 if clipped else 0.75}"/>'
        )
    for x, b in zip(n, after):
        out.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(b):.1f}" r="2.8" fill="{c_after}" opacity="0.85"/>'
        )
    out.append(
        f'<text x="{(L + R) / 2}" y="{h - 4}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text>'
    )
    if ylabel:
        out.append(
            f'<text x="12" y="{(T + B) / 2}" transform="rotate(-90 12 {(T + B) / 2})" text-anchor="middle" '
            f'font-size="11" fill="{MUTED}">{ylabel}</text>'
        )
    out.append("</svg>")
    return f"<div>{''.join(out)}</div>"


def paired_hbar_panel(
    labels,
    truth,
    est,
    est_lo,
    est_hi,
    notes,
    title,
    sub="",
    c_truth="#2a78d6",
    c_est="#eb6834",
    xmax=100,
    w=340,
    h=None,
    xlabel="% of the treatment effect",
):
    """Per category, a truth bar above an estimate bar with a whisker (est_lo to est_hi).

    notes: per category, a short label written after the estimate whisker, or "".
    """
    n = len(labels)
    row = 44
    h = h or 26 + n * row + 34
    L, R, T = 110, w - 150, 8
    B = T + n * row
    sx = lambda v: L + v / xmax * (R - L)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(0, xmax, 4):
        out.append(
            f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T}" y2="{B}" stroke="{GRID}"/>'
            f'<text x="{sx(t):.1f}" y="{B + 15}" text-anchor="middle" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    bh = 12
    for i, lab in enumerate(labels):
        y0 = T + i * row + 6
        out.append(
            f'<text x="{L - 6}" y="{y0 + bh + 4}" text-anchor="end" font-size="11" fill="{INK}">{lab}</text>'
        )
        out.append(
            f'<rect x="{L}" y="{y0}" width="{max(sx(truth[i]) - L, 1):.1f}" height="{bh}" rx="2" fill="{c_truth}">'
            f"<title>True share: {truth[i]:.0f}%</title></rect>"
        )
        y1 = y0 + bh + 2
        out.append(
            f'<rect x="{L}" y="{y1}" width="{max(sx(est[i]) - L, 1):.1f}" height="{bh}" rx="2" fill="{c_est}">'
            f"<title>SHAP share: {est[i]:.0f}% ({est_lo[i]:.0f}% to {est_hi[i]:.0f}%)</title></rect>"
        )
        yc = y1 + bh / 2
        out.append(
            f'<line x1="{sx(est_lo[i]):.1f}" x2="{sx(est_hi[i]):.1f}" y1="{yc:.1f}" y2="{yc:.1f}" '
            f'stroke="{INK}" stroke-width="1.2"/>'
        )
        for v in (est_lo[i], est_hi[i]):
            out.append(
                f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{yc - 3:.1f}" y2="{yc + 3:.1f}" stroke="{INK}" stroke-width="1.2"/>'
            )
        if notes[i]:
            out.append(
                f'<text x="{sx(max(est_hi[i], est[i])) + 5:.1f}" y="{yc + 4:.1f}" font-size="10" fill="{MUTED}">{notes[i]}</text>'
            )
    out.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{B}" stroke="{MUTED}"/>')
    out.append(
        f'<text x="{(L + R) / 2}" y="{h - 4}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text>'
    )
    out.append("</svg>")
    head = f'<div class="panel-title">{title}</div>' + (
        f'<div class="panel-sub">{sub}</div>' if sub else ""
    )
    return f"<div>{head}{''.join(out)}</div>"


def swatch_legend(items):
    """items: (label, color) shown as filled squares."""
    spans = [
        f'<span class="li"><svg width="12" height="12"><rect width="12" height="12" rx="2" fill="{c}"/></svg>{lab}</span>'
        for lab, c in items
    ]
    return f'<div class="legend">{"".join(spans)}</div>'


def dot_panel(
    labels,
    series,
    title,
    sub="",
    lo=-60,
    hi=220,
    ref=100,
    w=680,
    row=40,
    xlabel="Mean estimate, % of the true effect",
):
    """Rows of markers. series: list of (label, color, values); values in the same row order.

    Values outside [lo, hi] are drawn at the edge with a short note.
    """
    n, k = len(labels), len(series)
    L, R, T = 120, w - 16, 8
    B = T + n * row
    h = B + 34
    sx = lambda v: L + (min(max(v, lo), hi) - lo) / (hi - lo) * (R - L)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(lo, hi, 6):
        out.append(
            f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T}" y2="{B}" stroke="{GRID}"/>'
            f'<text x="{sx(t):.1f}" y="{B + 15}" text-anchor="middle" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    out.append(
        f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{T}" y2="{B}" stroke="{MUTED}"/>'
    )
    out.append(
        f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{T}" y2="{B}" stroke="{INK}" stroke-width="2"/>'
        f'<text x="{sx(ref) + 4:.1f}" y="{T + 9}" font-size="10" fill="{INK}">truth</text>'
    )
    for i, lab in enumerate(labels):
        yc = T + (i + 0.5) * row
        out.append(
            f'<text x="{L - 8}" y="{yc + 4:.1f}" text-anchor="end" font-size="11" fill="{INK}">{lab}</text>'
        )
        for j, (slab, col, vals) in enumerate(series):
            v = vals[i]
            y = yc + (j - (k - 1) / 2) * 9
            out.append(
                f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="4.5" fill="{col}" stroke="#fff" stroke-width="1.5">'
                f"<title>{slab}: {v:.0f}%</title></circle>"
            )
            if v < lo:
                out.append(
                    f'<text x="{sx(v) + 8:.1f}" y="{y + 3.5:.1f}" font-size="9.5" fill="{MUTED}">wrong sign ({v:,.0f}%)</text>'
                )
            elif v > hi:
                out.append(
                    f'<text x="{sx(v) - 8:.1f}" y="{y + 3.5:.1f}" text-anchor="end" font-size="9.5" fill="{MUTED}">{v:,.0f}%</text>'
                )
    out.append(
        f'<text x="{(L + R) / 2}" y="{h - 4}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text>'
    )
    out.append("</svg>")
    head = f'<div class="panel-title">{title}</div>' + (
        f'<div class="panel-sub">{sub}</div>' if sub else ""
    )
    return f"<div>{head}{''.join(out)}</div>"


def dot_legend(items):
    spans = [
        f'<span class="li"><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="{c}"/></svg>{lab}</span>'
        for lab, c in items
    ]
    return f'<div class="legend">{"".join(spans)}</div>'


def bar_panel(
    categories,
    values,
    title,
    sub="",
    color="#2a78d6",
    fmt=None,
    w=340,
    h=210,
    xlabel="",
    vmax=None,
):
    """Vertical bars, one per category."""
    fmt = fmt or (lambda v: f"{v:.2f}")
    L, R, T, B = 40, w - 10, 16, h - 40
    vmax = vmax or max(values) * 1.15
    sy = lambda v: B - v / vmax * (B - T)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(0, vmax, 4):
        out.append(
            f'<line x1="{L}" x2="{R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{GRID}"/>'
            f'<text x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    n = len(categories)
    cw = (R - L) / n
    bw = cw * 0.6
    for i, (c, v) in enumerate(zip(categories, values)):
        x0 = L + cw * i + (cw - bw) / 2
        out.append(
            f'<rect x="{x0:.1f}" y="{sy(v):.1f}" width="{bw:.1f}" height="{B - sy(v):.1f}" rx="3" fill="{color}"/>'
            f'<text x="{x0 + bw / 2:.1f}" y="{sy(v) - 4:.1f}" text-anchor="middle" font-size="10" fill="{MUTED}">{fmt(v)}</text>'
            f'<text x="{x0 + bw / 2:.1f}" y="{B + 15}" text-anchor="middle" font-size="11" fill="{INK}">{c}</text>'
        )
    out.append(f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{MUTED}"/>')
    out.append(
        f'<text x="{(L + R) / 2}" y="{h - 6}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text>'
    )
    out.append("</svg>")
    head = f'<div class="panel-title">{title}</div>' + (
        f'<div class="panel-sub">{sub}</div>' if sub else ""
    )
    return f"<div>{head}{''.join(out)}</div>"


def hbar_panel(
    labels,
    values,
    notes,
    title="",
    sub="",
    color="#2a78d6",
    w=680,
    row=34,
    xmax=100,
    xlabel="",
    fmt=None,
):
    """Horizontal bars with a note after each bar."""
    fmt = fmt or (lambda v: f"{v:.0f}%")
    n = len(labels)
    L, R, T = 150, w - 170, 8
    B = T + n * row
    h = B + 34
    sx = lambda v: L + min(v, xmax) / xmax * (R - L)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(0, xmax, 4):
        out.append(
            f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T}" y2="{B}" stroke="{GRID}"/>'
            f'<text x="{sx(t):.1f}" y="{B + 15}" text-anchor="middle" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    for i, (lab, v, note) in enumerate(zip(labels, values, notes)):
        y0 = T + i * row + 7
        bh = row - 14
        out.append(
            f'<text x="{L - 8}" y="{y0 + bh / 2 + 4:.1f}" text-anchor="end" font-size="11" fill="{INK}">{lab}</text>'
            f'<rect x="{L}" y="{y0}" width="{max(sx(v) - L, 1.5):.1f}" height="{bh}" rx="2" fill="{color}"/>'
            f'<text x="{max(sx(v), L + 1.5) + 6:.1f}" y="{y0 + bh / 2 + 4:.1f}" font-size="10.5" fill="{MUTED}">{fmt(v)} · {note}</text>'
        )
    out.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{B}" stroke="{MUTED}"/>')
    out.append(
        f'<text x="{(L + R) / 2}" y="{h - 4}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text>'
    )
    out.append("</svg>")
    head = (f'<div class="panel-title">{title}</div>' if title else "") + (
        f'<div class="panel-sub">{sub}</div>' if sub else ""
    )
    return f"<div>{head}{''.join(out)}</div>"


def interval_panel(
    labels,
    truth,
    est,
    est_lo,
    est_hi,
    title="",
    sub="",
    c_truth="#111827",
    c_est="#1baf7a",
    w=680,
    row=40,
    xmax=None,
    xlabel="",
    fmt=None,
):
    """Per row: an estimate with an interval, and a truth tick."""
    fmt = fmt or (lambda v: f"{v:.2f}")
    n = len(labels)
    L, R, T = 120, w - 20, 8
    B = T + n * row
    h = B + 34
    xmax = xmax or max(max(est_hi), max(truth)) * 1.1
    sx = lambda v: L + max(min(v, xmax), 0) / xmax * (R - L)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}">']
    for t in _nice_ticks(0, xmax, 5):
        out.append(
            f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T}" y2="{B}" stroke="{GRID}"/>'
            f'<text x="{sx(t):.1f}" y="{B + 15}" text-anchor="middle" font-size="11" fill="{MUTED}">{_fmt(t)}</text>'
        )
    for i, lab in enumerate(labels):
        yc = T + (i + 0.5) * row
        out.append(
            f'<text x="{L - 8}" y="{yc + 4:.1f}" text-anchor="end" font-size="11" fill="{INK}">{lab}</text>'
        )
        out.append(
            f'<line x1="{sx(est_lo[i]):.1f}" x2="{sx(est_hi[i]):.1f}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="{c_est}" stroke-width="3" opacity=".45"/>'
        )
        out.append(
            f'<circle cx="{sx(est[i]):.1f}" cy="{yc:.1f}" r="5" fill="{c_est}" stroke="#fff" stroke-width="1.5">'
            f"<title>Estimate {fmt(est[i])}</title></circle>"
        )
        out.append(
            f'<line x1="{sx(truth[i]):.1f}" x2="{sx(truth[i]):.1f}" y1="{yc - 10:.1f}" y2="{yc + 10:.1f}" stroke="{c_truth}" stroke-width="2.2">'
            f"<title>Truth {fmt(truth[i])}</title></line>"
        )
    out.append(
        f'<text x="{(L + R) / 2}" y="{h - 4}" text-anchor="middle" font-size="11" fill="{MUTED}">{xlabel}</text>'
    )
    out.append("</svg>")
    head = (f'<div class="panel-title">{title}</div>' if title else "") + (
        f'<div class="panel-sub">{sub}</div>' if sub else ""
    )
    return f"<div>{head}{''.join(out)}</div>"
