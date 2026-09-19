import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cpit_ensemble(y, dat, a=-np.inf, b=np.inf, rng=None):
    """Randomized conditional PIT for an ensemble forecast, given outcome in [a, b].
    Faithful port of pens()/cpit_sample(kde=FALSE) from sallen12/TailCalibration.
    y: (n,), dat: (n, m). Returns (n,) with NaN where y not in (a, b)."""
    if rng is None:
        rng = np.random.default_rng()
    y = np.asarray(y, float); dat = np.asarray(dat, float)
    p_q  = (dat <= y[:, None]).mean(1)
    p_qm = (dat <  y[:, None]).mean(1)
    p_a  = (dat <= a).mean(1)
    p_b  = (dat <= b).mean(1) if np.isfinite(b) else np.ones(len(y))
    v = rng.uniform(p_qm, p_q)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (v - p_a) / (p_b - p_a)
    p = np.where(p_b == 0, 0.0, p)
    p = np.where(p_a == 1, 1.0, p)
    p = np.where((y <= a) | (y >= b), np.nan, p)
    return p

def tc_prob_ensemble(y, dat, t, ratio="com", u=None, tail="upper", sup=False,
                     rng=None, n_rand=1):
    """Probabilistic tail-calibration ratios at threshold t for ensemble forecasts.
    ratio: 'com' | 'sev' | 'occ'.  tail: 'upper' | 'lower' (lower via reflection).
    sup=False -> (u, curve) for com/sev, scalar for occ.
    sup=True  -> sup-distance: max_u|rat-u| (com/sev) or |occ-1| (occ)."""
    if u is None:
        u = np.arange(0.01, 1.00, 0.01)
    if rng is None:
        rng = np.random.default_rng(0)
    y = np.asarray(y, float); dat = np.asarray(dat, float)
    if tail == "lower":
        y, dat, t = -y, -dat, -t
    elif tail != "upper":
        raise ValueError("tail must be 'upper' or 'lower'")
    n = len(y)
    F_t = (dat <= t).mean(1)
    exc_p = float(np.mean(1.0 - F_t))
    G_t = float(np.mean(y > t))
    occ = G_t / exc_p if exc_p > 0 else np.nan
    if ratio == "occ":
        return abs(occ - 1.0) if sup else occ
    curves = []
    for _ in range(n_rand):
        z = cpit_ensemble(y, dat, a=t, b=np.inf, rng=rng); z = z[~np.isnan(z)]
        if ratio == "sev":
            rat = np.array([np.mean(z <= uu) for uu in u]) if z.size else np.full_like(u, np.nan, float)
        elif ratio == "com":
            denom = n * exc_p
            rat = (np.array([np.sum(z <= uu) for uu in u]) / denom) if denom > 0 else np.full_like(u, np.nan, float)
        else:
            raise ValueError("ratio must be 'com', 'sev' or 'occ'")
        curves.append(rat)
    rat = np.nanmean(curves, axis=0)
    return float(np.nanmax(np.abs(rat - u))) if sup else (u, rat)

# ---------- threshold sweep + diagram ----------
_DEF_UP  = np.array([0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99])
_DEF_LOW = np.array([0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01])

def tail_cal_sweep(target_ensembles, y_true_1d, y_train_1d, tail="upper",
                   levels=None, n_rand=10, seed=0, min_exc=30):
    """Sweep thresholds (train quantile levels) -> sup-distance per model.
    target_ensembles: {model: (n_obs, m)} for a single univariate target."""
    levels = (_DEF_UP if tail == "upper" else _DEF_LOW) if levels is None else np.asarray(levels)
    yv = np.asarray(y_true_1d, float); ytr = np.asarray(y_train_1d, float)
    rows = []
    for model, dat in target_ensembles.items():
        for lvl in levels:
            t = float(np.quantile(ytr, lvl))
            n_exc = int(np.sum(yv > t)) if tail == "upper" else int(np.sum(yv < t))
            occ = tc_prob_ensemble(yv, dat, t, "occ", tail=tail)
            sev = tc_prob_ensemble(yv, dat, t, "sev", tail=tail, sup=True,
                                   rng=np.random.default_rng(seed), n_rand=n_rand)
            com = tc_prob_ensemble(yv, dat, t, "com", tail=tail, sup=True,
                                   rng=np.random.default_rng(seed), n_rand=n_rand)
            if n_exc < min_exc:
                sev = com = np.nan
            rows.append({"model": model, "tail": tail, "q": float(lvl), "t": t,
                         "n_exc": n_exc, "occ": occ, "sev_sup": sev, "com_sup": com})
    return pd.DataFrame(rows)

def plot_tail_cal_sweep(df, value="com_sup", x="q", out_path=None, title=None, ax=None):
    import matplotlib.pyplot as plt
    ref = 1.0 if value == "occ" else 0.0
    ylab = {"com_sup": "Miscalibration (combined sup-dist)",
            "sev_sup": "Miscalibration (severity sup-dist)",
            "occ": "Occurrence ratio"}[value]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in df.groupby("model"):
        g = g.sort_values(x)
        ax.plot(g[x], g[value], marker="o", ms=4, label=model)
    ax.axhline(ref, ls=":", color="k", lw=1)
    ax.set_xlabel("threshold quantile level q" if x == "q" else "threshold t (EUR/MWh)")
    ax.set_ylabel(ylab); ax.set_title(title or "Tail-calibration diagram"); ax.legend(fontsize=8)
    if own:
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        return fig
    return ax


U_GRID = np.arange(0.01, 1.00, 0.01)


def _curve_pieces(y, dat, t, tail, u, rng, n_rand):
    """Raw poolable pieces of the reliability curve at threshold t for one horizon.

    Returns (count_curve, n_exc, denom):
      count_curve[k] = #{ cPIT <= u_k AND y exceeds t }, averaged over n_rand draws
      n_exc          = #{ exceedances }
      denom          = sum_i (1 - F_i(t)) = expected #exceedances
    Lower tail via reflection. Combined ratio = count/denom; severity = count/n_exc;
    both are poolable across horizons by summing numerators and denominators.
    """
    y = np.asarray(y, float);
    dat = np.asarray(dat, float);
    t = float(t)
    if tail == "lower":
        y, dat, t = -y, -dat, -t
    elif tail != "upper":
        raise ValueError("tail must be 'upper' or 'lower'")
    F_t = (dat <= t).mean(1)
    denom = float(np.sum(1.0 - F_t))
    n_exc = int(np.sum(y > t))
    counts = np.zeros_like(u, dtype=float)
    for _ in range(n_rand):
        z = cpit_ensemble(y, dat, a=t, b=np.inf, rng=rng)
        z = z[~np.isnan(z)]
        if z.size:
            counts += np.searchsorted(np.sort(z), u, side="right").astype(float)
    counts /= n_rand
    return counts, n_exc, denom


def tail_cal_curves_pooled(loaded_preds, y_true, y_train, tail, q,
                           u=U_GRID, n_rand=10, seed=0):
    """Pooled-over-horizons reliability curves at one threshold quantile level q.

    Returns tidy DataFrame: model, tail, q, u, R_com, R_sev, occ, n_exc.
    occ/n_exc are constant within a (model, q) block (handy for annotation).
    """
    H = y_true.shape[1]
    rows = []
    for model, pred in loaded_preds.items():
        num = np.zeros_like(u, dtype=float)
        denom_com = 0.0
        n_exc_tot = 0
        for h in range(H):
            t = float(np.quantile(y_train[:, h], q))
            c, n_exc, denom = _curve_pieces(
                y_true[:, h], pred[:, h, :], t, tail, u,
                np.random.default_rng(seed + h), n_rand)
            num += c;
            denom_com += denom;
            n_exc_tot += n_exc
        R_com = num / denom_com if denom_com > 0 else np.full_like(u, np.nan)
        R_sev = num / n_exc_tot if n_exc_tot > 0 else np.full_like(u, np.nan)
        occ = (n_exc_tot / denom_com) if denom_com > 0 else np.nan
        for k, uu in enumerate(u):
            rows.append({"model": model, "tail": tail, "q": float(q), "u": float(uu),
                         "R_com": float(R_com[k]), "R_sev": float(R_sev[k]),
                         "occ": float(occ), "n_exc": int(n_exc_tot)})
    return pd.DataFrame(rows)


def plot_tail_cal_curves(curves_df, ratio="R_com", q_levels=None, ncols=3,
                         out_path=None, title=None):
    """plot_ptc analog: x=u, y=R(t,u), dashed 45-deg line = calibrated.
    One panel per q (small multiples), all models overlaid. ratio in {R_com, R_sev}.
    Combined curve ends at (1, occ): endpoint above 1 = under-forecast tail, below = over.
    """
    qs = list(q_levels) if q_levels is not None else sorted(curves_df["q"].unique())
    n = len(qs);
    ncols = min(ncols, n);
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows), squeeze=False)
    axf = axes.ravel()
    ylab = {"R_com": "Combined ratio  R_com(t,u)",
            "R_sev": "Severity ratio  R_sev(t,u)"}[ratio]
    for i, q in enumerate(qs):
        ax = axf[i]
        d = curves_df[np.isclose(curves_df["q"], q)]
        ax.plot([0, 1], [0, 1], ls="--", color="k", lw=1, zorder=1)
        ymax = 1.0
        for model, g in d.groupby("model"):
            g = g.sort_values("u")
            ax.plot(g["u"], g[ratio], lw=1.4, label=model, zorder=2)
            ymax = max(ymax, float(np.nanmax(g[ratio].values)))
        ax.set_xlim(0, 1);
        ax.set_ylim(0, max(1.05, ymax * 1.05))
        ax.set_xlabel("u");
        ax.set_ylabel(ylab)
        nexc = int(d["n_exc"].iloc[0]) if len(d) else 0
        ax.set_title(f"q = {q}   (n_exc = {nexc})")
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")
    for j in range(n, len(axf)):
        axf[j].set_visible(False)
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight");
        plt.close(fig)
        return out_path
    return fig


def plot_occ_by_horizon(tc_per_horizon_df, tail, q, out_path=None, title=None, ax=None):
    """p_tail_occ analog: occurrence ratio per subperiod, one line per model,
    dashed reference at 1. Reads the per-horizon tail-cal table already built.
    """
    d = tc_per_horizon_df[(tc_per_horizon_df["tail"] == tail)
                          & (np.isclose(tc_per_horizon_df["q"], q))]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1.0, ls="--", color="k", lw=1)
    for model, g in d.groupby("model"):
        g = g.sort_values("horizon")
        ax.plot(g["horizon"], g["occ"], marker="o", ms=4, lw=1.2, label=model)
    ax.set_xlabel("subperiod (horizon)")
    ax.set_ylabel("occurrence ratio  R_occ(t)")
    ax.set_title(title or f"Occurrence ratio — {tail} tail, q={q}")
    ax.legend(fontsize=7)
    if own:
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight");
            plt.close(fig)
            return out_path
        return fig
    return ax