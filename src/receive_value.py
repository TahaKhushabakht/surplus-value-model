"""V(receive) reception-quality discount — the second half of the CPV validation-
dashboard fix (see HISTORY.md §24 for V(turnover); this closes out the diagnosed
remaining cause).

THE BUG THIS ADDRESSES. V(receive) is currently xT(target zone): the AVERAGE value of
receiving the ball in that zone, as if every reception happens calmly. It does not
distinguish a pass received in space from a contested ball into traffic. Since
`control_target` (pitch control at the target, from the passing team's perspective) is
already computed for every candidate, the fix is to discount xT by it -- but the
discount function is ESTIMATED from data, not assumed, following the same discipline
used for the turnover fix.

METHOD. For every COMPLETED pass, ask whether the possession it belongs to ends in a
goal (`possession_scored`, same idea as the turnover surface's post-turnover goal
check). Test whether control_target predicts this OUTCOME-INDEPENDENTLY OF ZONE by
fitting:

    scored ~ xT(zone) + control_target      (logistic)

If control_target's coefficient survives controlling for zone value, low-control
receptions are measurably worse than the zone average suggests, and the effect is real
rather than a restatement of "high control zones are also high-xT zones" (which would
show up as a large xT coefficient and a small/insignificant control coefficient).

The discount is built as `sigmoid(a + b*control_target)`, normalized so its average over
the observed control distribution is 1.0 -- this keeps V(receive)'s overall scale
matched to the xT surface it modifies, so the fix changes RELATIVE reception value
without silently deflating every CPV number.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))

from cpv_scale import MATCH_DIST_M, PITCH_LENGTH_M, PITCH_WIDTH_M, XTLookup
from pitch_control import control_from_positions

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def build_reception_dataset(passes: pd.DataFrame, xt: XTLookup) -> pd.DataFrame:
    """Completed passes only: target zone value + reception control + outcome."""
    d = passes[(passes.completed == 1) & (passes.target_match_dist <= MATCH_DIST_M)].copy()
    controls = []
    for r in d.itertuples(index=False):
        att, opp = np.asarray(r.teammates, float), np.asarray(r.opponents, float)
        if len(att) == 0 or len(opp) == 0:
            controls.append(np.nan)
            continue
        c = control_from_positions(att, opp, np.array([[r.target_x, r.target_y]]))[0]
        controls.append(c)
    d["control_target"] = controls
    d["zone_xt"] = xt.value(d["target_x"].to_numpy(), d["target_y"].to_numpy())
    return d.dropna(subset=["control_target"])


def fit_discount(d: pd.DataFrame):
    """Logistic scored ~ zone_xt + control_target; returns (model, discount_fn, diagnostics)."""
    X = d[["zone_xt", "control_target"]].to_numpy()
    y = d["possession_scored"].to_numpy()
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    b_zone, b_control = model.coef_[0]
    intercept = model.intercept_[0]

    # Isolate the control effect: hold zone_xt at its mean, sweep control_target.
    zone_mean = float(d["zone_xt"].mean())
    ctl_grid = np.linspace(d["control_target"].quantile(0.02), d["control_target"].quantile(0.98), 50)
    logit = intercept + b_zone * zone_mean + b_control * ctl_grid
    raw = 1.0 / (1.0 + np.exp(-logit))

    # Normalize to a mean-1 multiplicative discount over the OBSERVED control distribution.
    obs_logit = intercept + b_zone * d["zone_xt"].to_numpy() + b_control * d["control_target"].to_numpy()
    obs_p = 1.0 / (1.0 + np.exp(-obs_logit))
    norm_const = float(obs_p.mean())

    def discount_fn(control):
        control = np.atleast_1d(np.asarray(control, float))
        lo = intercept + b_zone * zone_mean + b_control * control
        return (1.0 / (1.0 + np.exp(-lo))) / norm_const

    return model, discount_fn, {
        "b_zone": b_zone, "b_control": b_control, "intercept": intercept,
        "control_grid": ctl_grid, "p_at_mean_zone": raw, "norm_const": norm_const,
    }


class ReceiveLookup:
    """V(receive) = xT(zone) * discount(control), with the fixed discount from fit_discount."""

    def __init__(self, xt: XTLookup, discount_fn):
        self.xt = xt
        self.discount_fn = discount_fn

    def value(self, x_m, y_m, control):
        base = self.xt.value(x_m, y_m)
        return base * self.discount_fn(control)


if __name__ == "__main__":
    passes = pd.read_pickle(OUTPUT_DIR / "sb_passes.pkl")
    xt = XTLookup(np.load(OUTPUT_DIR / "sb_xt.npy"))

    d = build_reception_dataset(passes, xt)
    print(f"{len(d):,} completed passes with reception context")
    print(f"overall P(possession scores) = {d.possession_scored.mean():.4f}")
    print(f"control_target: mean={d.control_target.mean():.3f}, "
          f"p10={d.control_target.quantile(.1):.3f}, p90={d.control_target.quantile(.9):.3f}")

    print()
    print("=== Does reception control predict scoring BEYOND zone value? ===")
    model, discount_fn, diag = fit_discount(d)
    print(f"  logistic coefficients: b_zone_xt={diag['b_zone']:+.3f}  b_control_target={diag['b_control']:+.3f}")

    # Significance via bootstrap on the control coefficient (matches project convention
    # of measuring rather than trusting a single point estimate).
    rng = np.random.default_rng(0)
    boot_b = []
    for _ in range(300):
        idx = rng.integers(0, len(d), len(d))
        m = LogisticRegression(max_iter=2000).fit(d[["zone_xt", "control_target"]].to_numpy()[idx], d["possession_scored"].to_numpy()[idx])
        boot_b.append(m.coef_[0][1])
    lo, hi = np.percentile(boot_b, [2.5, 97.5])
    print(f"  bootstrap 95% CI on control coefficient: [{lo:+.3f}, {hi:+.3f}]  ({'excludes 0 -> real effect' if lo*hi>0 else 'includes 0 -> not significant'})")

    print()
    print("=== Discount curve at average zone value ===")
    for c, p in zip(diag["control_grid"][::7], diag["p_at_mean_zone"][::7]):
        print(f"  control_target={c:.2f}  ->  P(scores)={p:.4f}  ->  discount={p/diag['norm_const']:.3f}")

    np.save(OUTPUT_DIR / "sb_receive_discount_params.npy",
            np.array([diag["b_zone"], diag["b_control"], diag["intercept"], model.intercept_[0] and 0 or 0, diag["norm_const"], float(d.zone_xt.mean())]))
    print()
    print("saved discount params -> output/sb_receive_discount_params.npy")
