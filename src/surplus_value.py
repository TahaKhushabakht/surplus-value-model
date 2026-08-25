"""The surplus-value model — the culmination of the pipeline (impact -> scarcity -> price).

Predicts log market value from performance characteristics, then reads the RESIDUAL as
the surplus signal: a player priced below what their characteristics predict is a
candidate bargain; priced above, a candidate premium.

THE CENTRAL TEST, stated up front so it can't be quietly skipped: does adding CPV
(quality) and scarcity_density improve prediction beyond boring baseline features --
age, playing time, position, and a reputation proxy (international caps, which
directly answers the omitted-variable concern flagged in docs/literature.md's Chen
paper)? If not, the whole CPV/scarcity pipeline has added nothing a simpler model
didn't already know, and that would be the actual finding to report -- not something
to engineer around. BASELINE and FULL models are both fit and compared by
cross-validated R², not in-sample fit, which would flatter any model regardless of
whether it generalizes.

TARGET: log(market_value_eur). Market values are heavily right-skewed (this
population: mean EUR11.5M, median EUR6.75M, max EUR110M) -- same transform used in
docs/literature.md's Chen et al., for the same reason (a linear model in raw euros
would be dominated by the handful of superstar valuations).

SCOPE, stated honestly: GK excluded (no scarcity_density is computed for them --
docs/scarcity.md's exclusion carries through here). Single competition, single season,
258 players -- an OLS on n=258 is appropriately modest for this sample size; the GBM
used for the completion model (HISTORY.md #23) was justified by 20k+ rows, which this
is nowhere close to. No club/team fixed effects: with 258 players unevenly split
across 18 clubs (22 from Leverkusen's full season vs a handful each from 2-match
opponent samples), team dummies would eat degrees of freedom without reliable
identification, AND a team-level price aggregate would reintroduce exactly the
circularity that disqualified market-tier scarcity (docs/scarcity.md) -- price
appearing on both sides of a model that predicts price. Consequence: club prestige is
NOT controlled for and lives inside the residual along with genuine mispricing; this
is disclosed, not hidden.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))

import transfermarkt_io as TM

OUTPUT_DIR = Path(__file__).parent.parent / "output"
VALUATION_DATE = pd.Timestamp("2023-08-01")

BASELINE_FEATURES = ["age", "age_sq", "log_passes", "log_caps", "pos_MID", "pos_FWD"]
FULL_FEATURES = BASELINE_FEATURES + ["quality", "scarcity_density"]


def build_dataset() -> pd.DataFrame:
    df = pd.read_pickle(OUTPUT_DIR / "price_data.pkl")
    tm_players = TM.load("players")
    prof = tm_players.set_index("player_id")[["date_of_birth", "international_caps"]]
    df = df.merge(prof, left_on="tm_player_id", right_index=True, how="left")

    # quality (cpv_zz) is missing for ~5% of the scarcity population: they have role
    # features (all 34 matches, no 360 requirement) but no pass that qualified for CPV
    # scoring (the narrower 31-match, 6m-match-distance-filtered set, HISTORY.md #23).
    # Dropped rather than imputed -- fabricating a value for the one feature under test
    # would bias the central comparison toward whatever the imputation assumed.
    df = df[df.market_value_eur.notna() & df.date_of_birth.notna() & df.quality.notna()].copy()
    dob = pd.to_datetime(df["date_of_birth"])
    df["age"] = (VALUATION_DATE - dob).dt.days / 365.25
    df["age_sq"] = df["age"] ** 2
    df["log_passes"] = np.log1p(df["n_passes"])
    df["log_caps"] = np.log1p(df["international_caps"].fillna(0))
    df["log_value"] = np.log(df["market_value_eur"])

    pos_dummies = pd.get_dummies(df["inferred_position"], prefix="pos", drop_first=True)
    df = pd.concat([df, pos_dummies.astype(float)], axis=1)
    for col in ("pos_MID", "pos_FWD"):
        if col not in df.columns:
            df[col] = 0.0

    return df.reset_index(drop=True)


def cross_validated_r2(df: pd.DataFrame, features: list[str], target: str = "log_value", n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-sample R^2 per fold -- NOT in-sample fit, which would flatter any model
    regardless of whether it generalizes. Reported as a distribution, not one number,
    since n=258 makes any single split noisy."""
    X, y = df[features].to_numpy(), df[target].to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in kf.split(X):
        model = sm.OLS(y[train_idx], sm.add_constant(X[train_idx])).fit()
        pred = model.predict(sm.add_constant(X[test_idx], has_constant="add"))
        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - y[test_idx].mean()) ** 2)
        scores.append(1 - ss_res / ss_tot)
    return np.array(scores)


def fit_full_model(df: pd.DataFrame, features: list[str] = FULL_FEATURES, target: str = "log_value"):
    X = sm.add_constant(df[features])
    return sm.OLS(df[target], X).fit()


if __name__ == "__main__":
    df = build_dataset()
    print(f"regression dataset: {len(df)} players (GK excluded, priced, age known)")
    print(f"  positions: {df.inferred_position.value_counts().to_dict()}")
    print()

    print("=== Central test: does quality+scarcity improve out-of-sample prediction? ===")
    r2_base = cross_validated_r2(df, BASELINE_FEATURES)
    r2_full = cross_validated_r2(df, FULL_FEATURES)
    print(f"  baseline (age, playing time, position, reputation): R2 = {r2_base.mean():.3f} +- {r2_base.std():.3f}")
    print(f"  full (+ quality + scarcity_density):                R2 = {r2_full.mean():.3f} +- {r2_full.std():.3f}")
    print(f"  improvement: {r2_full.mean() - r2_base.mean():+.3f}")
    print()

    model = fit_full_model(df)
    print("=== Full model coefficients (in-sample, for interpretation only) ===")
    print(model.summary().tables[1])
    print()
    print(f"in-sample R2 = {model.rsquared:.3f} (compare to cross-validated {r2_full.mean():.3f} above --")
    print(f" a big gap between these two numbers would mean the model is overfitting)")

    df["predicted_log_value"] = model.predict(sm.add_constant(df[FULL_FEATURES]))
    df["residual"] = df["log_value"] - df["predicted_log_value"]  # negative = underpriced
    df["surplus"] = -df["residual"]  # positive = underpriced (bargain), matches intuitive "moneyball" sign
    df["predicted_value_eur"] = np.exp(df["predicted_log_value"])

    df.to_pickle(OUTPUT_DIR / "surplus_value.pkl")

    print()
    print("=== Residual sanity check: does it correlate with n_passes? (it shouldn't --")
    print("    that would mean the model is reading exposure noise as mispricing) ===")
    print(f"  corr(residual, n_passes) = {df['residual'].corr(df['n_passes']):+.3f}")

    print()
    print("=== Most underpriced (bargains), n_passes>=100 for reliability ===")
    reliable = df[df.n_passes >= 100]
    top = reliable.nlargest(12, "surplus")[["name", "team", "inferred_position", "market_value_eur", "predicted_value_eur", "surplus"]]
    print(top.assign(
        market_value_eur=top.market_value_eur.map(lambda v: f"{v/1e6:.1f}M"),
        predicted_value_eur=top.predicted_value_eur.map(lambda v: f"{v/1e6:.1f}M"),
    ).to_string(index=False))

    print()
    print("=== Most overpriced, n_passes>=100 ===")
    bot = reliable.nsmallest(12, "surplus")[["name", "team", "inferred_position", "market_value_eur", "predicted_value_eur", "surplus"]]
    print(bot.assign(
        market_value_eur=bot.market_value_eur.map(lambda v: f"{v/1e6:.1f}M"),
        predicted_value_eur=bot.predicted_value_eur.map(lambda v: f"{v/1e6:.1f}M"),
    ).to_string(index=False))
