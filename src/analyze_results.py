"""Statistical analysis for Fisher Re-Basin pilot."""
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def compute_barrier_reduction(df, method_a, method_b, metric="barrier_test"):
    """Compute paired barrier reduction: delta_i = barrier_a(i) - barrier_b(i).
    Positive delta means method_b is better (lower barrier)."""
    a = df[df.method == method_a].set_index("pair_id")[metric]
    b = df[df.method == method_b].set_index("pair_id")[metric]
    common = a.index.intersection(b.index)
    delta = a.loc[common] - b.loc[common]
    return delta


def compute_relative_reduction(df, baseline, method, metric="barrier_test"):
    """Compute relative barrier reduction per pair."""
    base = df[df.method == baseline].set_index("pair_id")[metric]
    test = df[df.method == method].set_index("pair_id")[metric]
    common = base.index.intersection(test.index)
    rel = (base.loc[common] - test.loc[common]) / (base.loc[common] + 1e-12)
    return rel


def report_comparison(delta, label):
    """Report statistics for a paired comparison."""
    n = len(delta)
    if n == 0:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        print(f"  No data available")
        return {"mean": np.nan, "std": np.nan, "p": np.nan, "d": np.nan, "median": np.nan}

    mean_d = delta.mean()
    std_d = delta.std(ddof=1)
    cohens_d = mean_d / std_d if std_d > 0 else 0

    # Wilcoxon signed-rank test (one-sided: barrier_a > barrier_b)
    nonzero = delta[delta != 0]
    if len(nonzero) >= 5:
        stat, p_val = stats.wilcoxon(nonzero, alternative='greater')
    else:
        stat, p_val = np.nan, np.nan

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  N pairs:           {n}")
    print(f"  Mean delta:        {mean_d:.6f} +/- {std_d:.6f}")
    print(f"  Median delta:      {delta.median():.6f}")
    print(f"  Cohen's d:         {cohens_d:.3f}")
    print(f"  Wilcoxon p-value:  {p_val:.4f}" if not np.isnan(p_val) else
          f"  Wilcoxon p-value:  N/A (insufficient non-zero pairs)")
    se = std_d / np.sqrt(n) if n > 0 else 0
    print(f"  95% CI:            [{mean_d - 1.96 * se:.6f}, {mean_d + 1.96 * se:.6f}]")

    return {"mean": mean_d, "std": std_d, "p": p_val, "d": cohens_d, "median": delta.median()}


def apply_decision_matrix(all_results, all_relative):
    """Apply V2 decision matrix based on pairwise comparisons."""
    def is_significant(label, threshold=0.10):
        """Check if comparison is significant AND has >10% median relative reduction."""
        if label not in all_results:
            return False
        r = all_results[label]
        if np.isnan(r["p"]):
            return False
        # Check relative reduction if available
        rel_label = label
        if rel_label in all_relative:
            median_rel = all_relative[rel_label]
            return r["p"] < 0.05 and median_rel > threshold
        return r["p"] < 0.05

    def is_equivalent(label):
        if label not in all_results:
            return True
        r = all_results[label]
        if np.isnan(r["p"]):
            return True
        return r["p"] > 0.10

    mag_vs_euc = is_significant("Magnitude vs Euclidean")
    fisher_vs_euc = is_significant("Fisher Inner vs Euclidean")
    fisher_vs_mag = is_significant("Fisher Inner vs Magnitude")
    l2_vs_inner = is_significant("Fisher L2 vs Fisher Inner")

    print(f"\n{'=' * 60}")
    print(f"  DECISION RECOMMENDATION")
    print(f"{'=' * 60}")

    if not mag_vs_euc and not fisher_vs_euc:
        fi_vs_euc = all_results.get("Fisher Inner vs Euclidean", {})
        l2_vs_euc = all_results.get("Fisher L2 Sym vs Euclidean", {})
        if (not np.isnan(fi_vs_euc.get("p", np.nan)) and fi_vs_euc["p"] > 0.10 and
            not np.isnan(l2_vs_euc.get("p", np.nan)) and l2_vs_euc["p"] > 0.10):
            print("  -> PERMANENT SHELVE")
            print("     Importance weighting does not help alignment.")
        else:
            print("  -> INCONCLUSIVE (need more data or CIFAR-100)")
    elif mag_vs_euc and not fisher_vs_mag:
        print("  -> MODERATE GO: Importance-weighted alignment paper")
        print("     Fisher ~= Magnitude. Simpler proxy wins.")
    elif fisher_vs_mag and not l2_vs_inner:
        print("  -> MODERATE-STRONG GO: Fisher alignment paper (inner product)")
        print("     Fisher beats magnitude; simple formulation suffices.")
    elif fisher_vs_mag and l2_vs_inner:
        print("  -> STRONG GO: Fisher alignment paper (per-neuron L2)")
        print("     Per-neuron Fisher structure provides unique value.")
    else:
        print("  -> MIXED RESULTS: Review individual comparisons above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/interpolation_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Methods: {sorted(df.method.unique())}")
    print(f"Pairs: {sorted(df.pair_id.unique())}")

    comparisons = [
        ("euclidean", "magnitude",     "Magnitude vs Euclidean"),
        ("euclidean", "fisher_inner",  "Fisher Inner vs Euclidean"),
        ("euclidean", "fisher_l2_sym", "Fisher L2 Sym vs Euclidean"),
        ("magnitude", "fisher_inner",  "Fisher Inner vs Magnitude"),
        ("fisher_inner", "fisher_l2_sym", "Fisher L2 vs Fisher Inner"),
    ]

    all_results = {}
    all_relative = {}
    for base, test_method, label in comparisons:
        if base not in df.method.values or test_method not in df.method.values:
            print(f"\nSkipping {label}: missing method data")
            continue
        delta = compute_barrier_reduction(df, base, test_method)
        all_results[label] = report_comparison(delta, label)

        # Also compute relative reduction vs euclidean baseline
        if base == "euclidean":
            rel = compute_relative_reduction(df, "euclidean", test_method)
            median_rel = float(rel.median())
            all_relative[label] = median_rel
            print(f"  Median relative reduction vs euclidean: {median_rel:.4f} "
                  f"({'> 10%' if median_rel > 0.10 else '<= 10%'})")

    # Holm-Bonferroni correction
    p_values = [(k, v["p"]) for k, v in all_results.items() if not np.isnan(v["p"])]
    p_values.sort(key=lambda x: x[1])
    m = len(p_values)
    if m > 0:
        print(f"\n{'=' * 60}")
        print(f"  Holm-Bonferroni Correction (m={m})")
        print(f"{'=' * 60}")
        for i, (label, p) in enumerate(p_values):
            adjusted_alpha = 0.05 / (m - i)
            sig = "SIGNIFICANT" if p < adjusted_alpha else "not significant"
            print(f"  {label}: p={p:.4f}, alpha_adj={adjusted_alpha:.4f} -> {sig}")

    # Decision matrix
    apply_decision_matrix(all_results, all_relative)
