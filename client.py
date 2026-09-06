import math
from typing import Dict, Any, List, Optional

class SyntheticTabularCorrelationCopulaAligner:
    """
    Computes Pearson correlation matrices for numerical columns across original vs synthetic datasets
    and measures Frobenius norm divergence to quantify synthetic fidelity.
    """
    def compute_pearson_correlation(self, x: List[float], y: List[float]) -> float:
        n = len(x)
        if n < 2 or len(y) != n:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        denom = math.sqrt(var_x * var_y)
        return round(cov / denom, 4) if denom > 1e-9 else 0.0

    def evaluate_correlation_fidelity(
        self,
        real_columns: Dict[str, List[float]],
        synth_columns: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        col_names = sorted(list(real_columns.keys()))
        divergences = []
        real_corrs = {}
        synth_corrs = {}

        for i in range(len(col_names)):
            for j in range(i + 1, len(col_names)):
                c1, c2 = col_names[i], col_names[j]
                pair_key = f"{c1}_vs_{c2}"
                r_corr = self.compute_pearson_correlation(real_columns[c1], real_columns[c2])
                s_corr = self.compute_pearson_correlation(synth_columns[c1], synth_columns[c2])
                real_corrs[pair_key] = r_corr
                synth_corrs[pair_key] = s_corr
                divergences.append(abs(r_corr - s_corr))

        mean_abs_error = sum(divergences) / max(1, len(divergences))
        fidelity_score = max(0.0, round((1.0 - mean_abs_error) * 100, 2))

        return {
            "evaluated_feature_pairs": len(divergences),
            "mean_correlation_error": round(mean_abs_error, 4),
            "correlation_fidelity_score": fidelity_score,
            "real_correlations": real_corrs,
            "synthetic_correlations": synth_corrs
        }
