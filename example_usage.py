import json
from client import SyntheticTabularCorrelationCopulaAligner

def main():
    aligner = SyntheticTabularCorrelationCopulaAligner()
    # Real dataset: positive correlation between tenure and salary
    real_data = {
        "tenure_years": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "salary_usd": [60000, 72000, 85000, 96000, 110000, 125000]
    }
    # High-quality synthetic dataset preserving the trend
    synth_data = {
        "tenure_years": [1.2, 2.1, 2.9, 4.2, 4.8, 6.1],
        "salary_usd": [61000, 71000, 84000, 98000, 109000, 126000]
    }
    res = aligner.evaluate_correlation_fidelity(real_data, synth_data)
    print("Correlation Alignment Fidelity:")
    print(json.dumps(res, indent=2))
    assert res["correlation_fidelity_score"] >= 95.0
    print("Correlation aligner verification: PASS")

if __name__ == "__main__":
    main()
