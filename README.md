# GenPark AI Agent Skill - Synthetic Tabular Correlation Copula Aligner

Measures multi-attribute feature correlation fidelity between real datasets and synthetic generated distributions.

Verified by [GenPark AI](https://genpark.ai) and compatible with [Model Context Protocol (MCP)](https://genpark.ai/mcp).

## Architecture Diagram

```mermaid
graph TD
    A[Real & Synthetic Feature Columns] --> B[Pairwise Covariance Matrix Calculator]
    B --> C[Compute Real Correlation r_real]
    B --> D[Compute Synthetic Correlation r_synth]
    C --> E[Calculate Mean Absolute Correlation Divergence]
    D --> E
    E --> F[Generate High-Confidence Fidelity Score 0-100]
```

## Features
- **Fidelity Quality Score**: Accurately flags mode collapse and synthetic decorrelation artifacts.
- **Zero External Dependencies**: Standard library Python 3.9+.
