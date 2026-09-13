# Abnormal Trading Volume and Short-Horizon Return Reversal

## Overview

This repository contains the code and outputs for my Quant Team Round 1 research assessment.

The project investigates whether abnormal trading volume contains incremental information about short-horizon future returns following large daily price moves in U.S. equities.

The original hypothesis was that unusually high trading volume would be associated with stronger same-direction price continuation. Development-sample evidence did not support this hypothesis symmetrically. Instead, the analysis identified an asymmetric relationship: following large negative daily price moves, High-AVOL stocks exhibited stronger subsequent short-horizon reversal than comparable Low-AVOL stocks.

The revised hypothesis was then evaluated using a locked February-December 2025 holdout period.

---

## Data

The analysis uses Massive U.S. equities data, primarily:

- Daily Market Summary / Aggregate Bars for daily OHLCV data;
- Historical Ticker Reference data for point-in-time common-stock universe construction.

Raw Massive licensed data are stored locally under:

`data/`

The entire `data/` directory is excluded from version control and is not distributed in this repository.

---

## API Credential Handling

The Massive API credential is personal, temporary and confidential.

The API key is not stored in:

- Python source code;
- notebooks;
- `.env` files;
- configuration files;
- repository history.

Scripts requiring API access request the credential interactively at runtime using Python's `getpass()`.

Massive API requests must only be made from an authorised Hong Kong IP address and without a VPN, in accordance with the assessment rules.

---

## Python Environment

The project was run using Python 3.

Required packages are listed in:

`requirements.txt`

Install them using:

```bash
python3 -m pip install -r requirements.txt

```

## Main Research Specification

The primary specification uses:

- U.S. common stocks from monthly point-in-time historical ticker universes;
- signal-day price >= $5;
- trailing 20-day average dollar volume >= $1 million;
- close-to-close signal-day returns;
- top and bottom 20% of the daily cross-sectional return distribution;
- AVOL20, defined as current volume divided by average volume over the previous 20 trading days;
- twenty 1-percentile return-rank bins within each mover tail;
- a within-bin median split into High- and Low-AVOL stocks;
- 1D, 3D and 5D forward returns;
- HAC/Newey-West inference with lag equal to horizon minus one.

The main development sample uses signal dates from February 2, 2015 to January 24, 2025.

The locked holdout contains 225 signal dates from February 3, 2025 to December 23, 2025.

## Repository Structure

- `research_notebook.ipynb`: reviewer-facing research narrative with code, outputs and interpretation.
- `download_*.py`: Massive data-acquisition scripts.
- `build_*.py`: feature and research-sample construction.
- `evaluate_*.py`: experiment evaluation.
- `diagnose_move_magnitude.py`: signal-day magnitude-confound diagnostic.
- `continuous_return_control.py`: residual signal-return-gap control.
- `run_development_robustness.py`: parameter robustness tests.
- `run_locked_holdout.py`: locked out-of-sample evaluation.
- `generate_final_outputs.py`: final aggregate tables and figures.
- `documentation/decision_log.md`: chronological research decisions.
- `documentation/experiment_record.md`: meaningful experiments, including failed and superseded versions.
- `documentation/assistance_and_sources.md`: source, data and AI-tool disclosure.
- `results/`: derived aggregate research outputs.
- `data/`: local raw/processed data; ignored by Git and not distributed.

## Reproducing the Analysis

Data-acquisition scripts require an authorised Massive API credential. The credential is requested interactively using `getpass()` and is never stored in the repository.

The full pipeline is:

1. `download_monthly_universes.py`
2. `download_full_daily.py`
3. `inspect_liquidity_distribution.py`
4. `build_formal_development.py`
5. `evaluate_formal_development.py`
6. `diagnose_move_magnitude.py`
7. `build_magnitude_controlled_development.py`
8. `evaluate_magnitude_controlled_development.py`
9. `continuous_return_control.py`
10. `run_development_robustness.py`
11. `run_locked_holdout.py`
12. `generate_final_outputs.py`

The reviewer-facing `research_notebook.ipynb` reproduces the main aggregate analysis from derived outputs without embedding or redistributing raw Massive licensed data.

## Main Findings

The original symmetric continuation hypothesis was not supported.

After controlling for signal-day return magnitude, the strongest relationship appears among negative movers. In the primary development specification, negative-mover High-minus-Low AVOL spreads are approximately:

- 1D: +4.52 bps
- 3D: +12.89 bps
- 5D: +21.04 bps

The relationship remains positive under nearby robustness specifications and in the locked holdout.

These estimates are interpreted as predictive associations rather than causal effects or directly implementable trading profits.

## Documentation

The research process is documented in:

- `documentation/decision_log.md`
- `documentation/experiment_record.md`
- `documentation/assistance_and_sources.md`

The Final Report provides the polished research narrative, while `research_notebook.ipynb` provides the code-and-evidence trail supporting it.