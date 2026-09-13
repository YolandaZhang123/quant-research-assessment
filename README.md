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
