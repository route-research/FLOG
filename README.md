# FLOG: Fairness-aware Latent Orienteering based on Graph

Anonymous implementation accompanying **On Fair Team Orienteering Problem**
(ICLR 2027 submission).

## Overview

FLOG is a fairness-aware neural framework for the Fair Team Orienteering Problem
(fTOP). The review artifact follows the **equal-budget, open-route** setting and
provides the public model backbone, data generation/loading protocol, metrics,
configuration, tests, and interface boundaries of the proposed method.

To protect unpublished implementation know-how during double-blind review, the
internal Fairness-Guided Construction (FGC), Residual Reward Recovery (RR), and
fairness-preserving rollout logic are intentionally withheld in this repository.
See `REDACTION_NOTICE.md` for the exact scope. The repository does not silently
replace the protected components with another algorithm.

## Repository structure

```text
FLOG/
├── README.md
├── REDACTION_NOTICE.md
├── LICENSE
├── requirements.txt
├── train.py
├── evaluate.py
├── configs/
│   └── flog.yaml
├── flog/
│   ├── __init__.py
│   ├── model.py
│   ├── decoder.py
│   ├── data.py
│   ├── metrics.py
│   └── options.py
├── scripts/
│   ├── smoke_test.sh
│   ├── train.sh
│   ├── eval_val.sh
│   └── eval_real.sh
├── tests/
│   ├── test_public_api.py
│   └── test_data.py
├── data/
│   └── README.md
└── baselines/
    └── README.md
```

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Smoke test

```bash
pytest -q tests
```

The smoke tests verify the public policy backbone, evaluation metrics, and the
equal-budget open-route data generator. They also verify that invoking a
protected rollout fails explicitly with `CoreImplementationUnavailable`.

## Data protocol

Synthetic instances follow the equal-budget open-route protocol. For each
instance, all agents receive a common budget. The common actual budget is the
maximum of the mean sampled budget and the largest direct start-to-end distance
plus a small epsilon.

Representative synthetic scales include `<5,50>`, `<5,100>`, `<7,70>`, and
`<10,100>`. Real-world evaluation uses Denver, Los Angeles, and Seattle.

Large benchmark files can be distributed separately through a release asset.

## Metrics

The included metric utilities report total reward, minimum agent reward, maximum
agent reward, and reward fairness statistics. The paper reports RFR using the
aggregate minimum/maximum reward convention described in the manuscript.

## Protected components

`flog/decoder.py` exposes the interfaces of the protected modules but not their
internal implementation. Running `train.py` or `evaluate.py` in this review
release prints an explicit redaction notice rather than producing results from a
substitute method.

## Baselines

The experimental comparison includes GRIP, Greedy-LKH3, OR-Tools, GCB, and
NSGA-II. Third-party code is not redistributed in this repository.

## Anonymity

This repository intentionally contains no author names, affiliations, personal
email addresses, or identifying repository history.
