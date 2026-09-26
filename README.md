# FLOG: Fairness-aware Latent Orienteering based on Graph

Official implementation of:

**On Fair Team Orienteering Problem**

(ICLR 2027 Submission)

---

## Overview

We introduce **FLOG (Fairness-aware Latent Orienteering based on Graph)**,
a neural framework for solving the **Fair Team Orienteering Problem (fTOP)**.

Different from conventional Team Orienteering Problem (TOP) solvers that mainly
optimize total collected reward, FLOG incorporates fairness considerations
among multiple agents by maximizing the minimum agent reward.

FLOG integrates:

- Fairness-Guided Construction (FGC)
- Residual Reward Recovery (RR)
- Fairness-preserving recovery strategy

to balance total reward and reward fairness.

---

## Environment

Tested environment:

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8

Install dependencies:

```bash
pip install -r requirements.txt