# Milestone-2-Feasibility-and-Direction
# Team Carrillo — ITAI 4376

Individual enrollment. Option B, instructor assigned.

This repository is coursework for a  self-checkout freeze log. It is not a store pilot, not an HEB project, and it does not connect to live point-of-sale, cameras, or customer accounts.

Student: Elizabeth Carrillo

## Problem

One attendant covers a bank of self-checkout kiosks. When a kiosk freezes (item will not scan, bag-scale mismatch, age-restricted item, unexpected item in bag), the customer waits, and other kiosks keep stopping. This semester, the system is a mock on a laptop or Colab: staged events in, attendant log out.

## Week 5 — Feasibility

Document: [M05_Carrillo_Carrillo_ITAI4376.pdf](M05_Carrillo_Carrillo_ITAI4376.pdf)

Access proof and throwaway experiment:

| File | What it is |
| --- | --- |
| `staged_events_sample.csv` | 120 staged events (60 freeze / 60 clear). No real customer identifiers. |
| `feasibility_experiment.py` | Generator + rule scorer + logistic-regression scorer. Seed 4376. |
| `M05_Carrillo_feasibility_experiment.ipynb` | Same experiment as a Colab / Jupyter notebook. |
| `holdout_predictions.csv` | 40-row holdout with both systems' predictions. |
| `experiment_summary.json` | Metrics from the 13 September 2026 run. |

### How to rerun

```bash
python3 -m pip install pandas scikit-learn numpy
python3 feasibility_experiment.py
