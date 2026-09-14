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

What the throwaway test was for
Kill assumption: mock-kiosk fields (scan result, scale delta, age-check flag, unexpected-bag flag, idle time, basket size, attendant-busy) can flag a freeze without using the label as an input.
On a 20 freeze / 20 clear holdout:

Hand rules: 19 / 20 hits, 0 false flags
Logistic regression: 17 / 20 hits, 2 false flags

The generator and the rules share an author. That circularity is stated in the PDF. This is not the official 40 + 40 charter test.
Design direction
Approach A (explicit rules + attendant log) is the semester spine. A learned model is on the bench until a later set is hand-coded from shift memory and labeled last.
Scope this semester
In scope: staged event file, first working loop to an attendant view, four charter tests on held-out staged events, updated risk register.
Out of scope: live POS, face video, theft prediction, replacing the attendant, a production app, any claim that a store agreed to pilot this.
Charter success tests (still targets)

At least 34 / 40 labeled freezes flagged (no more than 6 misses).
No more than 8 false flags in 40 non-freeze events.
Median inject-to-log time at or under 3 seconds over 20 injected freezes.
Zero real customer names, phones, or loyalty IDs in submitted materials.
### How to rerun

```bash
python3 -m pip install pandas scikit-learn numpy
python3 feasibility_experiment.py


