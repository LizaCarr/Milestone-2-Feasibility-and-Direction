# Milestone-2-Feasibility-and-Direction
# Team Carrillo — ITAI 4376

Individual enrollment. Option C. Student proposed. Not an instructor assignment.

This repository is coursework for a self-checkout freeze log. It is not a store pilot, not an HEB project, and it does not connect to live point-of-sale, cameras, or customer accounts.

Student: Elizabeth Carrillo

Outside owner: Bianca, assistant service manager. She confirmed in writing on 25 September 2026 that she will help for class and can ask a service manager if something needs confirming. She is not the instructor.

## Problem

One attendant covers a bank of self-checkout kiosks. When a kiosk freezes, the customer waits, and other kiosks keep stopping. Bianca’s account: customers usually wait 30 seconds to 2 minutes. First check is whether any buttons still work. A true freeze means a manual shutdown and moving the customer, because bringing the register back takes 5 to 10 minutes. On a normal day about one or two registers need a reset. This semester the system is a mock on a laptop or Colab: staged events in, attendant log out.

## Two layers

The mock runs on one laptop. The floor does not. The split is in TwoLayerNote_Carrillo_11Oct2026.md.

Near the kiosk: a watcher. It only sees that the lane has stopped answering. It does not reset the register.

With the attendant: the alert. Lane number and time. Nothing else.

On paper the watcher sends that short message to the attendant’s side. In the current mock both pieces are on one machine, so a pass on the laptop is not a pass on the walk.

Notice time starts at the freeze and stops when the alert is on the attendant side. Pass is under one minute. The old three-second clock is retired.

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
| `TwoLayerNote_Carrillo_11Oct2026.md` | Kiosk side, attendant side, and how the alert travels. |

What the throwaway test was for

Kill assumption: mock-kiosk fields (scan result, scale delta, age-check flag, unexpected-bag flag, idle time, basket size, attendant-busy) can flag a freeze without using the label as an input.

On a 20 freeze / 20 clear holdout:

Hand rules: 19 / 20 hits, 0 false flags

Logistic regression: 17 / 20 hits, 2 false flags

The generator and the rules share an author. That circularity is stated in the PDF. This is not the official 40 + 40 charter test. The 19 of 20 is not a claim that the charter target was met.

Design direction

Approach A (explicit rules + attendant log) is the semester spine. A learned model is on the bench until a later set is hand-coded from shift memory and labeled last.

Scope this semester

In scope: staged event file, first working loop to an attendant view, four charter tests on held-out staged events, updated risk register, the two-layer note.

Out of scope: live POS, face video, theft prediction, replacing the attendant, a production app, any claim that a store agreed to pilot this.

Charter success tests

At least 34 / 40 labeled freezes flagged (no more than 6 misses).

No more than 8 false flags in 40 non-freeze events.

Median time from freeze to the alert on the attendant side at or under one minute. The old three-second clock is retired.

Zero real customer names, phones, or loyalty IDs in submitted materials.

### How to rerun

python3 -m pip install pandas scikit-learn numpy
python3 feasibility_experiment.py
