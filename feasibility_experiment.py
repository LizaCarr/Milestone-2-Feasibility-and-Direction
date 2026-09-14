#!/usr/bin/env python3
"""Throwaway Week-5 experiment for Team Carrillo SCO freeze mock.

Kill assumption: observable mock-kiosk fields (no live POS) can separate
freeze vs no-freeze well enough that charter thresholds are not fantasy.
Labels are NEVER used as model inputs.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

OUT = Path("/home/workdir/artifacts")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(4376)

N = 120  # 60 freeze / 60 clear, then split

def make_row(is_freeze: bool, freeze_kind: str | None) -> dict:
    # Observables an attendant/kiosk mock can expose without store systems
    scan_ok = True
    scale_delta_g = float(rng.normal(8, 12))
    age_check_needed = False
    unexpected_bag = False
    idle_s = float(max(0.2, rng.normal(1.4, 0.6)))
    items_in_tx = int(rng.integers(1, 18))
    attendant_busy = int(rng.random() < 0.35)

    if is_freeze:
        if freeze_kind == "wont_scan":
            scan_ok = False
            idle_s = float(max(3.0, rng.normal(8.0, 2.5)))
        elif freeze_kind == "bag_scale":
            scale_delta_g = float(rng.choice([-1, 1]) * rng.uniform(80, 420))
            idle_s = float(max(2.5, rng.normal(6.0, 2.0)))
        elif freeze_kind == "age_restricted":
            age_check_needed = True
            idle_s = float(max(4.0, rng.normal(12.0, 3.0)))
        elif freeze_kind == "unexpected_bag":
            unexpected_bag = True
            scale_delta_g = float(rng.uniform(60, 350))
            idle_s = float(max(3.0, rng.normal(7.0, 2.0)))
        # messy overlap: some freezes look almost normal
        if rng.random() < 0.12:
            scan_ok = True
            if freeze_kind != "bag_scale":
                scale_delta_g = float(rng.normal(15, 20))
    else:
        # ugly clears: heavy produce, slow bagger, coupon pause
        if rng.random() < 0.15:
            scale_delta_g = float(rng.uniform(40, 90))
        if rng.random() < 0.08:
            idle_s = float(rng.uniform(4.0, 9.0))
        if rng.random() < 0.05:
            age_check_needed = True  # already cleared by attendant, still in log

    return {
        "scan_ok": int(scan_ok),
        "scale_delta_g": round(scale_delta_g, 1),
        "age_check_needed": int(age_check_needed),
        "unexpected_bag": int(unexpected_bag),
        "idle_s": round(idle_s, 2),
        "items_in_tx": items_in_tx,
        "attendant_busy": attendant_busy,
        "freeze_kind": freeze_kind if is_freeze else "none",
        "is_freeze": int(is_freeze),
    }

kinds = ["wont_scan", "bag_scale", "age_restricted", "unexpected_bag"]
rows = []
for i in range(60):
    rows.append(make_row(True, kinds[i % 4]))
for i in range(60):
    rows.append(make_row(False, None))

df = pd.DataFrame(rows)
df.insert(0, "event_id", [f"E{i:03d}" for i in range(1, len(df) + 1)])
df = df.sample(frac=1.0, random_state=4376).reset_index(drop=True)
df.to_csv(OUT / "staged_events_sample.csv", index=False)

FEATURES = [
    "scan_ok",
    "scale_delta_g",
    "age_check_needed",
    "unexpected_bag",
    "idle_s",
    "items_in_tx",
    "attendant_busy",
]

train, test = train_test_split(
    df, test_size=40, random_state=16, stratify=df["is_freeze"]
)
# charter talks about 40 freeze + 40 clear holdout; we use 40 mixed holdout
# plus report freeze-only / clear-only slices

def rule_flag(row) -> int:
    """Approach A: readable attendant rules. No learned weights."""
    if row["scan_ok"] == 0:
        return 1
    if abs(row["scale_delta_g"]) >= 75:
        return 1
    if row["age_check_needed"] == 1 and row["idle_s"] >= 5:
        return 1
    if row["unexpected_bag"] == 1:
        return 1
    if row["idle_s"] >= 10:
        return 1
    return 0


def eval_preds(y_true, y_pred, freeze_mask, clear_mask):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    freeze_n = int(freeze_mask.sum())
    clear_n = int(clear_mask.sum())
    freeze_hit = int(((y_true == 1) & (y_pred == 1)).sum())
    freeze_miss = int(((y_true == 1) & (y_pred == 0)).sum())
    false_flags = int(((y_true == 0) & (y_pred == 1)).sum())
    return {
        "n": int(len(y_true)),
        "freeze_n": freeze_n,
        "clear_n": clear_n,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "freeze_hit": freeze_hit,
        "freeze_miss": freeze_miss,
        "false_flags": false_flags,
        "freeze_hit_rate": freeze_hit / freeze_n if freeze_n else None,
        "false_flag_rate": false_flags / clear_n if clear_n else None,
    }


# Approach A on holdout
t0 = time.perf_counter()
pred_a = test.apply(rule_flag, axis=1).to_numpy()
time_a_ms = (time.perf_counter() - t0) * 1000 / len(test)

# Approach B: logistic regression on train only
X_train = train[FEATURES].to_numpy()
y_train = train["is_freeze"].to_numpy()
X_test = test[FEATURES].to_numpy()
y_test = test["is_freeze"].to_numpy()

clf = LogisticRegression(max_iter=400, random_state=4376)
t1 = time.perf_counter()
clf.fit(X_train, y_train)
fit_b_s = time.perf_counter() - t1
t2 = time.perf_counter()
pred_b = clf.predict(X_test)
time_b_ms = (time.perf_counter() - t2) * 1000 / len(test)

# Approach B-alt tree for the comparison writeup (not the selected path)
tree = DecisionTreeClassifier(max_depth=3, random_state=4376)
tree.fit(X_train, y_train)
pred_t = tree.predict(X_test)

fm = y_test == 1
cm = y_test == 0
res_a = eval_preds(y_test, pred_a, fm, cm)
res_b = eval_preds(y_test, pred_b, fm, cm)
res_t = eval_preds(y_test, pred_t, fm, cm)

# Charter-style 40+40 would need more holdout; also score rules on ALL 60/60 for transparency
pred_a_all = df.apply(rule_flag, axis=1).to_numpy()
res_a_all = eval_preds(df["is_freeze"].to_numpy(), pred_a_all, df["is_freeze"] == 1, df["is_freeze"] == 0)

# latency mock: injecting a freeze into a log
latencies = []
log = []
for i in range(20):
    ev = make_row(True, kinds[i % 4])
    t_inj = time.perf_counter()
    flag = rule_flag(ev)
    log.append({"seq": i, "flag": flag, "idle_s": ev["idle_s"]})
    latencies.append((time.perf_counter() - t_inj) * 1000)

summary = {
    "n_total": int(len(df)),
    "n_train": int(len(train)),
    "n_holdout": int(len(test)),
    "holdout_freeze": int(fm.sum()),
    "holdout_clear": int(cm.sum()),
    "approach_A_rules_holdout": res_a,
    "approach_B_logreg_holdout": res_b,
    "approach_B_tree_holdout": res_t,
    "approach_A_rules_all120": res_a_all,
    "median_rule_latency_ms_per_event": float(np.median(latencies)),
    "p95_rule_latency_ms_per_event": float(np.percentile(latencies, 95)),
    "mean_batch_rule_ms_per_row_holdout": float(time_a_ms),
    "mean_batch_logreg_ms_per_row_holdout": float(time_b_ms),
    "logreg_fit_s": float(fit_b_s),
    "logreg_coef": {k: float(v) for k, v in zip(FEATURES, clf.coef_[0])},
    "charter_freeze_target": "34/40 hits, <=6 miss",
    "charter_false_flag_target": "<=8 / 40 clears",
    "charter_notice_target_s": 3.0,
    "note": "Holdout is 40 mixed events (~20/20), not the full 40+40 charter set. Full 40+40 is a later test.",
}

(OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
test.assign(pred_rule=pred_a, pred_logreg=pred_b).to_csv(OUT / "holdout_predictions.csv", index=False)
print(json.dumps(summary, indent=2))
