## Analyzes model training outputs and summarizes evaluation metrics across variants and folds.

#!/usr/bin/env python3
import argparse, json, os, glob, re, math, statistics as stats
from datetime import datetime, timezone

METRIC_KEYS_DEFAULT = ["eval_fbeta", "eval_precision", "eval_recall", "eval_loss"]

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import csv
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, path)

def parse_step_from_ckpt(ckpt_path):
    m = re.search(r"checkpoint-(\d+)", ckpt_path)
    return int(m.group(1)) if m else -1

def pick_checkpoint(fold_dir, mode):
    ckpts = glob.glob(os.path.join(fold_dir, "checkpoint-*"))
    if not ckpts:
        return None

    if mode == "last":
        return sorted(ckpts, key=parse_step_from_ckpt)[-1]

    # mode == "best": try to read best_model_checkpoint from any ckpt's trainer_state
    # Prefer the latest ckpt's trainer_state to find best_model_checkpoint.
    last_ckpt = sorted(ckpts, key=parse_step_from_ckpt)[-1]
    state_path = os.path.join(last_ckpt, "trainer_state.json")
    if os.path.exists(state_path):
        state = read_json(state_path)
        best = state.get("best_model_checkpoint")
        if best and os.path.isdir(best):
            return best

    # fallback: if best not found, use last
    return last_ckpt

def extract_last_eval_log(trainer_state):
    # last entry that contains eval_loss (or any eval_* key)
    for log in reversed(trainer_state.get("log_history", [])):
        if any(k.startswith("eval_") for k in log.keys()):
            return log
    return None

def mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return float(vals[0]), 0.0, 1
    return float(stats.mean(vals)), float(stats.stdev(vals)), len(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output", help="Base output dir (default: output)")
    ap.add_argument("--variants", default="*", help="Variant name(s), comma-separated or '*' (default: *)")
    ap.add_argument("--mode", choices=["best", "last"], default="best", help="Use best or last checkpoint (default: best)")
    ap.add_argument("--metrics", default=",".join(METRIC_KEYS_DEFAULT),
                    help="Comma-separated metric keys to extract (default: eval_f1,eval_accuracy,eval_precision,eval_recall,eval_loss)")
    ap.add_argument("--outdir", default="analysis", help="Where to write results (default: analysis)")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    variants = [v.strip() for v in args.variants.split(",")] if args.variants != "*" else ["*"]

    # Expand variants
    all_variant_dirs = sorted(glob.glob(os.path.join(args.base, "*")))
    selected = []
    for vd in all_variant_dirs:
        if not os.path.isdir(vd):
            continue
        name = os.path.basename(vd)
        if variants == ["*"] or name in variants:
            selected.append((name, vd))

    fold_rows = []
    summary_rows = []
    summary_json = {
        "generated_at_utc": utc_now(),
        "base": args.base,
        "mode": args.mode,
        "metrics": metrics,
        "variants": [],
    }

    for variant_name, variant_dir in selected:
        fold_dirs = sorted(glob.glob(os.path.join(variant_dir, "fold_*")))
        per_metric_vals = {k: [] for k in metrics}
        per_fold = []

        for fold_dir in fold_dirs:
            fold_name = os.path.basename(fold_dir)  # fold_0, fold_1, ...
            ckpt = pick_checkpoint(fold_dir, args.mode)
            if ckpt is None:
                row = {
                    "variant": variant_name,
                    "fold": fold_name,
                    "checkpoint": "",
                    "step": "",
                    "status": "no_checkpoint",
                }
                fold_rows.append(row)
                per_fold.append(row)
                continue

            state_path = os.path.join(ckpt, "trainer_state.json")
            if not os.path.exists(state_path):
                row = {
                    "variant": variant_name,
                    "fold": fold_name,
                    "checkpoint": ckpt,
                    "step": parse_step_from_ckpt(ckpt),
                    "status": "missing_trainer_state",
                }
                fold_rows.append(row)
                per_fold.append(row)
                continue

            state = read_json(state_path)
            eval_log = extract_last_eval_log(state)
            if eval_log is None:
                row = {
                    "variant": variant_name,
                    "fold": fold_name,
                    "checkpoint": ckpt,
                    "step": parse_step_from_ckpt(ckpt),
                    "status": "no_eval_log",
                }
                fold_rows.append(row)
                per_fold.append(row)
                continue

            row = {
                "variant": variant_name,
                "fold": fold_name,
                "checkpoint": ckpt,
                "step": eval_log.get("step", parse_step_from_ckpt(ckpt)),
                "status": "ok",
            }
            for k in metrics:
                row[k] = safe_float(eval_log.get(k))
                if row[k] is not None:
                    per_metric_vals[k].append(row[k])

            fold_rows.append(row)
            per_fold.append(row)

        # Aggregate per variant
        summary = {"variant": variant_name, "mode": args.mode}
        for k in metrics:
            m, s, n = mean_std(per_metric_vals[k])
            summary[f"{k}_mean"] = m
            summary[f"{k}_std"] = s
            summary[f"{k}_n"] = n
        summary_rows.append(summary)

        summary_json["variants"].append({
            "variant": variant_name,
            "folds": per_fold,
            "summary": summary,
        })

    # Save outputs
    outdir = args.outdir
    folds_csv = os.path.join(outdir, "metrics_folds.csv")
    summ_csv = os.path.join(outdir, "metrics_summary.csv")
    folds_json = os.path.join(outdir, "metrics_folds.json")
    summ_json = os.path.join(outdir, "metrics_summary.json")

    fold_fields = ["variant", "fold", "status", "step", "checkpoint"] + metrics
    summ_fields = ["variant", "mode"] + [f"{k}_{suf}" for k in metrics for suf in ("mean", "std", "n")]

    write_csv(folds_csv, fold_rows, fold_fields)
    write_csv(summ_csv, summary_rows, summ_fields)

    # also write JSON
    write_json(folds_json, {"generated_at_utc": utc_now(), "rows": fold_rows})
    write_json(summ_json, summary_json)

    print(f"Wrote:\n  {folds_csv}\n  {summ_csv}\n  {folds_json}\n  {summ_json}")

if __name__ == "__main__":
    main()
