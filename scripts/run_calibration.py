#!/usr/bin/env python3
"""Pip-calibration batch runner. Parallel across pairs (both games on one worker,
iter17 loaded once/worker). Cumulative delta-hat + cluster-bootstrap CI after each
batch of eligible reps; stop at relative precision <= 0.15 or the eligible cap.

Usage: python scripts/run_calibration.py --cap 10000 --batch 500 --workers 7
       (the committed calibration run used --batch 500 --workers 7; see CALIBRATION.md)
Writes results/calib_rows.parquet and results/calib_progress.jsonl incrementally.
"""
import os, sys, json, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "engine"))

BASE = 5_000_000            # calibration seed range [5M, 6M)
_AGENT = None


def _init():
    import torch
    torch.set_num_threads(1)
    global _AGENT
    from crn.agent import make_agent
    _AGENT = make_agent("iter17")


def _worker(seed):
    """Run one calibration pair; return a compact flat row (never raises)."""
    focal = seed % 4
    try:
        from crn.calibration import run_calibration
        r = run_calibration(seed, _AGENT, focal, feature_every=0)
    except Exception as e:
        return {"pair_seed": seed, "focal": focal, "eligible": False,
                "reason": f"error:{type(e).__name__}"}
    if r["skipped"]:
        return {"pair_seed": seed, "focal": focal, "eligible": False,
                "reason": r["reason"], "n_focal_exclusive": r.get("n_focal_exclusive"),
                "base_total_pips": r.get("base_total_pips")}
    rp, rm = r["swap_records"]["P+"], r["swap_records"]["P-"]
    gP, gM = r["gameP"], r["gameM"]
    clean = (not rp["partner_adjacent_to_opponent"]) and (not rm["partner_adjacent_to_opponent"])
    return {
        "pair_seed": seed, "focal": focal, "eligible": True,
        "rank_plus": r["rank_plus"], "rank_minus": r["rank_minus"],
        "delta": r["rank_minus"] - r["rank_plus"],          # +ve => pip helps
        "clean_partner": clean,
        "opp_affected": not clean,
        "n_opp_plus": len(rp["partner_opponents"]), "n_opp_minus": len(rm["partner_opponents"]),
        "n_focal_exclusive": r["n_focal_exclusive"],
        "base_total_pips": r["base_total_pips"],
        "focal_pips_plus": r["focal_pips_plus"], "focal_pips_minus": r["focal_pips_minus"],
        "win_plus": int(gP["winner"] == focal), "win_minus": int(gM["winner"] == focal),
        "vp_plus": gP["vp"][focal], "vp_minus": gM["vp"][focal],
        "completed_plus": int(gP["completed"]), "completed_minus": int(gM["completed"]),
        "maritime_plus": gP["maritime_trades"][focal], "maritime_minus": gM["maritime_trades"][focal],
        "scarcity_plus": gP["bank_scarcity_events"], "scarcity_minus": gM["bank_scarcity_events"],
    }


def _boot_ci(vals, B=10000, seed=12345):
    import numpy as np
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    means = v[idx].mean(axis=1)
    return float(means.std(ddof=1)), (float(np.percentile(means, 2.5)),
                                      float(np.percentile(means, 97.5)))


def _stats(rows, label):
    import numpy as np
    d = np.array([r["delta"] for r in rows], float)
    dhat = float(d.mean())
    se, (lo, hi) = _boot_ci(d)
    relprec = se / abs(dhat) if dhat != 0 else float("inf")
    return {"label": label, "n": len(rows), "delta_hat": dhat, "se": se,
            "ci95": [lo, hi], "sd_within_pair": float(d.std(ddof=1)),
            "rel_precision": relprec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 2) - 1))
    ap.add_argument("--relprec", type=float, default=0.15)
    args = ap.parse_args()

    outdir = ROOT / "results"; outdir.mkdir(exist_ok=True)
    rows_path = outdir / "calib_rows.parquet"
    prog_path = outdir / "calib_progress.jsonl"
    prog_path.write_text("")

    import pandas as pd
    from multiprocessing import Pool

    t0 = time.time()
    all_rows, eligible = [], []
    next_report = args.batch
    stop = False
    max_seeds = args.cap * 3   # generous headroom given ~80% eligibility
    seeds = range(BASE, BASE + max_seeds)

    print(f"[calib] workers={args.workers} cap={args.cap} batch={args.batch} "
          f"relprec<={args.relprec}", flush=True)
    with Pool(args.workers, initializer=_init) as pool:
        for row in pool.imap(_worker, seeds, chunksize=2):
            all_rows.append(row)
            if row["eligible"]:
                eligible.append(row)
            if len(eligible) >= next_report or len(eligible) >= args.cap:
                st = _stats(eligible, "all")
                clean = [r for r in eligible if r["clean_partner"]]
                st_clean = _stats(clean, "clean") if len(clean) >= 30 else None
                elig_rate = len(eligible) / len(all_rows)
                opp_rate = sum(r["opp_affected"] for r in eligible) / len(eligible)
                rec = {"t_min": round((time.time() - t0) / 60, 1),
                       "seeds_tried": len(all_rows), "n_eligible": len(eligible),
                       "elig_rate": round(elig_rate, 3), "opp_affect_rate": round(opp_rate, 3),
                       "all": st, "clean": st_clean}
                with open(prog_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                pd.DataFrame(all_rows).to_parquet(rows_path, index=False)
                print(f"[batch] n={st['n']} dhat={st['delta_hat']:+.4f} "
                      f"SE={st['se']:.4f} CI=[{st['ci95'][0]:+.3f},{st['ci95'][1]:+.3f}] "
                      f"relprec={st['rel_precision']:.3f} elig={elig_rate:.2f} "
                      f"opp={opp_rate:.2f} ({rec['t_min']}min)", flush=True)
                next_report += args.batch
                if st["rel_precision"] <= args.relprec or len(eligible) >= args.cap:
                    stop = True
                    print(f"[stop] {'relprec<=%.2f' % args.relprec if st['rel_precision']<=args.relprec else 'cap'} "
                          f"at n={st['n']}", flush=True)
                    break
        pool.terminate()

    pd.DataFrame(all_rows).to_parquet(rows_path, index=False)
    final = {"final": _stats(eligible, "all"),
             "clean": _stats([r for r in eligible if r["clean_partner"]], "clean")
                      if sum(r["clean_partner"] for r in eligible) >= 30 else None,
             "n_seeds_tried": len(all_rows), "elig_rate": len(eligible) / len(all_rows),
             "opp_affect_rate": sum(r["opp_affected"] for r in eligible) / len(eligible),
             "runtime_min": round((time.time() - t0) / 60, 1), "stopped_early": stop}
    (outdir / "calib_summary.json").write_text(json.dumps(final, indent=2))
    print("[done] " + json.dumps(final["final"]), flush=True)


if __name__ == "__main__":
    main()
