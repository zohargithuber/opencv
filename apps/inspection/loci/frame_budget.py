#!/usr/bin/env python3
"""Per-frame time estimate for opencv_inspection from LOCI block timings.

LOCI times one pass through a loop body. A frame costs
    sum over kernels of (ns per pass x passes for the plane it runs on)
where passes = plane_pixels / px_per_iter.  Kernels are listed in kernels.json.

Usage (LOCI venv python):
    frame_budget.py --config inspection.yaml --kernels loci/kernels.json loci-result.csv [...]
"""
import argparse
import csv
import json
import re
import sys


def read_yaml_list(text, key):
    m = re.search(r"^%s:\s*\[([^\]]*)\]" % re.escape(key), text, re.M)
    return [float(v) for v in m.group(1).split(",")] if m else None


def read_yaml_scalar(text, key):
    m = re.search(r"^%s:\s*([0-9.eE+-]+)" % re.escape(key), text, re.M)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="inspection.yaml (OpenCV FileStorage YAML)")
    ap.add_argument("--kernels", required=True, help="kernels.json: block name, stage, px_per_iter, plane")
    ap.add_argument("results", nargs="+", help="LOCI get_assembly_block_exec_behavior result CSVs")
    args = ap.parse_args()

    text = open(args.config).read()
    budget_ms = read_yaml_scalar(text, "frame_budget_ms")
    roi = read_yaml_list(text, "roi")
    model = read_yaml_list(text, "model_input")
    # roi_2 (two parts per frame): count both slots back to back, no credit for the second thread.
    slots = 2 if read_yaml_list(text, "roi_2") else 1
    planes = {"roi": roi[2] * roi[3] * slots}
    if model:
        planes["model"] = model[0] * model[1]

    rows = {}
    for path in args.results:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["function_name"]] = r

    stages, flags, missing = {}, [], []
    measured = 0
    for k in json.load(open(args.kernels)):
        r = rows.get(k["block"])
        if r is None:
            missing.append(k["block"])
            continue
        measured += 1
        passes = planes[k["plane"]] / k["px_per_iter"]
        ns = float(r["execution_time_ns"])
        std = float(r["std_dev_ns"])
        ms = ns * passes / 1e6
        stages[k["stage"]] = stages.get(k["stage"], 0.0) + ms
        ood = "fd_score" in r and r["fd_score"] and float(r["fd_score"]) > float(r["threshold"])
        print("  %-10s %-48s %4.0f ns/pass x %9.0f passes = %6.2f ms  (std %.0f ns%s)"
              % (k["stage"], k["block"].split("::")[-1][:48], ns, passes, ms, std,
                 ", out-of-distribution" if ood else ""))
        if ood or std > ns:
            flags.append(k["block"])

    total = sum(stages.values())
    print()
    if slots == 2:
        print("  two slots (roi + roi_2): serial worst case, no parallel speed-up assumed")
    for s, ms in stages.items():
        print("  stage %-10s %6.2f ms" % (s, ms))
    print("  TOTAL (measured kernels) %6.2f ms  vs budget %.2f ms  -> %s"
          % (total, budget_ms, "WITHIN" if total <= budget_ms else "OVER"))
    if missing:
        print("  NOT MEASURED (estimate is a lower bound): %s" % ", ".join(missing))
    if flags:
        print("  LOW CONFIDENCE: %d of %d kernels out-of-distribution or std > mean; "
              "confirm with wall-clock on target hardware" % (len(flags), measured))
    return 0 if total <= budget_ms and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
