"""Evaluate RobotBrain on a fixed test set.

Usage:  python evaluate_brain.py
Output: eval/brain_results.csv (per request) + eval/brain_summary.json
"""
import csv, json, re, time
from collections import defaultdict
from brain import RobotBrain

ARABIC = re.compile(r"[\u0600-\u06FF]")


def score(pred, case):
    """correct = every required group is satisfied and nothing outside required/optional."""
    groups = [set(g) for g in case["required"]]
    allowed = set(case.get("optional", [])).union(*groups) if groups else set(case.get("optional", []))
    got = {f"{a['action']}:{a['object']}" for a in pred}
    hit = sum(1 for g in groups if got & g)
    extra = sorted(got - allowed)
    correct = (not got - set(case.get("optional", []))) if not groups else (hit == len(groups) and not extra)
    return correct, hit, len(groups), extra


def main():
    brain = RobotBrain()
    with open("eval/brain_tests.json") as f:
        cases = json.load(f)

    rows, by_cat = [], defaultdict(list)
    for c in cases:
        t0 = time.time()
        try:
            r = brain.ask(c["text"])
            error = ""
        except Exception as e:
            r, error = {"reply": "", "actions": []}, str(e)
        dt = time.time() - t0

        correct, hit, need, extra = score(r["actions"], c)
        lang_ok = bool(ARABIC.search(c["text"])) == bool(ARABIC.search(r["reply"]))
        acts = " ".join(f"{a['action']}:{a['object']}" for a in r["actions"]) or "-"
        print(f"{c['id']:>2} {c['cat']:<10} {'✓' if correct else '✗'}  lang:{'✓' if lang_ok else '✗'}  "
              f"{dt:5.1f}s  {acts}")
        rows.append({"id": c["id"], "cat": c["cat"], "text": c["text"], "actions": acts,
                     "reply": r["reply"], "correct": correct, "groups_hit": f"{hit}/{need}",
                     "extra": " ".join(extra), "lang_ok": lang_ok,
                     "seconds": round(dt, 1), "error": error})
        by_cat[c["cat"]].append(correct)

    with open("eval/brain_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    summary = {
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "language_accuracy": round(sum(r["lang_ok"] for r in rows) / n, 3),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in by_cat.items()},
        "n": n,
    }
    with open("eval/brain_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(f"Accuracy: {summary['accuracy']:.0%}   Language: {summary['language_accuracy']:.0%}   "
          f"Avg time: {summary['mean_seconds']}s")
    for k, v in summary["by_category"].items():
        print(f"  {k:<10} {v}")


if __name__ == "__main__":
    main()
