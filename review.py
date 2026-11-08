"""Human quality check: print each top promo next to the exact script lines it covers,
and write a review sheet (CSV) to fill in good/bad + reason.

  python review.py --input data/show --run outputs/<run_folder> --top 10
"""
import argparse
import csv
import json
from pathlib import Path

from script_parser import fmt, load_show, parse_ts


def clip_lines(eps, seg):
    ep = eps[seg["ep"]]
    a, b = parse_ts(seg["start"]), parse_ts(seg["end"])
    out = []
    for blk in ep["blocks"]:
        if blk["start"] < b and blk["end"] > a and blk["type"] != "camera":
            who = blk["speaker"] if blk["type"] == "dialogue" else blk["type"].upper()
            out.append(f"  +{blk['start'] - a:5.1f}s  {who}: {blk['text']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    eps = {e["ep"]: e for e in load_show(a.input)}
    promos = json.loads((Path(a.run) / "promos.json").read_text(encoding="utf-8"))[: a.top]

    text = []
    for p in promos:
        text.append(f"\n=== #{p['rank']} {p['id']} {p['title']} ({p['duration_s']}s) ===")
        text.append(f"HOOK: {p['hook']}\nCLIFF: {p['cliffhanger']}\nWHY: {p['why']}")
        for s in p["segments"]:
            text.append(f"-- Ep{s['ep']} {s['start']}-{s['end']}  (logline: {eps[s['ep']]['logline'][:120]})")
            text += clip_lines(eps, s)
    out_txt = Path(a.run) / "review_clips.txt"
    out_txt.write_text("\n".join(text), encoding="utf-8")

    sheet = Path(a.run) / "review_sheet.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "hook_in_3s (y/n)", "ends_unresolved (y/n)", "standalone (y/n)",
                    "verdict (good/ok/bad)", "reason"])
        for p in promos:
            w.writerow([p["rank"], p["id"], "", "", "", "", ""])
    print(f"Read {out_txt}\nFill in {sheet}")


if __name__ == "__main__":
    main()
