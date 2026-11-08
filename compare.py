"""Compare every run in outputs/ and collect concrete AI mistakes as evidence.

  python compare.py            -> outputs/COMPARISON.md
"""
import json
import sys
from collections import Counter
from pathlib import Path

from script_parser import parse_ts

CATS = [("too short", "too short (<20s)"), ("too long", "too long (>60s)"), ("hook lands", "hook not in first 3s"),
        ("past episode end", "timestamp outside episode"), ("ends before", "end before start"),
        ("duplicate", "duplicate"), ("does not exist", "episode does not exist")]


def cat(reason):
    return next((label for k, label in CATS if k in reason), None)


def claimed_len(c):
    try:
        return sum(parse_ts(s["end"]) - parse_ts(s["start"]) for s in c["segments"])
    except Exception:
        return None


def main(root="outputs"):
    runs = sorted(p for p in Path(root).iterdir() if (p / "run_meta.json").exists())
    rows, examples = [], []
    for r in runs:
        meta = json.loads((r / "run_meta.json").read_text(encoding="utf-8"))
        promos = json.loads((r / "promos.json").read_text(encoding="utf-8"))
        rej = json.loads((r / "rejected.json").read_text(encoding="utf-8"))
        v = meta.get("validation", {})
        reasons = Counter(c for x in rej if x.get("stage") != "repair" for c in [cat(x["reasons"][0])] if c)
        judge = json.loads((r / "judge.json").read_text(encoding="utf-8"))["summary"] if (r / "judge.json").exists() else None
        top = promos[:10]
        eps_covered = len({int(e["step"][2:].split("_")[0]) for e in meta.get("error_list", []) if e["step"].startswith("ep")})
        rows.append({
            "run": r.name, "model": meta["model"], "secs": meta["seconds"],
            "episodes_failed": eps_covered, "valid": meta["valid"],
            "first": f"{v.get('first_pass_valid', '?')}/{v.get('first_pass_total', '?')}",
            "repair": f"{v.get('repair_valid', 0)}/{v.get('repair_total', 0)}",
            "reasons": ", ".join(f"{k} {n}" for k, n in reasons.most_common(3)) or "-",
            "top10_avg_ep": round(sum(p["segments"][0]["ep"] for p in top) / max(1, len(top)), 1),
            "judge": judge,
        })
        if meta["model"] != "baseline":
            for x in rej:
                if x.get("stage") == "repair" or not isinstance(x.get("llm_output"), dict):
                    continue
                c, first = x["llm_output"], x["reasons"][0]
                if cat(first) in ("too short (<20s)", "too long (>60s)"):
                    examples.append(f"`{meta['model']}` {x['id']}: proposed "
                                    f"{', '.join(s['start'] + '-' + s['end'] for s in c['segments'])} "
                                    f"= **{claimed_len(c):.1f}s**, outside the 20-60s it was told to hit")
                elif cat(first) == "hook not in first 3s":
                    examples.append(f"`{meta['model']}` {x['id']}: claimed hook \"{c.get('hook', '')[:70]}\" "
                                    f"but {first}")

    md = ["# Run comparison\n",
          "| Run | Model | Time (s) | Episodes errored | Picks passing checks (1st pass) | Fixed by retry | "
          "Final promos | Top reject reasons | Avg episode # in top 10 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['run']} | `{x['model']}` | {x['secs']} | {x['episodes_failed']} | {x['first']} | "
                  f"{x['repair']} | {x['valid']} | {x['reasons']} | {x['top10_avg_ep']} |")
    md += ["\n## Blind cold-viewer test (top 10 of each run, judged with no show context)\n",
           "| Run | Judge | Hook | Standalone | Cliffhanger | Picker self-score (H/S/C) | Would watch next | Overrated by picker |",
           "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        j = x["judge"]
        if j:
            ja, pa = j["judge_avg"], j["picker_avg"]
            md.append(f"| {x['run']} | `{j['judge_model']}` | {ja['hook']} | {ja['standalone']} | {ja['cliffhanger']} | "
                      f"{pa['hook']}/{pa['standalone']}/{pa['cliffhanger']} | {j['would_watch_next']}/{j['judged']} | "
                      f"{len(j['overrated_by_picker'])} |")
    md += ["\n## Concrete AI mistakes caught by the code checks (sample)\n"] + [f"- {e}" for e in examples[:12]]
    out = Path(root) / "COMPARISON.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(*sys.argv[1:])
