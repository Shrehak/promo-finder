"""Blind cold-viewer test: does a promo work for someone who has never seen the show?

A *different* model (default: Gemma, a different family from the Gemini model that picked the promos)
sees ONLY the lines inside the clip: no title, no loglines, no hook/cliffhanger labels, no 'why'.
It must say what it understood. We then compare its scores with the picker's own self-scores.

  python judge.py --input data/show --run outputs/<run> --top 10
  JUDGE_MODEL=gemini/gemini-3-flash-preview python judge.py ...

Writes <run>/judge.json and <run>/judge.md
"""
import argparse
import json
import os
from pathlib import Path

import llm
from script_parser import load_show, parse_ts

SYSTEM = """You are a random person scrolling TikTok. You have NEVER seen this show and know nothing about it.
You are shown the script of a short vertical video clip, line by line, with the second at which each line plays.
Judge it honestly and harshly, as a viewer, not as a writer. Answer ONLY with valid JSON."""

PROMPT = """CLIP SCRIPT (seconds from the start of the clip; CAMERA = what you see, ACTION = what happens,
SFX = sound, NAME = dialogue):
{clip}

Answer as JSON:
{{"first_3_seconds": "what you see/hear in the first 3 seconds",
  "hook": 1-5,            // would that stop your thumb mid-scroll? 5 = definitely
  "understood": "in one sentence: who is in conflict with whom, and what is at stake",
  "confused_by": ["anything you did not understand without knowing the show"],
  "standalone": 1-5,      // 5 = fully clear with zero context
  "unresolved_question": "what question is left open at the very end, or 'none'",
  "cliffhanger": 1-5,     // 5 = I NEED to know what happens next
  "would_watch_next": true/false}}"""


def clip_text(eps, promo):
    lines, offset = [], 0.0
    for s in promo["segments"]:
        ep = eps[s["ep"]]
        a, b = parse_ts(s["start"]), parse_ts(s["end"])
        for blk in ep["blocks"]:
            if blk["start"] < b and blk["end"] > a:
                t = offset + max(0.0, blk["start"] - a)
                who = blk["speaker"] if blk["type"] == "dialogue" else blk["type"].upper()
                lines.append(f"{t:5.1f}s  {who}: {blk['text']}")
        offset += b - a
        if len(promo["segments"]) > 1:
            lines.append(f"{offset:5.1f}s  [CUT]")
    return "\n".join(lines)


def num(x, d=3.0):
    try:
        return max(1.0, min(5.0, float(x)))
    except (TypeError, ValueError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    judge_model = os.getenv("JUDGE_MODEL", "gemini/gemma-4-26b-a4b-it")
    run = Path(a.run)
    eps = {e["ep"]: e for e in load_show(a.input)}
    promos = json.loads((run / "promos.json").read_text(encoding="utf-8"))[: a.top]

    results = []
    for p in promos:
        try:
            v = llm.call(PROMPT.format(clip=clip_text(eps, p)), SYSTEM, tag=f"judge_{p['id']}",
                         raw_dir=run / "raw_judge", model=judge_model)
        except Exception as ex:
            results.append({"id": p["id"], "rank": p["rank"], "error": str(ex)[:200]})
            continue
        results.append({"id": p["id"], "rank": p["rank"], "title": p["title"],
                        "picker_scores": p["scores"],
                        "judge": {k: v.get(k) for k in ("first_3_seconds", "hook", "understood", "confused_by",
                                                         "standalone", "unresolved_question", "cliffhanger",
                                                         "would_watch_next")}})
    ok = [r for r in results if "judge" in r]
    avg = lambda f: round(sum(f(r) for r in ok) / max(1, len(ok)), 2)
    summary = {
        "judge_model": judge_model, "picker_run": run.name, "judged": len(ok), "errors": len(results) - len(ok),
        "judge_avg": {k: avg(lambda r, k=k: num(r["judge"][k])) for k in ("hook", "standalone", "cliffhanger")},
        "picker_avg": {k: avg(lambda r, k=k: num(r["picker_scores"][k])) for k in ("hook", "standalone", "cliffhanger")},
        "would_watch_next": sum(1 for r in ok if r["judge"]["would_watch_next"] is True),
        "overrated_by_picker": [r["id"] for r in ok if
                                sum(num(r["picker_scores"][k]) - num(r["judge"][k])
                                    for k in ("hook", "standalone", "cliffhanger")) >= 4],
    }
    (run / "judge.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2,
                                               ensure_ascii=False), encoding="utf-8")

    md = [f"# Blind cold-viewer test: {run.name}\n",
          f"Judge `{judge_model}` saw only each clip's lines (no show context).\n",
          f"| | hook | standalone | cliffhanger |", "|---|---|---|---|",
          f"| Picker's self-score (avg) | {summary['picker_avg']['hook']} | {summary['picker_avg']['standalone']} | "
          f"{summary['picker_avg']['cliffhanger']} |",
          f"| Blind judge (avg) | {summary['judge_avg']['hook']} | {summary['judge_avg']['standalone']} | "
          f"{summary['judge_avg']['cliffhanger']} |\n",
          f"Would watch the next episode: **{summary['would_watch_next']}/{len(ok)}**. "
          f"Overrated by the picker (self-score ≥4 points above judge): {summary['overrated_by_picker'] or 'none'}\n"]
    for r in ok:
        j = r["judge"]
        md += [f"### #{r['rank']} {r['id']} {r.get('title', '')}",
               f"- hook {j['hook']} · standalone {j['standalone']} · cliffhanger {j['cliffhanger']} · "
               f"watch next: {j['would_watch_next']}",
               f"- understood: {j['understood']}", f"- confused by: {j['confused_by']}",
               f"- left open: {j['unresolved_question']}\n"]
    (run / "judge.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
