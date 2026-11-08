"""Promo finder: screenplay folder in -> ranked promo list out.

  python pipeline.py --input path/to/show_folder            # uses $MODEL (default Gemini free tier)
  MODEL=baseline python pipeline.py --input ...              # no AI, offline
  python pipeline.py --input ... --episodes 1-5              # quick test on a few episodes

Steps:  parse (code) -> per-episode candidates (LLM) -> validate (code)
        -> cross-episode montages (LLM) -> validate (code) -> dedupe (code) -> rank (LLM judge) -> write files
"""
import argparse
import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import baseline
import llm
import prompts
from script_parser import load_show_with_report, parse_report_md, to_transcript
from validate import check, dedupe


def log(msg, cb=None):
    print(msg, flush=True)
    if cb:
        cb(msg)


def run(input_dir, out_root="outputs", episodes=None, per_ep=3, montages=4, top_k=10, progress=None, repair=True, late_cap=2):
    model = llm.model_name()
    use_llm = model != "baseline"
    show, parse_report = load_show_with_report(input_dir)
    if episodes:
        show = [e for e in show if episodes[0] <= e["ep"] <= episodes[1]]
    if not show:
        raise SystemExit(f"No episodes found in {input_dir}")
    by_num = {e["ep"]: e for e in show}
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + model.replace("/", "-")
    out = Path(out_root) / run_id
    raw = out / "raw_llm"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"{len(show)} episodes | model={model} | output -> {out}", progress)
    if not parse_report["ok"]:  # fail visibly: a new script format should not silently produce bad promos
        log("WARNING input check found problems:\n" + parse_report_md(parse_report), progress)

    valid, rejected, errors = [], [], []
    stats = Counter()

    # 1) Per-episode candidates (LLM calls run in parallel threads; validation stays in episode order)
    def get_cands(e):
        story = "\n".join(f"Ep{p['ep']}: {p['logline']}" for p in show if p["ep"] < e["ep"]) or "(first episode)"
        if not use_llm:
            return baseline.candidates(e, per_ep), []
        transcript = to_transcript(e)
        res = llm.call(prompts.episode_prompt(e, story, transcript, per_ep),
                       prompts.SYSTEM, tag=f"ep{e['ep']:03d}", raw_dir=raw)
        cands = res.get("promos", []) if isinstance(res, dict) else res
        # Feedback loop: picks that fail fixable checks go back to the model ONCE with the exact reason
        fails = [(c, r) for c in cands for ok, r in [check(c, by_num)] if ok is None]
        if not repair or not fails:
            return cands, []
        try:
            res2 = llm.call(prompts.repair_prompt(e, transcript, fails), prompts.SYSTEM,
                            tag=f"ep{e['ep']:03d}_repair", raw_dir=raw)
            return cands, (res2.get("promos", []) if isinstance(res2, dict) else res2)
        except llm.ConfigError:
            raise
        except Exception as ex:  # repair is a bonus; keep the first pass
            errors.append({"step": f"ep{e['ep']}_repair", "error": str(ex)[:500]})
            return cands, []

    workers = int(os.getenv("LLM_WORKERS", "6")) if use_llm else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {e["ep"]: pool.submit(get_cands, e) for e in show}
        for e in show:
            try:
                cands, repaired = futures[e["ep"]].result()
            except llm.ConfigError:
                pool.shutdown(cancel_futures=True)
                raise  # missing key / bad model name: stop now, not 50 times
            except Exception as ex:  # one bad episode should not kill the run
                errors.append({"step": f"ep{e['ep']}", "error": str(ex)[:500]})
                log(f"  ep{e['ep']}: ERROR {str(ex)[:120]}", progress)
                continue
            ok = {"first_pass": 0, "repair": 0}
            for stage, items, pre in (("first_pass", cands, ""), ("repair", repaired, "r")):
                for i, c in enumerate(items, 1):
                    stats[stage + "_total"] += 1
                    clean, notes = check(c, by_num)
                    pid = f"E{e['ep']:02d}-{pre}{i}"
                    if clean:
                        clean["id"], clean["stage"] = pid, stage
                        valid.append(clean)
                        ok[stage] += 1
                        stats[stage + "_valid"] += 1
                    else:
                        rejected.append({"id": pid, "stage": stage, "reasons": notes, "llm_output": c})
            log(f"  ep{e['ep']}: {len(cands)} candidates, {ok['first_pass']} valid"
                + (f"; repair: {len(repaired)} returned, {ok['repair']} valid" if repaired else ""), progress)

    # 2) Cross-episode montages, built from the moments found above
    if use_llm and montages and len(show) > 1 and valid:
        # v1 let the model write montage timestamps: all 4 came back 127-147 s long. Now code cuts every
        # single-episode promo into an opening beat and an ending beat with exact lengths; the model only
        # picks beat IDs in order, and code turns them into segments. The LLM chooses, code computes.
        top = [p for p in sorted(valid, key=lambda p: -p["score"]) if len(p["_secs"]) == 1][:30]
        beats = {}
        for p in top:
            s = p["_secs"][0]
            bl = [b for b in by_num[s["ep"]]["blocks"] if b["type"] != "camera" and s["start"] <= b["start"] < s["end"]]
            open_end = next((b["end"] for b in bl if b["end"] - s["start"] >= 7), s["end"])
            close_start = next((b["start"] for b in reversed(bl) if s["end"] - b["start"] >= 9), s["start"])
            beats[p["id"] + "a"] = (s["ep"], s["start"], min(open_end, s["end"]), "OPENS: " + p["hook"])
            beats[p["id"] + "b"] = (s["ep"], max(close_start, s["start"]), s["end"], "ENDS: " + p["cliffhanger"])
        moments = "\n".join(f"{k} | ep{v[0]} | {v[2] - v[1]:.1f}s | {v[3]}" for k, v in beats.items())
        loglines = "\n".join(f"Ep{e['ep']}: {e['logline']}" for e in show)
        try:
            res = llm.call(prompts.montage_prompt(loglines, moments, montages), prompts.SYSTEM,
                           tag="montages", raw_dir=raw)
            for i, c in enumerate(res.get("promos", []), 1):
                ids = [b for b in (c.get("beats") or []) if b in beats]
                c["segments"] = [{"ep": beats[b][0], "start": beats[b][1], "end": beats[b][2]} for b in ids]
                clean, notes = check(c, by_num)
                if clean:
                    clean["id"] = f"M{i}"
                    valid.append(clean)
                    stats["montage_valid"] += 1
                else:
                    rejected.append({"id": f"M{i}", "stage": "montage", "reasons": notes, "llm_output": c})
                stats["montage_total"] += 1
            log(f"  montages: {len(res.get('promos', []))} proposed", progress)
        except Exception as ex:
            errors.append({"step": "montages", "error": str(ex)[:500]})

    # 3) Remove near-duplicates (same moment found twice)
    promos, dupes = dedupe(valid)
    stats["duplicates_removed"] = len(dupes)
    for d in dupes:
        rejected.append({"id": d["id"], "reasons": ["duplicate of a higher-scored promo"], "llm_output": d["title"]})

    # 4) Rank: LLM judge compares the best candidates head-to-head; the rest fall back to score order
    promos.sort(key=lambda p: -p["score"])
    reasons = {}
    if use_llm and promos:
        pool = [p for p in promos if not p["multi_episode"]][:25] + [p for p in promos if p["multi_episode"]]
        last_ep = max(by_num)
        cards = "\n".join(f"- id={p['id']} | {p['duration_s']}s | episodes {sorted({s['ep'] for s in p['_secs']})} "
                          f"of {last_ep} | HOOK: {p['hook']} | CLIFF: {p['cliffhanger']} | WHY: {p['why']}"
                          for p in pool)
        try:
            res = llm.call(prompts.rank_prompt(cards, min(top_k, len(pool))), prompts.SYSTEM,
                           tag="ranking", raw_dir=raw)
            order = [r["id"] for r in res.get("ranking", []) if any(p["id"] == r["id"] for p in promos)]
            reasons = {r["id"]: r.get("reason", "") for r in res.get("ranking", [])}
            promos.sort(key=lambda p: (order.index(p["id"]) if p["id"] in order else 999, -p["score"]))
        except Exception as ex:
            errors.append({"step": "ranking", "error": str(ex)[:500]})

    # 5) Code guard, added after the blind test: the LLM ranker put ep42/ep44 clips in its top 5 even when told to
    #    prefer early episodes, and cold viewers were confused by them (and they spoil the big reveal).
    #    Late-season clips (start in the last 40% of the show) are capped at `late_cap` slots in the top 10.
    late_from = min(by_num) + 0.6 * (max(by_num) - min(by_num) + 1)
    is_late = lambda p: min(s["ep"] for s in p["segments"]) >= late_from
    if len(by_num) >= 10 and late_cap is not None:
        top, rest, n_late = [], [], 0
        for p in promos:
            if len(top) < top_k and (not is_late(p) or n_late < late_cap):
                n_late += is_late(p)
                top.append(p)
            else:
                rest.append(p)
        for p in rest[:]:
            if is_late(p) and p in promos[:top_k]:
                p["warnings"].append("moved out of top 10: late-season spoiler cap")
        promos = top + rest
    for p in promos:
        if len(by_num) >= 10 and is_late(p):
            p["warnings"].append(f"late-season clip (ep {min(s['ep'] for s in p['segments'])}): spoiler risk")
    for r, p in enumerate(promos, 1):
        p["rank"] = r
        p["rank_reason"] = reasons.get(p["id"], "ranked by weighted score")
        p.pop("_secs", None)

    write_outputs(out, promos, rejected, errors, {
        "model": model, "input": str(input_dir), "episodes": [e["ep"] for e in show],
        "seconds": round(time.time() - t0, 1), "valid": len(promos), "rejected": len(rejected),
        "errors": len(errors), "per_ep": per_ep, "montages": montages,
        "validation": dict(stats), "input_check": parse_report,
        "parse_report": parse_report_md(parse_report), "repair": repair, "late_cap": late_cap})
    log(f"Done: {len(promos)} promos, {len(rejected)} rejected, {len(errors)} errors -> {out}", progress)
    return out, promos


def seg_str(p):
    return " + ".join(f"Ep{s['ep']} {s['start']}-{s['end']}" for s in p["segments"])


def funnel_md(v, rejected, final):
    """One source of truth for every count in the run, so the numbers always reconcile."""
    g = lambda k: v.get(k, 0)
    failed = g("first_pass_total") - g("first_pass_valid")
    singles = g("first_pass_valid") + g("repair_valid")
    reasons = Counter()
    for r in rejected:
        if r.get("stage") == "first_pass":
            for x in r["reasons"]:
                for key, label in (("too short", "too short (<20s)"), ("too long", "too long (>60s)"),
                                   ("hook lands", "hook after 3s"), ("past episode end", "timestamp outside episode"),
                                   ("ends before", "end before start"), ("does not exist", "bad episode")):
                    if key in x:
                        reasons[label] += 1
    lines = ["## Funnel (every count in this run)\n", "| Stage | Count |", "|---|---|",
             f"| Episode picks proposed by the model (first pass) | {g('first_pass_total')} |",
             f"| ...passed code checks | {g('first_pass_valid')} |",
             f"| ...failed, sent back once with the reasons | {failed} |",
             f"| Fixed picks the model returned (it may drop some) | {g('repair_total')} |",
             f"| ...passed code checks | {g('repair_valid')} |",
             f"| Single-episode promos (first pass + fixed) | {singles} |",
             f"| Montages proposed / passed | {g('montage_total')} / {g('montage_valid')} |",
             f"| Removed as near-duplicates (>60% overlap) | {g('duplicates_removed')} |",
             f"| **Final promos** ({singles} + {g('montage_valid')} - {g('duplicates_removed')}) | **{final}** |\n",
             "First-pass failure reasons (one pick can fail several checks): "
             + (", ".join(f"{k} {n}" for k, n in reasons.most_common()) or "none") + "\n"]
    return "\n".join(lines)


def write_outputs(out, promos, rejected, errors, meta):
    (out / "promos.json").write_text(json.dumps(promos, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "rejected.json").write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps({**meta, "error_list": errors}, indent=2), encoding="utf-8")

    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "title", "segments", "duration_s", "hook", "cliffhanger", "why",
                    "rank_reason", "score", "multi_episode", "warnings"])
        for p in promos:
            w.writerow([p["rank"], p["id"], p["title"], seg_str(p), p["duration_s"], p["hook"], p["cliffhanger"],
                        p["why"], p["rank_reason"], p["score"], p["multi_episode"], "; ".join(p["warnings"])])

    def category(reason):
        for k, label in [("too short", "too short (<20s)"), ("too long", "too long (>60s)"),
                         ("hook lands", "hook not in first 3s"), ("past episode end", "timestamp outside episode"),
                         ("does not exist", "episode does not exist"), ("ends before", "end before start"),
                         ("duplicate", "duplicate"), ("unreadable", "unreadable segment")]:
            if k in reason:
                return label
        return reason[:40]
    reasons = Counter(category(r["reasons"][0]) for r in rejected if r["reasons"])
    md = [f"# Promo report\n", f"Model: `{meta['model']}` | episodes: {len(meta['episodes'])} | "
          f"valid promos: {meta['valid']} | rejected: {meta['rejected']} | errors: {meta['errors']} | "
          f"time: {meta['seconds']}s\n",
          meta.get("parse_report", ""),
          funnel_md(meta.get("validation", {}), rejected, meta["valid"]),
          "## Ranked promos\n"]
    for p in promos:
        md += [f"### {p['rank']}. {p['title']} (`{p['id']}`, {p['duration_s']}s"
               f"{', multi-episode' if p['multi_episode'] else ''})",
               f"- **Segments:** {seg_str(p)}", f"- **Hook:** {p['hook']}",
               f"- **Cliffhanger:** {p['cliffhanger']}", f"- **Why it works:** {p['why']}",
               f"- **Why this rank:** {p['rank_reason']}"]
        if p["warnings"]:
            md.append(f"- **Checks flagged:** {'; '.join(p['warnings'])}")
        md.append("")
    md += ["## Rejected by validator (first reason)\n"] + [f"- {k}: {v}" for k, v in reasons.most_common()]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder with episode .txt files")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--episodes", help="e.g. 1-5")
    ap.add_argument("--per-ep", type=int, default=3)
    ap.add_argument("--montages", type=int, default=4)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--late-cap", type=int, default=2, help="max late-season clips in the top 10 (-1 = no cap)")
    ap.add_argument("--no-repair", action="store_true", help="skip the fix-and-retry pass")
    ap.add_argument("--model", help="overrides $MODEL, e.g. gemini/gemini-flash-latest or baseline")
    a = ap.parse_args()
    if a.model:
        os.environ["MODEL"] = a.model
    rng = tuple(int(x) for x in a.episodes.split("-")) if a.episodes else None
    if rng and len(rng) == 1:
        rng = (rng[0], rng[0])
    run(a.input, a.out, rng, a.per_ep, a.montages, a.top, repair=not a.no_repair,
        late_cap=None if a.late_cap < 0 else a.late_cap)
