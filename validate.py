"""Rule-based checks on what the LLM returns. LLMs invent timestamps and miscount seconds,
so code (not the model) decides whether a promo is usable.

Checks:
  1. every segment points to a real episode and lies inside it
  2. cut points are snapped so no dialogue line is cut in half
  3. total length is 20-60 s
  4. the hook really appears within the first 3 s (fuzzy text match against the script)
  5. the cliffhanger text appears near the end
Failures go to rejected.json with the reason (that is our quality evidence).
"""
from difflib import SequenceMatcher
from script_parser import fmt, parse_ts

MIN_LEN, MAX_LEN, HOOK_WINDOW = 20.0, 60.0, 3.0


def _sim(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def snap(ep: dict, start: float, end: float) -> tuple[float, float]:
    """Move cut points out of the middle of a dialogue line."""
    for b in ep["blocks"]:
        if b["type"] != "dialogue":
            continue
        if b["start"] < start < b["end"]:
            start = b["start"]
        if b["start"] < end < b["end"]:
            end = b["end"]
    return max(0.0, start), min(ep["duration"], end)


def blocks_in(ep: dict, start: float, end: float) -> list[dict]:
    return [b for b in ep["blocks"] if b["start"] < end and b["end"] > start]


def find_offset(text: str, timeline: list[tuple[float, dict]], last: bool = False):
    """Promo-time offset of the block that best matches `text` (None if no good match)."""
    best, best_t = 0.0, None
    for t, b in (reversed(timeline) if last else timeline):
        s = _sim(text, b["text"])
        if s > best + 1e-9:
            best, best_t = s, t
    return best_t if best >= 0.45 else None


def check(promo: dict, eps_by_num: dict) -> tuple[dict | None, list[str]]:
    """Return (clean promo, warnings) or (None, [reasons])."""
    reasons, warnings, segs = [], [], []
    for s in promo.get("segments") or []:
        try:
            n, a, b = int(s["ep"]), parse_ts(s["start"]), parse_ts(s["end"])
        except (KeyError, ValueError, TypeError):
            reasons.append(f"unreadable segment {s}")
            continue
        ep = eps_by_num.get(n)
        if not ep:
            reasons.append(f"episode {n} does not exist")
            continue
        if b <= a:
            reasons.append(f"ep{n} segment ends before it starts ({s['start']}-{s['end']})")
            continue
        if a > ep["duration"] + 1:
            reasons.append(f"ep{n} start {s['start']} is past episode end {fmt(ep['duration'])}")
            continue
        a2, b2 = snap(ep, a, b)
        if abs(a2 - a) > 0.05 or abs(b2 - b) > 0.05:
            warnings.append(f"ep{n} snapped {fmt(a)}-{fmt(b)} -> {fmt(a2)}-{fmt(b2)}")
        segs.append({"ep": n, "start": round(a2, 2), "end": round(b2, 2)})
    if not segs:
        return None, reasons or ["no segments"]

    total = sum(s["end"] - s["start"] for s in segs)
    if total < MIN_LEN:
        reasons.append(f"too short: {total:.1f}s")
    if total > MAX_LEN:
        reasons.append(f"too long: {total:.1f}s")

    # Build the promo timeline: (seconds from promo start, block)
    timeline, offset = [], 0.0
    for s in segs:
        ep = eps_by_num[s["ep"]]
        for b in blocks_in(ep, s["start"], s["end"]):
            timeline.append((offset + max(0.0, b["start"] - s["start"]), b))
        offset += s["end"] - s["start"]

    hook_t = find_offset(promo.get("hook", ""), timeline)
    if hook_t is None:
        warnings.append("hook text not found in the clip")
    elif hook_t > HOOK_WINDOW:
        reasons.append(f"hook lands at {hook_t:.1f}s, not in the first {HOOK_WINDOW:.0f}s")

    cliff_t = find_offset(promo.get("cliffhanger", ""), timeline, last=True)
    if cliff_t is None:
        warnings.append("cliffhanger text not found in the clip")
    elif total - cliff_t > 12:
        warnings.append(f"clip runs {total - cliff_t:.0f}s past the cliffhanger (may resolve it)")

    if reasons:
        return None, reasons + warnings

    sc = promo.get("scores") or {}
    g = lambda k, d=3: max(1.0, min(5.0, float(sc.get(k, d) or d)))
    score = 0.35 * g("hook") + 0.30 * g("cliffhanger") + 0.25 * g("standalone") - 0.10 * g("spoiler_risk")
    score -= 0.3 * len([w for w in warnings if "not found" in w or "past the cliffhanger" in w])

    clean = {
        "title": promo.get("title", ""),
        "segments": [{"ep": s["ep"], "start": fmt(s["start"]), "end": fmt(s["end"])} for s in segs],
        "duration_s": round(total, 1),
        "hook": promo.get("hook", ""),
        "cliffhanger": promo.get("cliffhanger", ""),
        "why": promo.get("why", ""),
        "scores": {k: g(k) for k in ("hook", "cliffhanger", "standalone", "spoiler_risk")},
        "score": round(score, 3),
        "multi_episode": len({s["ep"] for s in segs}) > 1,
        "warnings": warnings,
        "_secs": segs,
    }
    return clean, warnings


def overlap(p: dict, q: dict) -> float:
    """Share of the shorter promo's time that the two promos have in common."""
    inter = 0.0
    for a in p["_secs"]:
        for b in q["_secs"]:
            if a["ep"] == b["ep"]:
                inter += max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / max(1e-6, min(p["duration_s"], q["duration_s"]))


def dedupe(promos: list[dict], thresh: float = 0.6) -> tuple[list[dict], list[dict]]:
    keep, dropped = [], []
    for p in sorted(promos, key=lambda x: -x["score"]):
        # a montage reuses moments from single-episode promos on purpose: only compare like with like
        if any(p["multi_episode"] == k["multi_episode"] and overlap(p, k) > thresh for k in keep):
            dropped.append(p)
        else:
            keep.append(p)
    return keep, dropped
