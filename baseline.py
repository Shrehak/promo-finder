"""A no-AI baseline: keyword intensity scoring. Two uses:
  1. lets the whole pipeline run offline (MODEL=baseline) for testing, at zero cost
  2. a yardstick: the LLM should clearly beat this, or it is not earning its cost
"""
import re
from script_parser import fmt, parse_ts

STRONG = r"slap|strik|punch|kiss|scream|gasp|blood|knife|sword|dagger|pistol|gun|shatter|collapse|faint|" \
         r"tear|sob|kneel|grab|shove|choke|throw|fall|reveal|betray|crash|seiz|drag|dead|death|kill|poison|" \
         r"pregnan|divorce|duke|heir|secret|guard"
EMO = r"furious|shock|horrif|terrif|desperate|venom|cruel|mock|sneer|cold|menac|rage|scream|disgust|smug|outrag|broken"


def intensity(b: dict) -> float:
    t = b["text"].lower()
    s = 1.5 * len(re.findall(STRONG, t)) + 1.0 * len(re.findall(EMO, b["emotion"].lower()))
    s += 0.5 * t.count("!") + (0.5 if "?" in t and b["type"] == "dialogue" else 0)
    s += {"sfx": 0.5, "action": 0.5, "dialogue": 0.3}.get(b["type"], 0)
    return s


def candidates(ep: dict, n: int = 3) -> list[dict]:
    blocks = [b for b in ep["blocks"] if b["type"] != "camera"]
    if not blocks:
        return []
    out = []

    def promo(hook_b, end_b, title):
        return {"title": title,
                "segments": [{"ep": ep["ep"], "start": fmt(hook_b["start"]), "end": fmt(end_b["end"])}],
                "hook": hook_b["text"], "cliffhanger": end_b["text"],
                "why": "baseline: high keyword intensity at start and end",
                "scores": {"hook": min(5, 1 + intensity(hook_b)), "cliffhanger": min(5, 1 + intensity(end_b)),
                           "standalone": 3, "spoiler_risk": 1 + min(4, ep["ep"] // 12)}}

    # 1) Episode ending: microdrama episodes are written to end on a cliffhanger.
    last = blocks[-1]
    starts = [b for b in blocks if 25 <= last["end"] - b["start"] <= 55]
    if starts:
        out.append(promo(max(starts, key=intensity), last, f"Ep{ep['ep']} ending"))

    # 2) Most intense moments as hooks, ending on the most intense line 20-60s later.
    for hb in sorted(blocks, key=intensity, reverse=True):
        if len(out) >= n:
            break
        ends = [b for b in blocks if 22 <= b["end"] - hb["start"] <= 55]
        if not ends:
            continue
        eb = max(ends, key=intensity)
        if any(abs(hb["start"] - parse_ts(o["segments"][0]["start"])) < 15 for o in out):
            continue  # too close to a promo we already have
        out.append(promo(hb, eb, f"Ep{ep['ep']} peak"))
    return out
