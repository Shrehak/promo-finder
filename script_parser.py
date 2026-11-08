"""Parse Kuku microdrama screenplay .txt files into structured blocks.

No AI here: the format is regular, so plain regex is cheaper, faster and exact.

Each episode becomes:
  {"ep": 1, "file": "...", "logline": "...", "continuity": "...", "duration": 138.0,
   "blocks": [{"id": 0, "scene": 1, "type": "dialogue"|"camera"|"action"|"sfx",
               "speaker": "MATRIARCH", "emotion": "Cold, Commanding",
               "start": 11.3, "end": 13.1, "text": "You should turn around, Juliana."}, ...]}
"""
import re
from pathlib import Path

TS = r"(\d+):(\d+(?:\.\d+)?)"
BLOCK_HEAD = re.compile(r"^\*\*(.+?):\*\*\s*$")
BLOCK_TIME = re.compile(rf"^Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)")
SCENE = re.compile(r"^SCENE\s+(\d+)")
EP_NUM = re.compile(r"EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
FIELD = lambda name: re.compile(rf"^{name}:\s*(.*)$", re.M)

TYPE_MAP = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}


def to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def fmt(t: float) -> str:
    """Seconds -> 'MM:SS.ss' (same style as the scripts)."""
    m, s = divmod(round(t, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


def parse_ts(x) -> float:
    """'01:14.50' or '1:14' or 74.5 -> seconds."""
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip().strip("()")
    if ":" in x:
        m, s = x.split(":", 1)
        return int(m) * 60 + float(s)
    return float(x)


def parse_episode(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    m = EP_NUM.search(raw)
    issues = []
    if m:
        ep = int(m.group(1))
    else:
        fm = re.search(r"(\d+)", path.name)
        ep = int(fm.group(1)) if fm else -1
        issues.append("no 'EPISODE NUMBER:' line; episode number taken from file name" if fm else
                      "no episode number found anywhere")
    logline = (FIELD("EPISODE LOGLINE").search(raw) or [None, ""])[1].strip()
    continuity = (FIELD("CONTINUITY NOTES").search(raw) or [None, ""])[1].strip()
    if not logline:
        issues.append("no EPISODE LOGLINE (story-so-far context will be thinner)")

    blocks, scene, cur, unparsed = [], 0, None, []
    for n, line in enumerate(raw.splitlines(), 1):
        line = line.rstrip()
        if sm := SCENE.match(line):
            scene = int(sm.group(1))
            cur = None  # scene header lines are not part of the previous block
            continue
        if hm := BLOCK_HEAD.match(line):
            head = hm.group(1).strip()
            # "JULIANA (Desperate, Unbelieving)" -> speaker + emotion
            em = re.match(r"^(.*?)\s*\((.*)\)\s*$", head)
            name, emotion = (em.group(1), em.group(2)) if em else (head, "")
            btype = TYPE_MAP.get(name.upper(), "dialogue")
            cur = {"scene": scene, "type": btype,
                   "speaker": name if btype == "dialogue" else "",
                   "emotion": emotion, "start": None, "end": None, "text": ""}
            blocks.append(cur)
            continue
        if tm := BLOCK_TIME.match(line):
            if cur is not None:
                cur["start"] = to_sec(tm.group(1), tm.group(2))
                cur["end"] = to_sec(tm.group(3), tm.group(4))
            else:
                unparsed.append((n, line))
            continue
        if cur is None:
            if line and not KNOWN_META.match(line):
                unparsed.append((n, line))  # keep it visible instead of silently dropping it
            continue
        if line and not line.startswith("---"):
            if cur["start"] is None:
                unparsed.append((n, line))  # text before the block's timestamp
            else:
                cur["text"] = (cur["text"] + " " + line.strip().strip('"')).strip()

    no_time = [b for b in blocks if b["start"] is None]
    bad_time = [b for b in blocks if b["start"] is not None and b["end"] <= b["start"]]
    if no_time:
        issues.append(f"{len(no_time)} blocks had no 'Start: (..) | End: (..)' line and were dropped "
                      f"(e.g. {(no_time[0]['speaker'] or no_time[0]['type'])!r})")
    if bad_time:
        issues.append(f"{len(bad_time)} blocks had end <= start and were dropped "
                      f"(e.g. {fmt(bad_time[0]['start'])}-{fmt(bad_time[0]['end'])})")
    if unparsed:
        issues.append(f"{len(unparsed)} lines not recognised (e.g. line {unparsed[0][0]}: {unparsed[0][1][:60]!r})")
    empty = sum(1 for b in blocks if b["start"] is not None and not b["text"])
    if empty:
        issues.append(f"{empty} blocks have a timestamp but no text")

    blocks = [b for b in blocks if b["start"] is not None and b["end"] > b["start"]]
    blocks.sort(key=lambda b: (b["start"], b["end"]))
    for i, b in enumerate(blocks):
        b["id"] = i
    duration = max((b["end"] for b in blocks), default=0.0)
    return {"ep": ep, "file": path.name, "logline": logline, "continuity": continuity,
            "duration": duration, "blocks": blocks, "issues": issues}


KNOWN_META = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES|Scene heading|Scene start|Scene end|"
                        r"Actions and dialogue)\s*:|^---", re.I)


def load_show_with_report(folder) -> tuple[list[dict], dict]:
    """Parse every episode .txt under folder and report anything that did not parse cleanly.
    Fails visibly: a new show in a different format shows up here, not as silently bad promos."""
    files = sorted(Path(folder).rglob("*.txt"))
    parsed = [parse_episode(f) for f in files]
    eps = sorted([e for e in parsed if e["blocks"]], key=lambda e: e["ep"])
    skipped = [e["file"] for e in parsed if not e["blocks"]]
    nums = [e["ep"] for e in eps]
    dupes = sorted({n for n in nums if nums.count(n) > 1})
    gaps = sorted(set(range(min(nums), max(nums) + 1)) - set(nums)) if nums else []
    report = {
        "files_found": len(files), "episodes_parsed": len(eps), "blocks": sum(len(e["blocks"]) for e in eps),
        "skipped_files_no_timestamped_blocks": skipped, "duplicate_episode_numbers": dupes,
        "missing_episode_numbers": gaps,
        "episodes_with_issues": {e["ep"]: e["issues"] for e in eps if e["issues"]},
        "very_short_episodes(<20s)": [e["ep"] for e in eps if e["duration"] < 20],
    }
    report["ok"] = not (skipped or dupes or gaps or report["episodes_with_issues"]
                        or report["very_short_episodes(<20s)"] or not eps)
    return eps, report


def parse_report_md(r: dict) -> str:
    head = (f"## Input check\n\n{r['files_found']} files -> {r['episodes_parsed']} episodes, {r['blocks']} timestamped "
            f"lines. " + ("**All clean.**\n" if r["ok"] else "**Problems found, check before trusting the promos:**\n"))
    items = []
    for k in ("skipped_files_no_timestamped_blocks", "duplicate_episode_numbers", "missing_episode_numbers",
              "very_short_episodes(<20s)"):
        if r[k]:
            items.append(f"- {k.replace('_', ' ')}: {r[k][:10]}")
    for ep, iss in list(r["episodes_with_issues"].items())[:10]:
        items.append(f"- ep{ep}: " + "; ".join(iss))
    return head + "\n".join(items) + "\n"


def load_show(folder) -> list[dict]:
    """Load every episode .txt under folder (recursively), sorted by episode number."""
    return load_show_with_report(folder)[0]


def to_transcript(ep: dict, include_camera: bool = True) -> str:
    """Compact, timestamped text the LLM reads. One line per block."""
    out = []
    for b in ep["blocks"]:
        if b["type"] == "camera" and not include_camera:
            continue
        t = f"[{fmt(b['start'])}-{fmt(b['end'])}]"
        if b["type"] == "dialogue":
            emo = f" ({b['emotion']})" if b["emotion"] else ""
            out.append(f'{t} {b["speaker"]}{emo}: "{b["text"]}"')
        else:
            out.append(f"{t} {b['type'].upper()}: {b['text']}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    show = load_show(sys.argv[1])
    print(f"{len(show)} episodes, {sum(len(e['blocks']) for e in show)} blocks")
    for e in show[:3]:
        print(e["ep"], fmt(e["duration"]), len(e["blocks"]), e["logline"][:80])
