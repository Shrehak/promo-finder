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
    ep = int(m.group(1)) if m else int(re.search(r"(\d+)", path.name).group(1))
    logline = (FIELD("EPISODE LOGLINE").search(raw) or [None, ""])[1].strip()
    continuity = (FIELD("CONTINUITY NOTES").search(raw) or [None, ""])[1].strip()

    blocks, scene, cur = [], 0, None
    for line in raw.splitlines():
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
        if cur is None:
            continue
        if tm := BLOCK_TIME.match(line):
            cur["start"] = to_sec(tm.group(1), tm.group(2))
            cur["end"] = to_sec(tm.group(3), tm.group(4))
            continue
        if line and not line.startswith("---") and cur["start"] is not None:
            cur["text"] = (cur["text"] + " " + line.strip().strip('"')).strip()

    blocks = [b for b in blocks if b["start"] is not None]
    blocks.sort(key=lambda b: (b["start"], b["end"]))
    for i, b in enumerate(blocks):
        b["id"] = i
    duration = max((b["end"] for b in blocks), default=0.0)
    return {"ep": ep, "file": path.name, "logline": logline, "continuity": continuity,
            "duration": duration, "blocks": blocks}


def load_show(folder) -> list[dict]:
    """Load every episode .txt under folder (recursively), sorted by episode number."""
    files = sorted(Path(folder).rglob("*.txt"))
    eps = [parse_episode(f) for f in files]
    eps = [e for e in eps if e["blocks"]]
    return sorted(eps, key=lambda e: e["ep"])


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
