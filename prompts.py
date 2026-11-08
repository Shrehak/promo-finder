"""All prompts in one place, so they are easy to read, tweak and compare."""

SYSTEM = """You are a senior promo editor for vertical microdrama apps (like ReelShort / DramaBox).
You cut 20-60 second promo clips from the show itself, for TikTok / Instagram Reels / YouTube Shorts,
to make cold, scrolling viewers stop and install the app.

A great promo:
1. HOOK: the first 3 seconds stop the scroll (a slap, a reveal, a shocking or humiliating line, a threat, a kiss, a betrayal).
2. CLIFFHANGER: it cuts at an unresolved peak, so the viewer must know what happens next. Never cut after the resolution.
3. STANDALONE: a stranger who has never seen the show understands who is wronged and what is at stake within seconds.
4. Prefer emotionally clear conflicts (humiliation -> revenge, hidden identity, betrayal, forbidden romance).
5. Avoid giving away late-season twists if the promo is for new viewers (spoiler risk).

You only pick timestamps that appear in the script. You never invent lines. You answer ONLY with valid JSON."""


def episode_prompt(ep: dict, story_so_far: str, transcript: str, n: int) -> str:
    return f"""SHOW CONTEXT (episode loglines so far, for your understanding only):
{story_so_far}

EPISODE {ep['ep']} (duration {ep['duration']:.1f}s)
LOGLINE: {ep['logline']}

TIMESTAMPED SCRIPT (format [MM:SS.ss-MM:SS.ss] TYPE/SPEAKER: text):
{transcript}

TASK: Find up to {n} of the best promo clips inside this episode.
- Each promo is 1 continuous segment (or at most 3 segments from THIS episode joined in order), total 20-60 seconds.
- Start each segment exactly at the start time of a script line, end exactly at the end time of a script line.
- The first 3 seconds of the promo must contain the hook.
- The promo must END on the cliffhanger moment (an unresolved peak), not after it is resolved.
- Only return promos that are genuinely good. Returning fewer is fine.

Return JSON:
{{"promos": [
  {{"title": "short catchy internal name",
    "segments": [{{"ep": {ep['ep']}, "start": "MM:SS.ss", "end": "MM:SS.ss"}}],
    "hook": "the exact line or action at the start that stops the scroll",
    "cliffhanger": "the exact line or action the promo ends on, and what is left unresolved",
    "why": "one line on why this works for a cold viewer",
    "scores": {{"hook": 1-5, "cliffhanger": 1-5, "standalone": 1-5, "spoiler_risk": 1-5}}
  }}
]}}"""


def repair_prompt(ep: dict, transcript: str, failures: list) -> str:
    import json
    items = "\n".join(f"- {json.dumps(c, ensure_ascii=False)}\n  PROBLEMS: {'; '.join(r)}" for c, r in failures)
    return f"""EPISODE {ep['ep']} (duration {ep['duration']:.1f}s)
TIMESTAMPED SCRIPT:
{transcript}

Your earlier promo picks for this episode failed automatic checks:
{items}

Fix each one so it passes: total length 20-60 seconds (aim for 30-45; compute end minus start before answering),
the hook line must start within the first 3 seconds, start/end must be real line timestamps, and it must still end
on the unresolved cliffhanger. Extend or move the window using real lines; if a pick cannot be fixed, drop it.

Return JSON in the same format: {{"promos": [ ... ]}}"""


def montage_prompt(loglines: str, moments: str, n: int) -> str:
    return f"""SHOW: all episode loglines in order:
{loglines}

BEATS you can use (beat id | episode | exact length | what it is). "a" beats open a moment (good hooks),
"b" beats end a moment (good cliffhangers):
{moments}

TASK: Build up to {n} MULTI-EPISODE montage promos by choosing 2-5 beats from at least 2 different episodes,
in play order, that tell a mini-story a stranger can follow: a hook beat first, escalation, a cliffhanger beat last.
The lengths of the beats you choose MUST add up to 20-60 seconds: add them up before answering.
Only use beat ids from the list. Do not write timestamps.

Return JSON:
{{"promos": [
  {{"title": "...",
    "beats": ["E03-1a", "E12-2b"],
    "total_seconds": 0.0,
    "hook": "...", "cliffhanger": "...", "why": "...",
    "scores": {{"hook": 1-5, "cliffhanger": 1-5, "standalone": 1-5, "spoiler_risk": 1-5}}
  }}
]}}"""


def rank_prompt(cards: str, k: int) -> str:
    return f"""You are deciding which promos to post FIRST on TikTok / Reels to acquire new viewers for this show.
Below are validated promo candidates (id, length, hook, cliffhanger, why, episodes used).

{cards}

Judge them against each other, as a head of growth would. Prefer: the strongest first-3-second hook,
the most unbearable cliffhanger, instant clarity for a stranger (a clip that needs earlier plot to make sense is
NOT clear), early-episode material (check 'episodes X of N': late episodes spoil twists and confuse strangers; and
viewers who install land near the start), and variety (do not pick 5 versions of the same scene).

Return JSON with your top {k} in posting order:
{{"ranking": [{{"id": "...", "reason": "one line: why post this one at this position"}}]}}"""
