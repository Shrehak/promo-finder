"""One function, any model: call(prompt) -> dict (parsed JSON).

Swap models with one env var, no code change:
  MODEL=gemini/gemini-flash-latest        (free tier, key: GEMINI_API_KEY)
  MODEL=groq/llama-3.3-70b-versatile      (free tier, key: GROQ_API_KEY)
  MODEL=openrouter/<any model id>         (key: OPENROUTER_API_KEY)
  MODEL=openai/gpt-4o-mini                (key: OPENAI_API_KEY)
  MODEL=anthropic/claude-sonnet-5-5       (key: ANTHROPIC_API_KEY)
  MODEL=ollama/llama3.1                   (local, no key)
  MODEL=compat/<model>  + LLM_BASE_URL + LLM_API_KEY   (any OpenAI-compatible API, e.g. one Kuku provides)

Standard library only (urllib), so there is nothing to install for the CLI.
Responses are cached on disk by hash(model + prompt): re-runs are free and reproducible.
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

def _load_dotenv(path: Path = Path(__file__).with_name(".env")):
    """Read KEY=VALUE lines from .env (never committed) without overriding real env vars."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() and not k.strip().startswith("#") and v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_dotenv()
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".llm_cache"))
OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "compat": (os.getenv("LLM_BASE_URL", ""), "LLM_API_KEY"),
}
_last_call = 0.0
_rate_lock = threading.Lock()


class ConfigError(RuntimeError):
    """Missing key / unknown provider: fail fast, retrying will not help."""


def _key(env: str) -> str:
    v = os.getenv(env, "").strip()
    if not v:
        raise ConfigError(f"Set {env} (or switch MODEL). See .env.example")
    return v


def model_name() -> str:
    return os.getenv("MODEL", "gemini/gemini-flash-latest")


def _post(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def _raw_call(prompt: str, system: str, model: str) -> str:
    provider, _, name = model.partition("/")
    temp = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    if provider == "gemini":
        key = _key("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent?key={key}"
        if name.startswith("gemma"):  # Gemma on this API: no system role / JSON mode, so inline it
            body = {"contents": [{"role": "user", "parts": [{"text": system + "\n\n" + prompt}]}],
                    "generationConfig": {"temperature": temp}}
        else:
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temp, "responseMimeType": "application/json"}}
        r = _post(url, body, {})
        return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"] if not p.get("thought"))

    if provider == "anthropic":
        r = _post("https://api.anthropic.com/v1/messages",
                  {"model": name, "max_tokens": 8000, "temperature": temp, "system": system,
                   "messages": [{"role": "user", "content": prompt}]},
                  {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in r["content"])

    if provider in OPENAI_COMPAT:
        base, key_env = OPENAI_COMPAT[provider]
        headers = {"Authorization": f"Bearer {_key(key_env)}"} if key_env else {}
        body = {"model": name, "temperature": temp,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}]}
        if provider != "ollama":
            body["response_format"] = {"type": "json_object"}
        r = _post(base.rstrip("/") + "/chat/completions", body, headers)
        return r["choices"][0]["message"]["content"]

    raise ConfigError(f"Unknown provider in MODEL={model!r}")


def extract_json(text: str):
    """Models sometimes wrap JSON in ``` fences or add chatter. Pull out the JSON."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def call(prompt: str, system: str = "", tag: str = "call", raw_dir: Path | None = None, model: str | None = None):
    """Call the configured model, return parsed JSON. Retries on rate limits / bad JSON."""
    global _last_call
    model = model or model_name()
    key = hashlib.sha256(f"{model}\n{system}\n{prompt}".encode()).hexdigest()[:24]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{key}.txt"

    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        rpm = float(os.getenv("LLM_RPM", "8"))  # stay under free-tier rate limits
        text, err = None, None
        for attempt in range(6):
            with _rate_lock:  # space out request *starts* across threads
                wait = 60 / rpm - (time.time() - _last_call)
                if wait > 0:
                    time.sleep(wait)
                _last_call = time.time()
            try:
                text = _raw_call(prompt, system, model)
                extract_json(text)  # validate before caching
                break
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
                if e.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(err)
                time.sleep(min(60, 5 * 2 ** attempt))
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                err = f"bad response: {e}"
                text = None
        if text is None:
            raise RuntimeError(f"LLM failed after retries ({tag}): {err}")
        cache.write_text(text, encoding="utf-8")

    if raw_dir:  # keep the unedited model output for the 'raw outputs' folder
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{tag}.json").write_text(text, encoding="utf-8")
    return extract_json(text)
