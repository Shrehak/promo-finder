"""Streamlit front end: upload a zip of episode scripts -> ranked promos.
Run locally:  streamlit run app.py
"""
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Promo Finder")
st.caption("Reads microdrama screenplays and finds 20-60s promo clips: hook in the first 3s, ends on a cliffhanger.")

with st.sidebar:
    model = st.text_input("Model", os.getenv("MODEL", "gemini/gemini-flash-latest"),
                          help="provider/model, e.g. gemini/gemini-flash-latest, groq/llama-3.3-70b-versatile, "
                               "openai/gpt-4o-mini, anthropic/claude-sonnet-5-5, or 'baseline' (no AI)")
    key = st.text_input("API key (if not set in secrets)", type="password")
    per_ep = st.slider("Candidates per episode", 1, 5, 3)
    montages = st.slider("Multi-episode montages", 0, 6, 4)
    ep_range = st.text_input("Episodes (blank = all, e.g. 1-10)", "")

# Keys from Streamlit secrets (deployment) become env vars for llm.py
for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
    try:
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = st.secrets[k]
    except Exception:
        pass
os.environ["MODEL"] = model
if key:
    env = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY",
           "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(model.split("/")[0])
    if env:
        os.environ[env] = key

import pipeline  # noqa: E402  (after env is set)

up = st.file_uploader("Zip of episode .txt screenplays", type="zip")
if up and st.button("Find promos", type="primary"):
    tmp = Path(tempfile.mkdtemp())
    zipfile.ZipFile(io.BytesIO(up.read())).extractall(tmp / "show")
    rng = None
    if ep_range.strip():
        parts = [int(x) for x in ep_range.split("-")]
        rng = (parts[0], parts[-1])
    status = st.status("Running...", expanded=True)
    try:
        out, promos = pipeline.run(tmp / "show", tmp / "outputs", rng, per_ep, montages,
                                   progress=lambda m: status.write(m))
        status.update(label=f"Done: {len(promos)} promos", state="complete", expanded=False)
        st.session_state["out"] = str(out)
    except BaseException as ex:  # SystemExit from 'no episodes found' too
        status.update(label="Failed", state="error")
        st.error(str(ex))

if "out" in st.session_state:
    out = Path(st.session_state["out"])
    promos = json.loads((out / "promos.json").read_text(encoding="utf-8"))
    rows = [{"rank": p["rank"], "id": p["id"], "title": p["title"],
             "segments": pipeline.seg_str(p), "sec": p["duration_s"], "hook": p["hook"],
             "cliffhanger": p["cliffhanger"], "why": p["why"], "why this rank": p["rank_reason"],
             "flags": "; ".join(p["warnings"])} for p in promos]
    st.subheader("Ranked promos")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out))
    st.download_button("Download all outputs (zip)", buf.getvalue(), f"{out.name}.zip")
    with st.expander("Rejected candidates (why the validator threw them out)"):
        st.json(json.loads((out / "rejected.json").read_text(encoding="utf-8")))
