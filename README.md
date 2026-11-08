# Promo Finder (Kuku, Problem 1)

Reads a folder of microdrama screenplays and finds 20-60 s promo clips: a hook in the first 3 s, a cliffhanger
at the end, and a clip that makes sense on its own. For each promo it outputs the episode, start/end of every
segment, the hook, the cliffhanger, why it works, and a posting rank.

## Run

```bash
# CLI: no install needed (Python 3.10+, standard library only)
# put GEMINI_API_KEY=... in .env (git-ignored); free key: https://aistudio.google.com/apikey
python pipeline.py --input data/show_folder
python pipeline.py --input data/show_folder --episodes 1-5      # quick test
python pipeline.py --input data/show_folder --model baseline    # no AI, keyword baseline

# Web app
pip install -r requirements.txt
streamlit run app.py                 # upload a zip of the episode .txt files
```

Outputs go to `outputs/<time>_<model>/`:
- `report.md`: the ranked list
- `promos.csv` / `promos.json`
- `rejected.json`: candidates the validator threw out, with the reason
- `raw_llm/`: every unedited model response
- `run_meta.json`

## Swap the model (one env var)
`MODEL=gemini/gemini-flash-latest` (default) · `gemini/gemini-flash-lite-latest` (fast) · `groq/llama-3.3-70b-versatile` · `openai/gpt-4o-mini` ·
`anthropic/claude-sonnet-5-5` · `openrouter/<id>` · `ollama/<local>` · `compat/<model>` with `LLM_BASE_URL` + `LLM_API_KEY`
(any OpenAI-compatible endpoint) · `baseline` (no AI). Every call goes through `llm.call()` in `llm.py`.

## How it works
| Step | Done by | File |
|---|---|---|
| 1. Parse scripts into timestamped blocks (dialogue / action / camera / sfx) | code (regex) | `script_parser.py` |
| 2. Up to 3 promo candidates per episode, with story-so-far loglines as context | LLM, 1 call per episode | `prompts.py` |
| 3. Validate: real episode, cut points snapped so no line is cut mid-sentence, 20-60 s, hook found in first 3 s, cliffhanger near the end | code | `validate.py` |
| 4. Multi-episode montages built from the validated moments | LLM, 1 call | `prompts.py` |
| 5. Remove near-duplicates (>60% time overlap) | code | `validate.py` |
| 6. Rank the top 25 head-to-head ("which do we post first?") | LLM judge, 1 call | `prompts.py` |

That's about 52 calls per 50-episode show. Responses are cached, so re-runs are free and repeatable.
On a rate limit the call retries with backoff, and one failed episode does not stop the run.

Data note: the episode scripts are confidential, so `data/` and `outputs/` are git-ignored.

## Quality checking
```bash
python judge.py   --input data/show --run outputs/<run> --top 10   # blind cold-viewer test (a different model sees only the clip)
python review.py  --input data/show --run outputs/<run> --top 10   # clips next to their script lines + a sheet for human review
python compare.py                                                  # outputs/COMPARISON.md: all runs side by side + sample AI mistakes
```
Run history for the sample show (all in `outputs/`): v1 no retry → v2 fix-and-retry → v3 late-season spoiler cap →
v4/v5 beat-based montages (final). Each change was made because a check above showed a specific failure.
