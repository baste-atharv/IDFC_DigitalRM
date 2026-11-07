# FIRST Saathi: the Connector's Digital RM

FAME 6.0 (IDFC FIRST Bank), Problem Statement 4: a Digital RM for the Connector Channel, without losing the human touch.

Every connector gets a digital twin of **their own** RM. It talks in the RM's style and language, can send voice notes in the RM's cloned voice (only with the RM's written consent), checks leads and payouts, matches the right product to a customer, nudges at the right time, and hands the conversation to the human RM when trust is at stake. It always says it is the RM's AI assistant.

**All demo data is synthetic**: 240 fictional connectors, 6 fictional RMs, leads and payouts. Product facts marked [SOURCED] in `kb/` come from public IDFC FIRST material; the rest is illustrative.

## Try it online
Open the Streamlit app link. It runs in offline mode with no key. To try the LLM brain or voices, add your own key on page **7. Settings**: on the online demo, keys stay in your browser session and are never saved on the server.

## Run it on Windows (one click)
| File | What it does |
|---|---|
| `SETUP_AND_RUN.bat` | First run: finds or installs Python, creates `.venv`, installs packages, opens the app at http://localhost:8501 |
| `RUN_APP.bat` | Opens the app again later |
| `RUN_TESTS.bat` | 20 checks: offline behaviour in 3 languages, router, consent gate, hash-chained logs, and the Anthropic and OpenAI tool loops through a fake transport (no key needed) |
| `BUILD_RAFT_DATASET.bat` | Writes `data\raft_dataset.jsonl` for fine-tuning an in-house model |
| `setup_env.bat` | Helper the others call |

Any other OS: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, `streamlit run app.py`.

## Keys (page 7, Settings)
| What | Options | Needed? |
|---|---|---|
| Brain (LLM) | Anthropic Claude, OpenAI, Google Gemini, Groq, OpenRouter, or Ollama on your own PC (local runs only) | Optional. Without it the app replies with rule-based templates in English, Hindi and Marathi |
| Voice out | edge-tts (free, no key), ElevenLabs (cloned voice), Sarvam (Indian voices) | Optional |
| Voice in | Sarvam speech-to-text (22+ Indian languages) or OpenAI transcription | Optional |

Never commit keys to this repository.

## Screens
1. **Connector app**: chat in any language, voice notes, today's nudge from the RM, payout tracker, tier and streak, "Talk to RM".
2. **Add a lead**: describe the customer in your words; get the product, indicative eligibility and documents; consent tick required.
3. **RM cockpit**: hand-offs from the Digital RM, connectors to call this week, portfolio by stage.
4. **Persona studio**: RM call notes become a style card; consented voice samples become a cloned voice; kill switch.
5. **Nudge engine**: per-connector nudge, behavioural principle, channel, send time, weekly cap and quiet hours.
6. **Quality and governance**: 10 behaviour checks, guardrails, hash-chained logs, RAFT dataset.
7. **Settings**: API keys.
8. **How it is built**: design patterns and honesty notes.

## Going live
Point `saathi/data.py` at the MyFIRST Partner app, the loan origination system and the payout system. Load the bank's product sheets and connector code of conduct into `kb/`. Get each RM's written voice consent. Re-validate behaviour with the checks on page 6 before any connector sees it.
