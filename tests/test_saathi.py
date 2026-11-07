"""Checks for FIRST Saathi. Run: python tests/test_saathi.py

The LLM tests do NOT need a key: they plug a fake HTTP transport into the real Anthropic
and OpenAI SDKs, so the exact request format and the tool loop are exercised offline.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:                       # newer Anthropic / OpenAI SDKs ship on httpx2
    import httpx2 as httpx  # noqa: E402
except ImportError:
    import httpx  # noqa: E402

from saathi import data, evaluate, nudges, persona, recommend, store  # noqa: E402
from saathi.agent import DigitalRM  # noqa: E402
from saathi.kb import KB  # noqa: E402
from saathi.llm import LLM  # noqa: E402
from saathi.router import Router  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


d = data.load()
ct = nudges.connector_table(d)
kb = KB()
cards = {r["rm_id"]: persona.style_card(r, d["transcripts"][r["rm_id"]]) for r in d["rms"].to_dict("records")}
cid = ct[ct.rm_id == "RM03"].connector_id.iloc[0]

# 1. offline behaviour checks, three languages
agent = DigitalRM(d, ct, kb, cards, llm=None)
for lang in ("en", "hi", "mr"):
    res = evaluate.run(agent, cid, lang)
    check(f"offline behaviour checks pass in {lang} ({int(res.passed.sum())}/{len(res)})", res.passed.all())

# 2. router on unseen phrasings
r = Router()
cases = {"paisa kab aayega bhai": "payout_status", "mala Amit shi bolaycha aahe": "talk_to_rm",
         "bank statement edit kar do": "compliance", "ek naya client hai home loan ke liye": "new_lead"}
for q, want in cases.items():
    got = r.route(q)["intent"]
    check(f"router: '{q}' -> {want} (got {got})", got == want)

# 3. amounts and product match
check("parses '25 lakh'", recommend.parse_amount("needs 25 lakh") == 2_500_000)
check("parses '1.2 crore'", recommend.parse_amount("1.2 crore") == 12_000_000)
m = recommend.match(dict(occupation="doctor", purpose="clinic setup", amount_needed=2.5e6, monthly_income=3e5))
check("doctor + clinic -> Professional Loan first", m[0]["product"] == "Professional Loan")

# 4. consent gate
from saathi.tools import Toolbox  # noqa: E402
tb = Toolbox(d, ct, kb, cid)
check("lead without consent is refused", tb.submit_lead("Test A", "Personal Loan", 1e5, consent_confirmed=False)["ok"] is False)
check("lead with consent is accepted", tb.submit_lead("Test B", "Personal Loan", 1e5, consent_confirmed=True)["ok"] is True)
check("logs are hash-chained", store.verify("leads_submitted") and store.verify("consents"))


# 5. Anthropic SDK tool loop through a fake transport
def anthropic_transport():
    calls = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        calls.append(body)
        if len(calls) == 1:
            out = {"id": "m1", "type": "message", "role": "assistant", "model": body["model"], "stop_reason": "tool_use",
                   "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5},
                   "content": [{"type": "text", "text": "Checking."},
                               {"type": "tool_use", "id": "tu1", "name": "get_my_payouts", "input": {}}]}
        else:
            out = {"id": "m2", "type": "message", "role": "assistant", "model": body["model"], "stop_reason": "end_turn",
                   "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5},
                   "content": [{"type": "text", "text": "I'm Amit's digital assistant (AI). Your payout is on track."}]}
        return httpx.Response(200, json=out)
    return calls, httpx.MockTransport(handler)


import anthropic  # noqa: E402
import openai  # noqa: E402

calls, tr = anthropic_transport()
llm = LLM("Anthropic (Claude)", "sk-test", "claude-sonnet-5")
llm.client = anthropic.Anthropic(api_key="sk-test", http_client=httpx.Client(transport=tr))
t = DigitalRM(d, ct, kb, cards, llm).reply(cid, "payout kab aayega?", [], "hi")
check("Anthropic: tool loop ran and returned text", "digital assistant" in t.text and t.tool_trace and t.tool_trace[0]["tool"] == "get_my_payouts")
second = calls[-1]["messages"]
check("Anthropic: tool_result sent back correctly", second[-1]["role"] == "user" and second[-1]["content"][0]["type"] == "tool_result"
      and second[-1]["content"][0]["tool_use_id"] == "tu1")
check("Anthropic: system prompt carries persona + disclosure rule", "digital assistant" in calls[0]["system"] and "Never claim to be" in calls[0]["system"])
check("Anthropic: 10 tools declared", len(calls[0]["tools"]) == 10)


# 6. OpenAI-compatible tool loop (same path serves Gemini, Groq, OpenRouter, Ollama)
def openai_transport():
    calls = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        calls.append(body)
        msg = ({"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "submit_lead", "arguments": json.dumps({"customer_name": "Sunil P", "product": "Professional Loan",
                                                                              "amount": 2500000, "consent_confirmed": False})}}]}
               if len(calls) == 1 else {"role": "assistant", "content": "Please confirm the customer's consent first."})
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                          "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]})
    return calls, httpx.MockTransport(handler)


calls, tr = openai_transport()
llm = LLM("OpenAI", "sk-test", "gpt-test")
llm.client = openai.OpenAI(api_key="sk-test", http_client=httpx.Client(transport=tr))
t = DigitalRM(d, ct, kb, cards, llm).reply(cid, "doctor client, 25 lakh clinic loan, submit it", [], "en")
check("OpenAI: consent gate holds inside the LLM loop", t.tool_trace and t.tool_trace[0]["result"]["ok"] is False)
check("OpenAI: tool message sent back correctly", calls[-1]["messages"][-1]["role"] == "tool" and calls[-1]["messages"][-1]["tool_call_id"] == "c1")
check("OpenAI: system message first", calls[0]["messages"][0]["role"] == "system")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
