"""Persona cards: clone the RM's way of talking, not the RM.

Persona-prompt pattern: a person's own words become a persona the bot speaks in.
Here an RM's call notes / WhatsApp scripts become a persona
card: greeting, sign-offs, signature phrases, language mix, sentence length. With an LLM
key the card is enriched by the model; without one, it is extracted by simple statistics.

Guardrail baked into every prompt: the bot is the RM's DIGITAL ASSISTANT and says so.
It never claims to be the human RM (RBI FREE-AI: tell users they are dealing with AI).
"""
from __future__ import annotations

import json
import re
from collections import Counter

from . import config as C

HINDI_MARKERS = {"hai", "ji", "aap", "kijiye", "nahi", "bhai", "kar", "ke", "mein", "ho", "sab", "aur", "toh", "fikar", "chaliye"}
MARATHI_MARKERS = {"aahe", "tumcha", "tumhi", "kara", "naka", "jhala", "chala", "mag", "ek", "baghuya", "kasa", "tai", "saheb"}


def style_card(rm: dict, transcripts: list[str]) -> dict:
    words = [w.lower().strip(".,!?:;") for t in transcripts for w in t.split()]
    n = max(len(words), 1)
    hi = sum(w in HINDI_MARKERS for w in words) / n
    mr = sum(w in MARATHI_MARKERS for w in words) / n
    greet = Counter(t.split()[0].strip("!,") for t in transcripts if "!" in t[:25]).most_common(2)
    cues = ("tension", "fikar", "fikr", "worry", "bilkul", "simple", "cricket", "boond", "steps", "consistency",
            "steady", "thoda", "kamaal", "chhan", "super", "permission", "parvangi")
    phrases = []
    for t in transcripts:
        if any(c in t.lower() for c in cues):
            clause = re.split(r"[,.!?]", t)[0].strip()
            phrases.append(" ".join(clause.split()[:7]))
    avg_len = round(sum(len(t.split()) for t in transcripts) / max(len(transcripts), 1), 1)
    metaphors = [t for t in transcripts if any(k in t.lower() for k in ("cricket", "boond", "match", "game", "steps", "thoda"))]
    return dict(
        rm_id=rm["rm_id"], rm_name=rm["name"], city=rm["city"],
        greeting=", ".join(g for g, _ in greet) or "Hello",
        language_mix=f"Hindi markers {hi:.0%}, Marathi markers {mr:.0%}, rest English",
        avg_words_per_line=avg_len,
        signature_phrases=phrases[:6],
        motivational_line=metaphors[0] if metaphors else "",
        tone="warm, respectful (uses ji / tai / garu / sir), short sentences, step-by-step reassurance, celebrates wins",
        always="reassures before explaining; breaks tasks into one-by-one steps; reminds about customer consent",
        source="statistics on call notes (offline)",
    )


def enrich_with_llm(card: dict, transcripts: list[str], llm) -> dict:
    prompt = ("Read these call notes from a bank relationship manager and describe HOW they talk, "
              "as JSON with keys: tone, greeting, signature_phrases (list of up to 6), motivational_style, "
              "language_mix, do (list), dont (list). Do not invent facts about the person.\n\n" + "\n".join(transcripts))
    try:
        txt = llm.complete(prompt, system="You extract communication style. Reply with JSON only.", max_tokens=500)
        j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        card = {**card, **{k: v for k, v in j.items() if v}, "source": "LLM-enriched style card"}
    except Exception as e:  # keep the offline card
        card = {**card, "llm_note": f"LLM enrichment failed: {e}"}
    return card


def system_prompt(card: dict, connector: dict, lang: str, knowledge: str, today: str = C.TODAY) -> str:
    lname, script = C.LANGS.get(lang, C.LANGS["en"])[:2]
    first = card["rm_name"].split()[0]
    return f"""You are "Saathi", the AI digital assistant of {card['rm_name']}, IDFC FIRST Bank relationship manager for connectors in {card['city']}.
You help a connector (a referral partner who brings loan customers to the bank) the way {first} would.

SPEAK LIKE {first.upper()} (style card, learned from {first}'s call notes):
- tone: {card.get('tone')}
- greeting: {card.get('greeting')}; signature phrases: {', '.join(card.get('signature_phrases', [])[:6])}
- motivational line he/she uses: {card.get('motivational_line', '')}
- {card.get('always', '')}

RULES (never break):
1. In your FIRST reply of a conversation say you are {first}'s digital assistant (AI). Never claim to be {first}.
2. Use only facts from tool results and KNOWLEDGE below. If unsure, say so and offer to connect to {first}.
3. Never promise approval, interest rates or payouts. Eligibility is always "indicative".
4. Never help alter documents, bypass consent or pressure customers. Coach on fair practice and call flag_compliance.
5. Call handoff_to_rm for payout disputes, anger or distress, requests to talk to {first}, or loans above Rs 50 lakh.
6. Before submit_lead, the connector must confirm the customer agreed to share details (consent_confirmed=true).
7. Reply in {lname} ({script}). WhatsApp style: warm, short (max 80 words), one clear next step.

CONNECTOR: {json.dumps(connector, ensure_ascii=False)}
TODAY: {today}

KNOWLEDGE (cite as [doc > section] when you use it):
{knowledge or '(none retrieved)'}"""


STANDALONE_PROMPT = """Rewrite the connector's latest message as a standalone question, using the chat history only if the
message depends on it (e.g. "and his?", "what about the payout?"). If it is a greeting or already standalone, return it unchanged.
Return only the question.

Chat history:
{history}

Latest message: {question}
Standalone question:"""
