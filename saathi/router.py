"""Intent router: classify first, then answer.

Classify-then-route pattern: every query is classified before it is answered.
Here the routes are what a
connector actually asks an RM. Character n-grams cope with English, Hinglish and Marathi
written in Latin script. With an LLM key, the LLM's tool calls do the routing and this
classifier is kept for analytics and as the offline fallback.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

EXAMPLES = {
    "lead_status": ["status of my leads", "mera lead kahan tak pahuncha", "Patil cha loan kuthe aahe", "lead update please",
                    "kitne lead sanction hue", "is my client's file logged in", "which documents are pending for my lead",
                    "lead ka kya hua", "majha lead disburse jhala ka", "show my pipeline", "Sunil ka loan approve hua kya",
                    "my customer's loan status", "file kab login hogi"],
    "payout_status": ["payout kab aayega", "my payment not received", "paisa kab milega", "payout hold kyu hai",
                      "commission status", "when will I get my payout", "majha payout kadhi yeil", "how much did I earn",
                      "is mahine kitna payout", "payout on hold why", "Friday ko paisa aayega kya", "earnings this month"],
    "new_lead": ["naya customer hai", "I have a client who needs 10 lakh", "ek client la business loan pahije",
                 "add a new lead", "doctor ko clinic ke liye loan chahiye", "my friend wants a home loan of 40 lakh",
                 "refer a customer", "ek customer aaya hai car loan ke liye", "new lead for personal loan",
                 "client needs working capital 15 lakh", "majhya mitrala ghar ghyaycha aahe loan pahije"],
    "product_query": ["personal loan ki details", "what documents for home loan", "LAP kya hota hai", "education loan details",
                      "credit card eligibility", "business loan maximum amount", "professional loan for CA",
                      "what is the tenure of personal loan", "used car loan documents", "home loan ke liye kya chahiye",
                      "which product is best for a doctor", "rate kitna hai"],
    "lead_ideas": ["lead kaise laun", "how to get more leads", "ideas for this week", "kisko approach karu",
                   "how can I find customers", "lead generation tips", "konala vicharu", "what should I sell this month",
                   "customers kahan milenge", "give me some lead ideas"],
    "issue": ["app not working", "login problem", "wrong payout amount", "I want to raise a complaint", "lead stuck for 10 days",
              "app crash ho raha hai", "OTP nahi aa raha", "payout galat aaya", "problem with my account", "tar app chalat nahi"],
    "talk_to_rm": ["Rahul sir se baat karni hai", "call me please", "connect me to my RM", "manager ko bulao",
                   "mala Rahul shi bolaycha aahe", "I want to talk to a human", "insaan se baat karao", "RM ka number do"],
    "progress": ["my points", "tier kya hai", "what is my streak", "leaderboard", "how far from gold tier",
                 "majha rank kay aahe", "my rewards", "kitne leads is mahine"],
    "compliance": ["salary slip thoda adjust kar do", "can you change the bank statement", "customer ko bina bataye document bhej du",
                   "CIBIL manage karo", "can we edit the ITR", "approval guarantee de do customer ko",
                   "customer se processing fee le lu", "statement madhe badal kara"],
    "smalltalk": ["hi", "hello", "namaste", "thank you", "kaise ho", "good morning", "namaskar", "thanks a lot", "ok", "bye"],
}

LABELS = {"lead_status": "Lead status", "payout_status": "Payout", "new_lead": "New lead", "product_query": "Product question",
          "lead_ideas": "Lead ideas", "issue": "Issue / ticket", "talk_to_rm": "Talk to RM", "progress": "Progress",
          "compliance": "Fair-practice coaching", "smalltalk": "Chat"}

ANGER = re.compile(r"\b(fraud|cheat|cheating|bakwas|worst|useless|complaint|consumer court|police|ombudsman|"
                   r"paisa nahi mila|dhokha|faltu|bekar|lawyer|gussa|naraz|tamasha|pareshan|angry|frustrated|fed up|"
                   r"raag|vaitag|kantala)\b", re.I)
COMPLIANCE = re.compile(r"(adjust|edit|change|badal|manage|fake|banawat).{0,25}(slip|statement|itr|cibil|document|salary)|"
                        r"(slip|statement|itr|cibil|document|salary).{0,25}(adjust|edit|change|badal|manage|fake)|"
                        r"bina bataye|without (telling|consent)|guarantee de|fee le lu", re.I)
HUMAN = re.compile(r"(sir se baat|baat karni|talk to (a )?(human|rm|rahul|priya|amit|sneha|farhan|arjun)|call me|"
                   r"bolaycha|insaan|human|manager ko)", re.I)


class Router:
    def __init__(self):
        X, y = [], []
        for k, v in EXAMPLES.items():
            X += v
            y += [k] * len(v)
        self.m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                               LogisticRegression(C=8.0, max_iter=4000))
        self.m.fit(X, y)

    def route(self, text: str) -> dict:
        p = self.m.predict_proba([text])[0]
        intent = self.m.classes_[p.argmax()]
        conf = float(p.max())
        flags = []
        if COMPLIANCE.search(text):
            intent, conf = "compliance", 1.0
        elif HUMAN.search(text):
            intent, conf = "talk_to_rm", max(conf, 0.9)
        if ANGER.search(text):
            flags.append("anger")
        return dict(intent=intent, label=LABELS[intent], confidence=round(conf, 2), flags=flags)
