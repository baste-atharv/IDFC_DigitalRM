"""Tools the Digital RM can call. Each returns plain JSON-able data, so any LLM (or the offline
agent) can use it. Tool schemas follow JSON Schema and are converted per provider in llm.py."""
from __future__ import annotations

import pandas as pd

from . import config as C, recommend, store
from .nudges import IDEAS


class Toolbox:
    def __init__(self, data: dict, contable: pd.DataFrame, kb, connector_id: str, llm=None):
        self.d, self.ct, self.kb, self.cid, self.llm = data, contable, kb, connector_id, llm
        self.con = contable.set_index("connector_id").loc[connector_id]
        self.events = []          # side effects shown in the UI (handoffs, leads, tickets)

    # ------------------------------------------------------------------ tools
    def get_my_leads(self, status: str = "") -> dict:
        l = self.d["leads"]
        l = l[l.connector_id == self.cid].sort_values("submitted", ascending=False)
        if status:
            l = l[l.stage.str.lower().str.contains(status.lower())]
        rows = l.head(8)[["lead_id", "customer", "product", "amount", "stage", "pending_item", "submitted"]].to_dict("records")
        return dict(count=int(len(l)), leads=rows,
                    open_by_stage=l[~l.stage.isin(["Disbursed", "Rejected"])].stage.value_counts().to_dict())

    def get_my_payouts(self) -> dict:
        p = self.d["payouts"]
        p = p[p.connector_id == self.cid].sort_values("date", ascending=False)
        return dict(paid_last_90d=float(self.con.earned_90d), processing=float(self.con.payout_processing),
                    on_hold=float(self.con.payout_on_hold), tier=self.con.tier, multiplier=float(self.con.multiplier),
                    recent=p.head(6)[["lead_id", "product", "amount", "status", "date", "hold_reason"]].to_dict("records"),
                    rule="Payouts are credited every Friday for disbursals cleared by Monday (demo rule).")

    def search_products(self, query: str) -> dict:
        hits = self.kb.search(query, k=3)
        return dict(results=[dict(source=c.cite, text=c.text[:700]) for c, _ in hits])

    def match_products(self, description: str = "", occupation: str = "", purpose: str = "", amount_needed: float = 0,
                       monthly_income: float = 0, existing_emi: float = 0, owns_property: bool = False) -> dict:
        prof = recommend.extract_offline(description) if description else {}
        prof.update({k: v for k, v in dict(occupation=occupation, purpose=purpose, amount_needed=amount_needed,
                                           monthly_income=monthly_income, existing_emi=existing_emi,
                                           owns_property=owns_property).items() if v})
        return dict(profile=prof, matches=recommend.match(prof),
                    note="Indicative only. The bank's credit assessment decides eligibility, amount and rate.")

    def submit_lead(self, customer_name: str, product: str, amount: float = 0, phone_last4: str = "",
                    notes: str = "", consent_confirmed: bool = False) -> dict:
        if not consent_confirmed:
            return dict(ok=False, error="Customer consent not confirmed. Ask the connector to confirm the customer agreed to share details.")
        rec = store.append("leads_submitted", connector_id=self.cid, customer=customer_name, product=product,
                           amount=amount, phone_last4=phone_last4, notes=notes, consent=True)
        store.append("consents", connector_id=self.cid, customer=customer_name, purpose=f"{product} application",
                     channel="Digital RM chat")
        self.events.append(("lead", rec))
        res = dict(ok=True, lead_ref=f"NEW-{rec['hash'][:6].upper()}", next_step="Upload PAN and last 3 months bank statement in the app.")
        if amount and amount >= C.HANDOFF_AMOUNT:
            res["handoff"] = self.handoff_to_rm(f"High-value lead Rs {amount:,.0f} ({product}) for {customer_name}", "high")
        return res

    def raise_ticket(self, category: str, description: str) -> dict:
        rec = store.append("tickets", connector_id=self.cid, category=category, description=description, status="Open")
        self.events.append(("ticket", rec))
        return dict(ok=True, ticket=f"TK-{rec['hash'][:6].upper()}", sla="RM notified; escalates if open after 7 days")

    def get_lead_ideas(self) -> dict:
        hits = self.kb.search(self.con.archetype + " lead ideas", k=1)
        return dict(archetype=self.con.archetype, idea_this_week=IDEAS[self.con.archetype],
                    playbook=hits[0][0].text if hits else "", season="Festive and wedding season (Oct-Dec)")

    def get_my_progress(self) -> dict:
        c = self.con
        return dict(tier=c.tier, disbursed_90d=int(c.disbursed_90d), next_tier_gap=int(c.next_tier_gap),
                    streak_weeks=int(c.streak_weeks), leads_30d=int(c.leads_30d), earned_90d=float(c.earned_90d))

    def handoff_to_rm(self, reason: str, urgency: str = "normal") -> dict:
        if any(e[0] == "handoff" for e in self.events):           # one handoff per turn is enough
            return dict(ok=True, message=f"{self.con.rm_name} is already notified", urgency=urgency)
        rec = store.append("handoffs", connector_id=self.cid, connector=self.con["name"], rm=self.con.rm_name,
                           reason=reason, urgency=urgency, status="Open")
        self.events.append(("handoff", rec))
        return dict(ok=True, message=f"{self.con.rm_name} has been notified and will call back", urgency=urgency)

    def flag_compliance(self, issue: str) -> dict:
        rec = store.append("compliance_flags", connector_id=self.cid, issue=issue)
        self.events.append(("compliance", rec))
        return dict(ok=True, logged=True, coaching="Explain the fair-practice rule kindly; do not help with the request.")

    # ------------------------------------------------------------------ registry
    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name not in SCHEMAS:
            return dict(error=f"unknown tool {name}")
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return dict(error=f"bad arguments: {e}")


S = lambda **p: dict(type="object", properties=p, required=[])   # noqa: E731
SCHEMAS = {
    "get_my_leads": ("List this connector's leads and their stage. Optional filter by stage name.",
                     S(status=dict(type="string", description="e.g. 'Documents pending', 'Sanctioned'"))),
    "get_my_payouts": ("Payout summary: paid, processing, on hold (with reasons), tier multiplier.", S()),
    "search_products": ("Search the product, payout-policy and fair-practice library.",
                        {**S(query=dict(type="string")), "required": ["query"]}),
    "match_products": ("Indicative product match and eligibility for a customer the connector describes.",
                       S(description=dict(type="string", description="the connector's own words about the customer"),
                         occupation=dict(type="string"), purpose=dict(type="string"), amount_needed=dict(type="number"),
                         monthly_income=dict(type="number"), existing_emi=dict(type="number"), owns_property=dict(type="boolean"))),
    "submit_lead": ("Submit a new lead. Only after the connector confirms the customer agreed to share details.",
                    {**S(customer_name=dict(type="string"), product=dict(type="string"), amount=dict(type="number"),
                         phone_last4=dict(type="string"), notes=dict(type="string"), consent_confirmed=dict(type="boolean")),
                     "required": ["customer_name", "product", "consent_confirmed"]}),
    "raise_ticket": ("Raise a support ticket (payout delay, lead stuck, wrong payout, app problem).",
                     {**S(category=dict(type="string"), description=dict(type="string")), "required": ["category", "description"]}),
    "get_lead_ideas": ("This week's lead idea for this connector's type and season.", S()),
    "get_my_progress": ("Tier, streak, gap to next tier, recent activity.", S()),
    "handoff_to_rm": ("Bring the human RM in (raise hand): disputes, anger, request for RM, loans above Rs 50 lakh.",
                      {**S(reason=dict(type="string"), urgency=dict(type="string", enum=["normal", "high"])), "required": ["reason"]}),
    "flag_compliance": ("Log a request that breaks fair-practice rules (altering documents, no consent, promising approval).",
                        {**S(issue=dict(type="string")), "required": ["issue"]}),
}
