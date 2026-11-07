"""SYNTHETIC demo data: RMs (with call-note transcripts), connectors, leads, payouts, tickets.

Everything here is fictional. Replace `build()` with extracts from the MyFIRST Partner
app, the LOS (loan origination system), the payout system and the RM CRM.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C

RMS = [
    dict(rm_id="RM01", name="Rahul Deshmukh", city="Nashik", langs="mr,hi,en", gender="M", sarvam_speaker="rahul",
         transcripts=[
             "Namaskar Sunita tai! Kasa chalu aahe? Aaj ek chhota update: tumcha Patil saheb cha loan sanction jhala.",
             "Tension nahi ghyaycha, documents ek ek karun baghuya. Pehle PAN, mag bank statement.",
             "Dekha, cricket madhe single-single ghetla tari match jinkta yeto. Ek lead per week, bas.",
             "Payout Friday la account madhe yeil. Kahi adchan aali tar mala direct call kara.",
             "Tumhi aaj kiti chhan kaam kela! Chala, pudhchya aathavdyacha plan karuya.",
             "Customer chi permission ghetlyashivay kuthlahi document share karu naka, he important aahe.",
         ]),
    dict(rm_id="RM02", name="Priya Iyer", city="Chennai", langs="ta,en", gender="F", sarvam_speaker="priya",
         transcripts=[
             "Vanakkam Karthik sir! Quick update for you: your client's car loan is at login stage.",
             "Don't worry, we will sort the documents step by step. First KYC, then income proof.",
             "Consistency is everything. Two good leads a month beats ten weak ones.",
             "Your payout will reach on Friday. Any issue, just WhatsApp me.",
             "Super work this week! Shall we plan next week's calls together?",
         ]),
    dict(rm_id="RM03", name="Amit Sharma", city="Jaipur", langs="hi,en", gender="M", sarvam_speaker="amit",
         transcripts=[
             "Ram Ram Mahesh ji! Sab badhiya? Aapke client ka business loan sanction ho gaya hai.",
             "Fikar mat kijiye, documents ek-ek karke dekh lete hain. Pehle PAN, phir bank statement.",
             "Bhai sahab, boond boond se sagar banta hai. Har hafte ek lead, bas itna.",
             "Payout Shukravar ko aapke khate mein aa jayega. Koi dikkat ho toh seedha phone kijiye.",
             "Aaj aapne kamaal kar diya! Chaliye agle hafte ka plan banate hain.",
             "Customer ki permission ke bina koi document share mat kijiye, yeh niyam hai.",
         ]),
    dict(rm_id="RM04", name="Sneha Kulkarni", city="Pune", langs="mr,en", gender="F", sarvam_speaker="pooja",
         transcripts=[
             "Hello Nikhil! Ek good news: tumchya client cha home loan disburse jhala.",
             "Ekdum simple aahe: KYC, salary slips, ani bank statement.",
             "Small steps, big results. Ya mahinyat 3 leads cha target theuya?",
             "Payout status app madhe disel, Friday paryant credit hoil.",
         ]),
    dict(rm_id="RM05", name="Farhan Qureshi", city="Lucknow", langs="hi,en", gender="M", sarvam_speaker="kabir",
         transcripts=[
             "Adaab Rizwan bhai! Aapke client ki personal loan file login ho gayi hai.",
             "Bilkul fikr na karein, hum saath mein documents check kar lenge.",
             "Thoda thoda roz, bahut ho jaata hai. Is hafte do clients se baat kijiye.",
             "Payout Jumme tak aa jayega, inshallah. Koi pareshani ho toh mujhe batayiye.",
         ]),
    dict(rm_id="RM06", name="Arjun Reddy", city="Hyderabad", langs="te,en", gender="M", sarvam_speaker="aditya",
         transcripts=[
             "Namaskaram Srinivas garu! Your client's LAP file is sanctioned.",
             "No tension, we will check the documents one by one.",
             "Steady game wins. One quality lead every week.",
             "Payout comes on Friday. Call me directly if anything is stuck.",
         ]),
]

ARCHETYPES = {   # archetype: (product affinity weights, typical customer, lead-idea seeds)
    "CA / tax consultant": ({"Business Loan": 4, "Professional Loan": 3, "Loan Against Property": 2, "Personal Loan": 1},
                            "business owners and professionals who file returns with them"),
    "Real-estate broker": ({"Home Loan": 5, "Loan Against Property": 3, "Personal Loan": 1},
                           "home buyers and property owners"),
    "Insurance advisor": ({"Personal Loan": 3, "Education Loan": 2, "Credit Card": 3, "Home Loan": 1},
                          "salaried families they already serve"),
    "Car dealer sales executive": ({"Used Car Loan": 5, "Two-Wheeler Loan": 3, "Personal Loan": 1},
                                   "walk-in car and bike buyers"),
    "Shop owner / trader": ({"Business Loan": 3, "Personal Loan": 2, "Two-Wheeler Loan": 2, "Credit Card": 1},
                            "fellow traders and regular customers"),
    "Coaching institute owner": ({"Education Loan": 5, "Personal Loan": 2},
                                 "students and parents planning admissions"),
    "Clinic / pharmacy owner": ({"Professional Loan": 4, "Personal Loan": 2, "Business Loan": 2},
                                "doctors, chemists and clinic staff"),
}

FIRST = ["Sunita", "Mahesh", "Karthik", "Nikhil", "Rizwan", "Srinivas", "Pooja", "Vikram", "Anita", "Imran", "Lakshmi",
         "Ganesh", "Meera", "Rohit", "Fatima", "Suresh", "Kavita", "Arvind", "Deepa", "Harish", "Nandini", "Sachin",
         "Shabana", "Venkat", "Jyoti", "Prakash", "Rekha", "Manoj", "Sameer", "Usha"]
LAST = ["Patil", "Joshi", "Kumar", "Gupta", "Khan", "Rao", "Shinde", "Mehta", "Naidu", "Ansari", "Pawar", "Iyer",
        "Verma", "Kale", "Shaikh", "Reddy", "Jain", "More"]
STAGES = ["Submitted", "Documents pending", "Login", "Sanctioned", "Disbursed", "Rejected"]
AMT = {"Personal Loan": (2e5, 12e5), "Business Loan": (5e5, 40e5), "Professional Loan": (5e5, 30e5), "Home Loan": (20e5, 90e5),
       "Loan Against Property": (15e5, 80e5), "Used Car Loan": (3e5, 12e5), "Two-Wheeler Loan": (6e4, 2e5),
       "Education Loan": (3e5, 25e5), "Credit Card": (0, 0)}


def _lang_for(rm, rng):
    langs = rm["langs"].split(",")
    p = np.array([0.6, 0.25, 0.15][: len(langs)])
    return str(rng.choice(langs, p=p / p.sum()))


def build(seed: int = C.SEED, n_connectors: int = 240) -> dict:
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(C.TODAY)
    rms = pd.DataFrame([{k: v for k, v in r.items() if k != "transcripts"} for r in RMS])
    con_rows, lead_rows, pay_rows, tix_rows = [], [], [], []
    archs = list(ARCHETYPES)
    lid = 1
    for i in range(n_connectors):
        rm = RMS[i % len(RMS)]
        arch = archs[rng.integers(0, len(archs))]
        name = f"{FIRST[rng.integers(0, len(FIRST))]} {LAST[rng.integers(0, len(LAST))]}"
        lang = _lang_for(rm, rng)
        maturity = rng.choice(["Low", "Medium", "High"], p=[0.4, 0.4, 0.2])
        joined = today - pd.Timedelta(days=int(rng.integers(2, 25) if rng.random() < 0.08 else rng.integers(25, 720)))
        engage = rng.beta(2, 3) if rng.random() > 0.1 else rng.uniform(0.75, 1.0)   # latent activity; 10% stars
        if (today - joined).days < 15:
            engage *= 0.3
        n_leads = rng.poisson(1 + 22 * engage * min(1, (today - joined).days / 180))
        affinity = ARCHETYPES[arch][0]
        prods, w = list(affinity), np.array(list(affinity.values()), float)
        cid = f"CN{i + 1:04d}"
        last_lead = None
        for _ in range(n_leads):
            p = prods[rng.choice(len(prods), p=w / w.sum())]
            age = int(rng.integers(0, max(1, min(240, (today - joined).days))))
            d = today - pd.Timedelta(days=age)
            lo, hi = AMT[p]
            amt = 0 if p == "Credit Card" else round(rng.uniform(lo, hi) / 10000) * 10000
            if age > 45:
                stage = rng.choice(["Disbursed", "Rejected", "Sanctioned"], p=[0.62, 0.3, 0.08])
            elif age > 15:
                stage = rng.choice(STAGES, p=[0.05, 0.15, 0.2, 0.2, 0.3, 0.1])
            else:
                stage = rng.choice(STAGES, p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05])
            cust = f"{FIRST[rng.integers(0, len(FIRST))]} {LAST[rng.integers(0, len(LAST))][0]}."
            lead_rows.append(dict(lead_id=f"LD{lid:05d}", connector_id=cid, product=p, customer=cust, amount=amt,
                                  submitted=d.date().isoformat(), stage=stage,
                                  pending_item=("Last 3 months bank statement" if stage == "Documents pending" else ""),
                                  days_in_stage=int(rng.integers(0, 12))))
            if stage == "Disbursed":
                rate = C.PAYOUT_RATES.get(p)
                payout = C.CARD_FLAT_PAYOUT if p == "Credit Card" else round(amt * rate / 100)
                pstat = rng.choice(["Paid", "Processing", "On hold"], p=[0.75, 0.17, 0.08]) if age < 30 else "Paid"
                reason = ""
                if pstat == "On hold":
                    reason = rng.choice(["GST invoice not uploaded", "Bank account name mismatch", "Customer cancelled within 15 days"])
                paydate = (d + pd.Timedelta(days=int(rng.integers(10, 20))))
                nxt_fri = today + pd.Timedelta(days=(4 - today.weekday()) % 7 or 7)
                pay_rows.append(dict(lead_id=f"LD{lid:05d}", connector_id=cid, product=p, amount=payout, status=pstat,
                                     date=(paydate if pstat == "Paid" else nxt_fri).date().isoformat(), hold_reason=reason))
            if last_lead is None or d > last_lead:
                last_lead = d
            lid += 1
        if rng.random() < 0.12:
            tix_rows.append(dict(ticket_id=f"TK{len(tix_rows) + 1:04d}", connector_id=cid,
                                 category=rng.choice(["Payout delay", "Lead stuck", "App login", "Wrong payout amount"]),
                                 opened=(today - pd.Timedelta(days=int(rng.integers(1, 20)))).date().isoformat(), status="Open"))
        con_rows.append(dict(
            connector_id=cid, name=name, archetype=arch, rm_id=rm["rm_id"], rm_name=rm["name"], city=rm["city"],
            language=lang, digital_maturity=maturity, joined=joined.date().isoformat(),
            last_lead=last_lead.date().isoformat() if last_lead is not None else "",
            app_opens_14d=int(rng.poisson(1 + 10 * engage)), rm_touches_30d=int(rng.poisson(0.5 + 2 * engage)),
            preferred_hour=int(rng.choice([10, 11, 12, 13, 16, 17, 18, 19])), consent_whatsapp=bool(rng.random() < 0.9),
            latent_engagement=round(float(engage), 3),
        ))
    con = pd.DataFrame(con_rows)
    leads = pd.DataFrame(lead_rows)
    pays = pd.DataFrame(pay_rows)
    tix = pd.DataFrame(tix_rows, columns=["ticket_id", "connector_id", "category", "opened", "status"])
    # demo connectors: a new one, a slipping one, a star (fixed so the demo is repeatable)
    out = dict(rms=rms, connectors=con, leads=leads, payouts=pays, tickets=tix,
               transcripts={r["rm_id"]: r["transcripts"] for r in RMS})
    for k in ("rms", "connectors", "leads", "payouts", "tickets"):
        out[k].to_csv(C.DATA / f"{k}.csv", index=False)
    (C.DATA / "rm_transcripts.json").write_text(json.dumps(out["transcripts"], ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load() -> dict:
    if not (C.DATA / "connectors.csv").exists():
        return build()
    out = {k: pd.read_csv(C.DATA / f"{k}.csv", keep_default_na=False) for k in ("rms", "connectors", "leads", "payouts", "tickets")}
    out["transcripts"] = json.loads((C.DATA / "rm_transcripts.json").read_text(encoding="utf-8"))
    return out
