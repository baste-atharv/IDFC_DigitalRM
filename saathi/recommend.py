"""Right product for the right customer: profile extraction + indicative product match.

All numbers are INDICATIVE (illustrative FOIR and rates). The bank's credit decision is final.
"""
from __future__ import annotations

import json
import re

RATES = {"Personal Loan": 13.0, "Business Loan": 16.0, "Professional Loan": 10.5, "Home Loan": 8.9,
         "Loan Against Property": 10.0, "Used Car Loan": 12.5, "Two-Wheeler Loan": 13.5, "Education Loan": 11.0}  # ILLUSTRATIVE
TENURE_M = {"Personal Loan": 60, "Business Loan": 60, "Professional Loan": 60, "Home Loan": 240, "Loan Against Property": 180,
            "Used Car Loan": 60, "Two-Wheeler Loan": 36, "Education Loan": 120}
LIMITS = {"Personal Loan": (2e4, 40e5), "Business Loan": (5e5, 1e7), "Professional Loan": (5e5, 75e5),
          "Home Loan": (5e5, 5e7), "Loan Against Property": (10e5, 5e7), "Used Car Loan": (1e5, 30e5),
          "Two-Wheeler Loan": (2e4, 3e5), "Education Loan": (1e5, 1e7)}
DOCS = {
    "Personal Loan": ["PAN", "Aadhaar or address proof", "Last 3 months salary slips or bank statement"],
    "Business Loan": ["PAN", "GST registration", "Last 2 years ITR", "12 months business bank statement"],
    "Professional Loan": ["Degree / registration certificate", "PAN", "Last 2 years ITR", "6 months bank statement"],
    "Home Loan": ["KYC", "Income proof", "Property papers / sale agreement", "6 months bank statement"],
    "Loan Against Property": ["KYC", "Income proof", "Property title documents", "12 months bank statement"],
    "Used Car Loan": ["KYC", "Income proof", "Car RC and insurance (for valuation)"],
    "Two-Wheeler Loan": ["KYC", "Income proof or bank statement"],
    "Education Loan": ["Student and parent KYC", "Admission letter", "Fee structure", "Co-applicant income proof"],
    "Credit Card": ["PAN", "Address proof", "Income proof"],
}
PROFESSIONS = ("doctor", "ca ", "chartered accountant", "architect", "company secretary", "cs ", "dentist", "physician")

WORDNUM = {"lakh": 1e5, "lac": 1e5, "lakhs": 1e5, "l": 1e5, "crore": 1e7, "cr": 1e7, "k": 1e3, "thousand": 1e3, "hazaar": 1e3, "hazar": 1e3}


def parse_amount(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lakhs?|lac|crore|cr|k|thousand|hazaa?r|l)\b", text, re.I)
    if m:
        return float(m.group(1)) * WORDNUM[m.group(2).lower()]
    m = re.search(r"(?:rs\.?|₹|inr)\s*([\d,]{4,})", text, re.I)
    return float(m.group(1).replace(",", "")) if m else None


def extract_offline(text: str) -> dict:
    t = " " + text.lower() + " "
    occ = ("doctor" if re.search(r"doctor|dr\.|clinic|dentist", t) else
           "chartered accountant" if re.search(r"\bca\b|chartered", t) else
           "business owner" if re.search(r"business|shop|dukan|trader|vyapar|vyapari|factory|gst", t) else
           "salaried" if re.search(r"salar|job|naukri|employee|company mein|kaam karta", t) else "")
    purpose = ("home purchase" if re.search(r"\bghar\b|home|house|flat|makaan", t) else
               "car" if re.search(r"\bcar\b|gaadi|gadi", t) else
               "two-wheeler" if re.search(r"bike|scooter|scooty|two.?wheeler", t) else
               "education" if re.search(r"educat|admission|college|study|padhai|abroad", t) else
               "clinic setup" if re.search(r"clinic|hospital|equipment", t) else
               "working capital" if re.search(r"working capital|stock|expan|business", t) else
               "wedding" if re.search(r"wedding|shaadi|lagna|marriage", t) else "")
    name = re.search(r"(?:client|customer|friend|mitra|named?|naam)\s+(?:hai\s+|is\s+)?(?:Dr\.?\s+|Mr\.?\s+|Mrs\.?\s+|Ms\.?\s+)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", text)
    inc = re.search(r"(?:income|salary|kamata|kamai|earns?)\D{0,15}(\d+(?:\.\d+)?)\s*(lakhs?|lac|k|thousand|hazaa?r)?", t)
    income = None
    if inc:
        income = float(inc.group(1)) * (WORDNUM.get((inc.group(2) or "").lower(), 1))
        if income < 1000:
            income = None
    return dict(customer_name=name.group(1) if name else "", occupation=occ, purpose=purpose,
                amount_needed=parse_amount(text), monthly_income=income,
                owns_property=bool(re.search(r"owns? (a )?(property|house|shop)|property hai|apna ghar", t)),
                source="offline parser")


def extract_with_llm(text: str, llm) -> dict:
    prompt = ("Extract the loan customer's details from this connector message. Return JSON with keys: customer_name, "
              "occupation, purpose, amount_needed (rupees, number), monthly_income (rupees, number or null), "
              "existing_emi (number or null), owns_property (bool), business_vintage_years (number or null), city. "
              "Use null when not stated. Message:\n" + text)
    try:
        out = llm.complete(prompt, system="You extract structured data. Reply with JSON only.", max_tokens=300)
        j = json.loads(out[out.find("{"): out.rfind("}") + 1])
        j["source"] = "LLM extraction"
        return j
    except Exception:
        return extract_offline(text)


def emi_capacity_to_amount(emi: float, rate_pa: float, months: int) -> float:
    r = rate_pa / 1200
    return emi * (1 - (1 + r) ** -months) / r


def match(profile: dict) -> list[dict]:
    occ = (profile.get("occupation") or "").lower()
    pur = (profile.get("purpose") or "").lower()
    amt = profile.get("amount_needed") or 0
    inc = profile.get("monthly_income")
    emi_now = profile.get("existing_emi") or 0
    cands = []
    if "home" in pur or "house" in pur:
        cands.append(("Home Loan", "buying a home"))
    if "car" in pur:
        cands.append(("Used Car Loan", "car purchase"))
    if "two" in pur or "bike" in pur:
        cands.append(("Two-Wheeler Loan", "two-wheeler purchase"))
    if "educat" in pur:
        cands.append(("Education Loan", "studies"))
    if any(p.strip() in " " + occ + " " for p in PROFESSIONS) or "clinic" in pur:
        cands.append(("Professional Loan", "qualified professional; market rates ~10-11% [SOURCED]"))
    if "business" in occ or "working capital" in pur or "trader" in occ:
        cands.append(("Business Loan", "unsecured Rs 5 lakh-1 crore, 1-7 years [SOURCED]"))
    if profile.get("owns_property") and amt and amt > 20e5:
        cands.append(("Loan Against Property", "large need, owns property: lower rate than unsecured"))
    if not cands or "wedding" in pur or ("salaried" in occ and not cands):
        cands.append(("Personal Loan", "general purpose; Rs 20,000-40 lakh [SOURCED, 2021 blog]"))
    if "salaried" in occ:
        cands.append(("Credit Card", "salaried with income proof"))
    out = []
    for prod, why in dict(cands).items():
        row = dict(product=prod, why=why, documents=DOCS[prod], indicative_max=None, indicative_emi=None)
        if prod in RATES and inc:
            foir = 0.5 if inc < 1e5 else 0.6                       # ILLUSTRATIVE FOIR
            cap = max(0.0, inc * foir - emi_now)
            mx = emi_capacity_to_amount(cap, RATES[prod], TENURE_M[prod])
            lo, hi = LIMITS[prod]
            row["indicative_max"] = round(min(mx, hi), -4) if mx >= lo else 0
            if amt:
                r = RATES[prod] / 1200
                n = TENURE_M[prod]
                row["indicative_emi"] = round(amt * r / (1 - (1 + r) ** -n))
        out.append(row)
    return out
