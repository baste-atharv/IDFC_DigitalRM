"""Behavioural nudge engine + connector stage + dormancy risk.

Each nudge names the behavioural principle it uses, respects a weekly cap, quiet hours and
WhatsApp consent, and is written in the connector's language. Dormancy risk is a small
logistic model on engagement features (SYNTHETIC training labels); in production it is
trained on real 'no lead in the next 30 days' outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as C

TODAY = pd.Timestamp(C.TODAY)


def connector_table(d: dict) -> pd.DataFrame:
    con = d["connectors"].copy()
    leads = d["leads"].copy()
    leads["submitted"] = pd.to_datetime(leads["submitted"])
    pays = d["payouts"].copy()
    g = leads.groupby("connector_id")
    con["leads_30d"] = con.connector_id.map(leads[leads.submitted > TODAY - pd.Timedelta(days=30)].groupby("connector_id").size()).fillna(0).astype(int)
    con["leads_90d"] = con.connector_id.map(leads[leads.submitted > TODAY - pd.Timedelta(days=90)].groupby("connector_id").size()).fillna(0).astype(int)
    con["leads_total"] = con.connector_id.map(g.size()).fillna(0).astype(int)
    disb90 = leads[(leads.stage == "Disbursed") & (leads.submitted > TODAY - pd.Timedelta(days=90))].groupby("connector_id").size()
    con["disbursed_90d"] = con.connector_id.map(disb90).fillna(0).astype(int)
    con["docs_pending"] = con.connector_id.map(leads[leads.stage == "Documents pending"].groupby("connector_id").size()).fillna(0).astype(int)
    con["payout_on_hold"] = con.connector_id.map(pays[pays.status == "On hold"].groupby("connector_id")["amount"].sum()).fillna(0)
    con["payout_processing"] = con.connector_id.map(pays[pays.status == "Processing"].groupby("connector_id")["amount"].sum()).fillna(0)
    pdt = pd.to_datetime(pays.date)
    paid90 = pays[(pays.status == "Paid") & (pdt > TODAY - pd.Timedelta(days=90))]
    con["earned_90d"] = con.connector_id.map(paid90.groupby("connector_id")["amount"].sum()).fillna(0)
    con["days_since_lead"] = con.last_lead.map(lambda x: (TODAY - pd.Timestamp(x)).days if x else 999)
    con["tenure_days"] = con.joined.map(lambda x: (TODAY - pd.Timestamp(x)).days)
    con["stage"] = con.apply(stage, axis=1)
    con["tier"], con["multiplier"], con["next_tier_gap"] = zip(*con.disbursed_90d.map(tier))
    con["streak_weeks"] = con.connector_id.map(lambda c: streak(leads[leads.connector_id == c].submitted))
    con["dormancy_risk"] = dormancy_model(con)
    return con


def stage(r) -> str:
    if r.leads_total == 0:
        return "New - not activated" if r.tenure_days <= 30 else "Signed up, never active"
    if r.days_since_lead <= 14:
        return "Habitual" if r.leads_90d >= 3 else "Activated"
    if r.days_since_lead <= 45:
        return "Slipping"
    return "Dormant"


def tier(n):
    name, mult, gap = "Bronze", 1.0, 3 - n
    for thr, nm, m in C.TIERS:
        if n >= thr:
            name, mult = nm, m
    nxt = [thr for thr, _, _ in C.TIERS if thr > n]
    gap = (nxt[0] - n) if nxt else 0
    return name, mult, gap


def streak(dates: pd.Series) -> int:
    if dates.empty:
        return 0
    weeks = set(((TODAY - dates).dt.days // 7).tolist())
    # the current week is still open: a streak can run up to last week
    start = 0 if 0 in weeks else 1
    k = start
    while k in weeks:
        k += 1
    return k - start


def dormancy_model(con: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.minimum(con.days_since_lead, 200) / 30, con.leads_90d, con.app_opens_14d, con.rm_touches_30d,
                         (con.payout_on_hold > 0).astype(int)])
    # SYNTHETIC label: probability of 'no lead next 30 days' from the latent engagement used to make the data
    rng = np.random.default_rng(C.SEED)
    y = (rng.random(len(con)) > con.latent_engagement.values * 1.3).astype(int)
    m = LogisticRegression(max_iter=1000).fit(X, y)
    return m.predict_proba(X)[:, 1].round(3)


TEXT = {
    "en": {
        "new": "Hi {name}! You're 2 steps into MyFIRST: profile done, bank account done. One lead and your first payout is on its way. Shall I help you add one now?",
        "activated": "{name}, great start! Which 2 people will you speak to this week? Tell me their names and I'll remind you on {day}.",
        "habitual": "{name}, {streak}-week streak! {peer}. Keep it going this week?",
        "slipping_docs": "{name}, Rs {amount} is waiting: {n} of your files need one document. Upload today and the payout moves this Friday.",
        "slipping": "{name}, you're {gap} disbursal(s) from {next_tier} tier (higher payout on every loan). Here's an idea for this week: {idea}",
        "dormant": "{name}, {rm} would like to catch up with you personally this week. Good time for a call?",
    },
    "hi": {
        "new": "Namaste {name} ji! MyFIRST mein aapke 2 step ho gaye: profile aur bank account. Bas ek lead, aur pehla payout raste mein. Abhi ek lead add karein?",
        "activated": "{name} ji, badhiya shuruaat! Is hafte kin 2 logon se baat karenge? Naam bataiye, main {day} ko yaad dila dunga.",
        "habitual": "{name} ji, {streak} hafte ki streak! {peer}. Is hafte bhi jaari rakhein?",
        "slipping_docs": "{name} ji, Rs {amount} ka payout aapka intezaar kar raha hai: {n} file mein ek document baaki hai. Aaj upload kijiye, payout isi Shukravar.",
        "slipping": "{name} ji, {next_tier} tier se bas {gap} disbursal door (har loan pe zyada payout). Is hafte ka idea: {idea}",
        "dormant": "{name} ji, {rm} is hafte aapse khud baat karna chahte hain. Kab call karein?",
    },
    "mr": {
        "new": "Namaskar {name}! MyFIRST madhe tumche 2 step jhale: profile ani bank account. Fakt ek lead, ani pahila payout yeil. Ata ek lead add karuya?",
        "activated": "{name}, chhan suruvat! Ya aathavdyat konashi 2 lokanshi bolnar? Nave sanga, mi {day} la aathvan karun den.",
        "habitual": "{name}, {streak} aathavdyanchi streak! {peer}. Ya aathavdyat pan chalu thevuya?",
        "slipping_docs": "{name}, Rs {amount} payout tumchi vaat baghtay: {n} file madhe ek document baki aahe. Aaj upload kara, payout ya Shukravari.",
        "slipping": "{name}, {next_tier} tier pasun fakt {gap} disbursal laamb (pratyek loan var jasta payout). Ya aathavdyachi kalpana: {idea}",
        "dormant": "{name}, {rm} ya aathavdyat tumchyashi swatah bolu ichhitat. Call sathi kadhi vel aahe?",
    },
}
PRINCIPLE = {"new": "Endowed progress + fresh start", "activated": "Implementation intention", "habitual": "Social proof + streak",
             "slipping_docs": "Loss aversion (money already earned)", "slipping": "Goal gradient (next tier)",
             "dormant": "Human touch: RM call, not another message"}

IDEAS = {
    "CA / tax consultant": "ask 3 clients who filed GST this month if they plan to expand",
    "Real-estate broker": "ask this week's site-visit buyers how they plan to finance",
    "Insurance advisor": "on renewal calls, ask about wedding or education costs next year",
    "Car dealer sales executive": "offer finance at every test drive this weekend",
    "Shop owner / trader": "ask 2 fellow traders about stock for the wedding season",
    "Coaching institute owner": "share an education-loan note with parents of final-year students",
    "Clinic / pharmacy owner": "ask doctors in your network about equipment upgrades",
}


def next_nudge(r, sent_this_week: int = 0, hour: int = 11) -> dict:
    lang = r.language if r.language in TEXT else "en"
    first = r["name"].split()[0] if isinstance(r, pd.Series) else r.name.split()[0]
    s = r.stage
    if s.startswith("New") or s.startswith("Signed"):
        key = "new"
    elif s == "Activated":
        key = "activated"
    elif s == "Habitual":
        key = "habitual" if r.streak_weeks >= 2 else "activated"
    elif s == "Slipping":
        key = "slipping_docs" if r.docs_pending > 0 and r.payout_processing + r.payout_on_hold > 0 else "slipping"
    else:
        key = "dormant"
    amount = int(r.payout_processing + r.payout_on_hold) or 4500
    peer = {"en": f"Top connectors in {r.city} averaged 4 leads last month",
            "hi": f"{r.city} ke top connectors ne pichhle mahine average 4 lead diye",
            "mr": f"{r.city} madhil top connectors ni gelya mahinyat sarasari 4 lead dile"}[lang]
    msg = TEXT[lang][key].format(name=first, streak=r.streak_weeks, peer=peer, amount=f"{amount:,}", n=max(1, r.docs_pending),
                                 gap=max(1, r.next_tier_gap), next_tier=_next_tier_name(r.tier), idea=IDEAS[r.archetype],
                                 rm=r.rm_name.split()[0], day="Thursday")
    blocked = []
    if sent_this_week >= C.NUDGE_CAP_PER_WEEK:
        blocked.append("weekly cap reached")
    if not r.consent_whatsapp:
        blocked.append("no WhatsApp consent: in-app only")
    q0, q1 = C.QUIET_HOURS
    send_hour = r.preferred_hour if not (hour >= q0 or hour < q1) else max(q1, r.preferred_hour)
    channel = "RM call task" if key == "dormant" else ("WhatsApp voice note" if r.digital_maturity == "Low" else "In-app + WhatsApp text")
    return dict(connector_id=r.connector_id, stage=s, nudge=key, principle=PRINCIPLE[key], channel=channel,
                send_at=f"{send_hour:02d}:00", message=msg, blocked="; ".join(blocked), language=lang)


def _next_tier_name(cur):
    names = [n for _, n, _ in C.TIERS]
    i = names.index(cur)
    return names[min(i + 1, len(names) - 1)]
