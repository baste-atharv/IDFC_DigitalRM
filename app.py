"""FIRST Saathi - the Connector's Digital RM (FAME 6.0, PS4 prototype).

Run: double-click RUN_APP.bat (or: streamlit run app.py). Add your API keys on the Settings page.
All connector, lead and payout data is SYNTHETIC.
"""
from __future__ import annotations

import inspect
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from saathi import config as C
from saathi import data as D
from saathi import evaluate, keys, nudges, persona, raft, recommend, store, voice
from saathi.agent import DigitalRM
from saathi.kb import KB
from saathi.llm import LLM, LLMError, test_connection
from saathi.tools import Toolbox

st.set_page_config(page_title="FIRST Saathi | Connector's Digital RM", layout="wide")
MAROON = "#9C1D26"
TTS_OPTIONS = ["Off", "edge-tts (free, no key)", "ElevenLabs (RM's cloned voice)", "Sarvam (Indian voices)"]
STT_OPTIONS = ["Off", "Sarvam (Indian languages)", "OpenAI transcription"]


def table(df, **kw):
    if "width" in inspect.signature(st.dataframe).parameters:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    else:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def show(fig, **kw):
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), **kw)
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_resource(show_spinner="Loading the Digital RM ...")
def base():
    d = D.load()
    return d, KB()


d, kb = base()
ss = st.session_state
if "apikeys" not in ss:
    ss.apikeys = keys.load()
    ss.apikeys.setdefault("LLM_PROVIDER", "Offline (no LLM)")
    ss.apikeys.setdefault("TTS_PROVIDER", "edge-tts (free, no key)")
    ss.apikeys.setdefault("STT_PROVIDER", "Off")
ss.setdefault("chats", {})
ss.setdefault("cards", {})
ss.setdefault("voice_killed", False)
ss.setdefault("last_audio_id", None)
K = ss.apikeys


def contable():
    if "ct" not in ss:
        ss.ct = nudges.connector_table(d)
    return ss.ct


ct = contable()
for r in d["rms"].to_dict("records"):
    ss.cards.setdefault(r["rm_id"], persona.style_card(r, d["transcripts"][r["rm_id"]]))


def get_llm():
    prov = K.get("LLM_PROVIDER", "Offline (no LLM)")
    if C.PROVIDERS[prov]["kind"] == "offline":
        return None, "Offline (no LLM key): rule-based replies"
    sig = (prov, K.get("LLM_MODEL"), K.get("LLM_BASE_URL"), K.get("LLM_API_KEY"))
    if ss.get("llm_sig") != sig:
        try:
            ss.llm = LLM(prov, K.get("LLM_API_KEY", ""), K.get("LLM_MODEL", ""), K.get("LLM_BASE_URL", ""))
            ss.llm_err = ""
        except LLMError as e:
            ss.llm, ss.llm_err = None, str(e)
        ss.llm_sig = sig
    if ss.llm is None:
        return None, f"Offline: {ss.llm_err}"
    return ss.llm, f"{ss.llm.provider} · {ss.llm.model}"


llm, mode_txt = get_llm()
agent = DigitalRM(d, ct, kb, ss.cards, llm)


def rm_voice_keys(rm_id):
    k = dict(K)
    vid = K.get(f"ELEVENLABS_VOICE_ID_{rm_id}") or K.get("ELEVENLABS_VOICE_ID", "")
    k["ELEVENLABS_VOICE_ID"] = vid
    return k


def speak(text, lang, rm_row):
    prov = K.get("TTS_PROVIDER", "Off")
    if prov == "Off":
        return None, None
    if ss.voice_killed and prov.startswith("ElevenLabs"):
        prov = "edge-tts (free, no key)"
    try:
        audio, mime = voice.tts(text, lang, prov, rm_voice_keys(rm_row["rm_id"]), gender=rm_row["gender"],
                                sarvam_speaker=rm_row.get("sarvam_speaker", ""))
        return audio, mime
    except voice.VoiceError as e:
        st.caption(f"Voice not generated: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
_pref = [ct[(ct.rm_id == "RM01") & (ct.docs_pending > 0) & (ct.payout_on_hold > 0)],
         ct[(ct.rm_id == "RM01") & (ct.docs_pending > 0)], ct[ct.rm_id == "RM01"]]
default_cid = next(p for p in _pref if len(p)).connector_id.iloc[0]


def inr(x):
    return f"Rs {x / 1e5:.1f} L" if x >= 1e5 else f"Rs {x:,.0f}"

with st.sidebar:
    st.markdown(f"<h2 style='color:{MAROON};margin-bottom:0'>FIRST Saathi</h2><div>the Connector's Digital RM</div>", unsafe_allow_html=True)
    page = st.radio("Page", ["1. Connector app", "2. Add a lead", "3. RM cockpit", "4. Persona studio (clone the RM)",
                             "5. Nudge engine", "6. Quality & governance", "7. Settings (API keys)", "8. How it is built"])
    opts = ct.sort_values(["rm_id", "connector_id"]).connector_id.tolist()
    lab = ct.set_index("connector_id")
    cid = st.selectbox("Connector (demo login)", opts, index=opts.index(default_cid),
                       format_func=lambda c: f"{c} · {lab.loc[c, 'name']} · {lab.loc[c, 'archetype'].split(' /')[0]} · {lab.loc[c, 'language']} · RM {lab.loc[c, 'rm_name'].split()[0]}")
    con = lab.loc[cid]
    lang_choice = st.selectbox("Reply language", ["auto (connector's)"] + list(C.LANGS),
                               format_func=lambda x: x if x.startswith("auto") else C.LANGS[x][0])
    lang = con.language if lang_choice.startswith("auto") else lang_choice
    st.caption(f"Brain: **{mode_txt}**")
    st.caption(f"Voice out: **{K.get('TTS_PROVIDER')}** · Voice in: **{K.get('STT_PROVIDER')}**")
    st.warning("SYNTHETIC demo data: fictional connectors, RMs, leads and payouts.", icon="⚠️")

rm_row = d["rms"].set_index("rm_id").loc[con.rm_id].to_dict() | {"rm_id": con.rm_id}
rm_first = con.rm_name.split()[0]

# ---------------------------------------------------------------------------
# 1. Connector app
# ---------------------------------------------------------------------------
if page.startswith("1"):
    st.title(f"{con['name'].split()[0]}'s Saathi")
    st.caption(f"{con.archetype} · {con.city} · {C.LANGS[con.language][0]} · digital maturity {con.digital_maturity} · "
               f"RM {con.rm_name}. The assistant is AI and says so; {rm_first} is one tap away.")
    left, right = st.columns([3, 2])
    hist = ss.chats.setdefault(cid, [])

    def send(msg):
        turn = agent.reply(cid, msg, [h for h in hist if h["role"] in ("user", "assistant")], lang=lang)
        hist.append(dict(role="user", content=msg))
        a = dict(role="assistant", content=turn.text, meta=dict(label=turn.label, conf=turn.confidence, sources=turn.sources,
                                                                  trace=turn.tool_trace, handoff=turn.handoff, mode=turn.mode,
                                                                  error=turn.error, events=[e[0] for e in turn.events]))
        audio, mime = speak(turn.text, lang, rm_row)
        if audio:
            a["audio"], a["mime"] = audio, mime
        hist.append(a)
        store.append("conversations", connector_id=cid, intent=turn.intent, handoff=turn.handoff, mode=turn.mode)

    with right:
        nd = nudges.next_nudge(ct[ct.connector_id == cid].iloc[0])
        st.markdown(f"**Today from {rm_first}** · _{nd['principle']}_")
        st.info(nd["message"])
        if st.button("▶ Play as voice note"):
            a, m = speak(nd["message"], lang, rm_row)
            if a:
                st.audio(a, format=m)
                st.caption("AI-generated voice" + (f" in {rm_first}'s cloned voice (with consent)" if K.get("TTS_PROVIDER", "").startswith("Eleven") and not ss.voice_killed else ""))
        k1, k2, k3 = st.columns(3)
        k1.metric("Paid, 90 days", inr(con.earned_90d))
        k2.metric("Coming Friday", inr(con.payout_processing))
        k3.metric("On hold", inr(con.payout_on_hold))
        k4, k5, k6 = st.columns(3)
        k4.metric("Tier", con.tier, help=f"Payout multiplier x{con.multiplier} (illustrative)")
        k5.metric("Lead streak", f"{con.streak_weeks} wk")
        k6.metric("To next tier", f"{con.next_tier_gap} disb.")
        if st.button(f"📞 Talk to {rm_first} (human)", type="primary"):
            send(f"I want to talk to {rm_first}")
            st.rerun()

    with left:
        chips = ["Payout kab aayega?", "Status of my leads", "Is hafte ka lead idea?",
                 "Client doctor hai, clinic ke liye 25 lakh chahiye", "Salary slip thoda adjust kar do", "Business loan documents?"]
        cc = st.columns(3)
        for i, ch in enumerate(chips):
            if cc[i % 3].button(ch, key=f"chip{i}"):
                send(ch)
                st.rerun()
        for h in hist[-14:]:
            with st.chat_message(h["role"], avatar="🧑‍💼" if h["role"] == "user" else "🤝"):
                st.markdown(h["content"])
                if h["role"] == "assistant":
                    m = h["meta"]
                    bits = [f"route: {m['label']} ({m['conf']:.2f})", m["mode"]]
                    if m["handoff"]:
                        bits.append(f"✋ handed to {rm_first}")
                    if m["events"]:
                        bits.append("actions: " + ", ".join(m["events"]))
                    st.caption(" · ".join(bits))
                    if m["sources"]:
                        st.caption("Sources: " + "; ".join(m["sources"]))
                    if m["error"]:
                        st.caption(m["error"])
                    if h.get("audio"):
                        st.audio(h["audio"], format=h["mime"])
                    if m["trace"]:
                        with st.expander("Tool calls"):
                            st.json(m["trace"])
        if K.get("STT_PROVIDER", "Off") != "Off" and hasattr(st, "audio_input"):
            rec = st.audio_input("🎙️ Send a voice note")
            if rec is not None:
                rid = hash(rec.getvalue()[:2000])
                if rid != ss.last_audio_id:
                    ss.last_audio_id = rid
                    try:
                        txt = voice.stt(rec.getvalue(), "note.wav", K["STT_PROVIDER"], K, con.language)
                        if txt:
                            send(txt)
                            st.rerun()
                    except voice.VoiceError as e:
                        st.error(str(e))
        msg = st.chat_input(f"Message {rm_first}'s assistant (any language) ...")
        if msg:
            send(msg)
            st.rerun()

# ---------------------------------------------------------------------------
# 2. Add a lead
# ---------------------------------------------------------------------------
elif page.startswith("2"):
    st.title("Add a lead: describe the customer, get the right product")
    st.caption("Type or speak the way you would to your RM. Eligibility is indicative; the bank decides.")
    txt = st.text_area("Customer, in your words", "Mera client Dr. Sunil Patil hai, apna clinic expand karna hai, 25 lakh chahiye, monthly income 3 lakh, koi EMI nahi")
    if st.button("Understand this customer"):
        ss.profile = recommend.extract_with_llm(txt, llm) if llm else recommend.extract_offline(txt)
    prof = ss.get("profile")
    if prof:
        st.caption(f"Read by: {prof.get('source')}")
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Customer name", prof.get("customer_name") or "")
        occ = c2.text_input("Occupation", prof.get("occupation") or "")
        pur = c3.text_input("Purpose", prof.get("purpose") or "")
        amt = c1.number_input("Amount needed (Rs)", value=float(prof.get("amount_needed") or 0), step=50000.0)
        inc = c2.number_input("Monthly income (Rs)", value=float(prof.get("monthly_income") or 0), step=5000.0)
        emi = c3.number_input("Existing EMIs (Rs/month)", value=float(prof.get("existing_emi") or 0), step=1000.0)
        own = c1.checkbox("Owns property", value=bool(prof.get("owns_property")))
        matches = recommend.match(dict(occupation=occ, purpose=pur, amount_needed=amt, monthly_income=inc, existing_emi=emi, owns_property=own))
        rows = [dict(product=m["product"], why=m["why"],
                     indicative_max=f"Rs {m['indicative_max']:,.0f}" if m["indicative_max"] else "-",
                     indicative_emi=f"Rs {m['indicative_emi']:,.0f}" if m["indicative_emi"] else "-",
                     documents=", ".join(m["documents"])) for m in matches]
        table(pd.DataFrame(rows))
        prod = st.selectbox("Product to submit", [m["product"] for m in matches])
        last4 = st.text_input("Customer phone, last 4 digits only (data minimisation)", max_chars=4)
        consent = st.checkbox("The customer has agreed to share their details with IDFC FIRST Bank for this application (required)")
        if st.button("Submit lead", type="primary", disabled=not (consent and name)):
            tb = Toolbox(d, ct, kb, cid)
            res = tb.submit_lead(name, prod, amt, last4, notes=txt[:200], consent_confirmed=consent)
            st.success(f"Lead {res['lead_ref']} submitted. Next: {res['next_step']}")
            if res.get("handoff"):
                st.info(f"Big-ticket loan: {con.rm_name} has been asked to join personally.")

# ---------------------------------------------------------------------------
# 3. RM cockpit
# ---------------------------------------------------------------------------
elif page.startswith("3"):
    rms = d["rms"]
    rid = st.selectbox("RM", rms.rm_id, index=list(rms.rm_id).index(con.rm_id), format_func=lambda r: rms.set_index("rm_id").loc[r, "name"])
    rname = rms.set_index("rm_id").loc[rid, "name"]
    st.title(f"{rname}'s cockpit: the human moments")
    port = ct[ct.rm_id == rid]
    hand = [h for h in store.read("handoffs") if h.get("rm") == rname]
    done = {h.get("ref") for h in store.read("handoff_updates")}
    open_h = [h for h in hand if h["hash"] not in done]
    k = st.columns(5)
    k[0].metric("Connectors", len(port))
    k[1].metric("Active (lead in 30 days)", int((port.leads_30d > 0).sum()))
    k[2].metric("At risk of going quiet", int((port.dormancy_risk > 0.6).sum()))
    k[3].metric("Hand-offs waiting", len(open_h))
    k[4].metric("Compliance flags", len([f for f in store.read("compliance_flags") if f["connector_id"] in set(port.connector_id)]))
    st.subheader("Hand-offs from the Digital RM (raise hand)")
    if open_h:
        for h in open_h[::-1][:10]:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{h['connector']}** ({h['connector_id']}) · {h['urgency']} · {h['ts'][:16]}  \n{h['reason']}")
            if c2.button("Done", key=h["hash"]):
                store.append("handoff_updates", ref=h["hash"], status="Done", by=rname)
                st.rerun()
    else:
        st.info("No open hand-offs. Try 'Talk to RM' or an angry message on the Connector app page.")
    st.subheader("Call these connectors this week (value at risk)")
    port = port.assign(value_at_risk=port.dormancy_risk * (port.earned_90d + 2000))
    cand = port[port.stage.isin(["Slipping", "Dormant", "Signed up, never active"]) | (port.days_since_lead > 21)]
    top = cand.sort_values("value_at_risk", ascending=False).head(6)
    opener = {"en": "I noticed it's been a while; anything blocking you?", "hi": "Kaafi din ho gaye, koi dikkat toh nahi?",
              "mr": "Khup divas jhale, kahi adchan aahe ka?", "ta": "It's been a while; anything I can help with?",
              "te": "It's been a while; anything I can help with?"}
    table(top[["connector_id", "name", "archetype", "stage", "days_since_lead", "dormancy_risk", "earned_90d", "language"]]
          .assign(opener=top.language.map(opener)))
    st.subheader("Portfolio by stage")
    sc = port.stage.value_counts().reindex(["New - not activated", "Signed up, never active", "Activated", "Habitual", "Slipping", "Dormant"]).fillna(0)
    fig = go.Figure(go.Bar(x=sc.values, y=sc.index, orientation="h", marker_color=MAROON))
    show(fig, height=260)
    conv = store.read("conversations")
    if conv:
        cv = pd.DataFrame(conv)
        cv = cv[cv.connector_id.isin(port.connector_id)]
        if len(cv):
            st.caption(f"This session: {len(cv)} conversations handled, {int(cv.handoff.sum())} needed {rname.split()[0]} "
                       f"({100 * (1 - cv.handoff.mean()):.0f}% handled by the Digital RM alone).")

# ---------------------------------------------------------------------------
# 4. Persona studio
# ---------------------------------------------------------------------------
elif page.startswith("4"):
    rms = d["rms"]
    rid = st.selectbox("RM to clone", rms.rm_id, index=list(rms.rm_id).index(con.rm_id), format_func=lambda r: rms.set_index("rm_id").loc[r, "name"])
    rrow = rms.set_index("rm_id").loc[rid].to_dict() | {"rm_id": rid}
    st.title(f"Clone {rrow['name']}'s way of talking")
    st.caption("An RM's call notes become a style card, "
               "and a consented voice sample becomes a cloned voice. The bot always says it is the RM's AI assistant.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Style card from call notes")
        tx = st.text_area("Call notes / WhatsApp scripts (one per line)", "\n".join(d["transcripts"][rid]), height=200)
        b1, b2 = st.columns(2)
        if b1.button("Build style card (offline)"):
            ss.cards[rid] = persona.style_card(rrow, [t for t in tx.splitlines() if t.strip()])
        if b2.button("Enrich with LLM", disabled=llm is None):
            ss.cards[rid] = persona.enrich_with_llm(persona.style_card(rrow, [t for t in tx.splitlines() if t.strip()]),
                                                     [t for t in tx.splitlines() if t.strip()], llm)
        st.json(ss.cards[rid])
        with st.expander("System prompt the LLM receives"):
            st.code(persona.system_prompt(ss.cards[rid], {"name": "Sunita P.", "archetype": "CA / tax consultant"}, "mr", "(retrieved knowledge)"), language="text")
    with c2:
        st.subheader("2. Voice clone (ElevenLabs)")
        consent = st.checkbox(f"{rrow['name']} has given written consent to create a synthetic copy of their voice for "
                              "messages to their connectors. Every message is labelled AI. Consent can be withdrawn.")
        files = st.file_uploader("Voice samples (1-3 minutes total, clean speech)", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
        if st.button("Create voice clone", disabled=not (consent and files and K.get("ELEVENLABS_API_KEY"))):
            try:
                vid = voice.clone_voice_elevenlabs(K["ELEVENLABS_API_KEY"], f"Saathi - {rrow['name']}", [(f.name, f.getvalue()) for f in files])
                K[f"ELEVENLABS_VOICE_ID_{rid}"] = vid
                store.append("voice_consents", rm_id=rid, rm=rrow["name"], voice_id=vid, consent=True)
                st.success(f"Voice created: {vid}. Set 'Voice out' to ElevenLabs in Settings.")
            except voice.VoiceError as e:
                st.error(str(e))
        if not K.get("ELEVENLABS_API_KEY"):
            st.caption("Add an ElevenLabs key in Settings to enable cloning.")
        K[f"ELEVENLABS_VOICE_ID_{rid}"] = st.text_input("...or paste an existing ElevenLabs voice ID for this RM",
                                                      K.get(f"ELEVENLABS_VOICE_ID_{rid}", ""))
        ss.voice_killed = st.toggle("Kill switch: stop using the cloned voice (falls back to a generic voice)", value=ss.voice_killed)
        st.subheader("3. Hear it")
        sample = st.text_input("Text", d["transcripts"][rid][0])
        if st.button("Speak"):
            a, m = speak(sample, rrow["langs"].split(",")[0], rrow)
            if a:
                st.audio(a, format=m)
                st.caption("AI-generated voice")

# ---------------------------------------------------------------------------
# 5. Nudges
# ---------------------------------------------------------------------------
elif page.startswith("5"):
    st.title("Nudge engine: the right push, in the right language, at the right time")
    rid = st.selectbox("Portfolio", ["All"] + list(d["rms"].rm_id), format_func=lambda r: r if r == "All" else d["rms"].set_index("rm_id").loc[r, "name"])
    port = ct if rid == "All" else ct[ct.rm_id == rid]
    nd = pd.DataFrame([nudges.next_nudge(r) for _, r in port.iterrows()])
    c1, c2 = st.columns(2)
    with c1:
        pc = nd.principle.value_counts()
        show(go.Figure(go.Bar(x=pc.values, y=pc.index, orientation="h", marker_color=MAROON)), height=280)
    with c2:
        ch = nd.channel.value_counts()
        show(go.Figure(go.Bar(x=ch.values, y=ch.index, orientation="h", marker_color="#4a6fa5")), height=280)
    st.caption(f"Caps: at most {C.NUDGE_CAP_PER_WEEK} nudges a week, none between {C.QUIET_HOURS[0]}:00 and {C.QUIET_HOURS[1]}:00, "
               "WhatsApp only with consent. Dormant connectors get an RM call task, not another message.")
    table(nd.merge(port[["connector_id", "name", "archetype"]], on="connector_id")[["connector_id", "name", "archetype", "stage", "principle", "channel", "send_at", "message", "blocked"]])

# ---------------------------------------------------------------------------
# 6. Quality & governance
# ---------------------------------------------------------------------------
elif page.startswith("6"):
    st.title("Quality and governance")
    st.subheader("Behaviour checks (run on the selected connector)")
    if st.button(f"Run 10 checks with: {mode_txt}"):
        ss.eval = evaluate.run(agent, cid, lang)
    if "eval" in ss:
        e = ss.eval
        st.metric("Checks passed", f"{int(e.passed.sum())} / {len(e)}")
        table(e)
    st.subheader("Guardrails built in")
    st.markdown("""
| Guardrail | How it works | Rule it answers |
|---|---|---|
| AI disclosure | First reply says "I'm {RM}'s digital assistant (AI)"; voice notes labelled AI | RBI FREE-AI; RBI draft MRM 2026 para 59 |
| Route to a human | "Talk to RM" always visible; anger, disputes, requests and loans above Rs 50 lakh hand over automatically | RBI draft MRM 2026 (route to human) |
| Customer consent gate | No lead is submitted without the consent tick; consent logged | RBI Digital Lending Directions 2025; DPDP Act |
| Fair-practice coach | Requests to alter documents or skip consent are refused, coached and flagged to the RM | DSA conduct: 30% of IDFC FIRST complaints (FY26) |
| Voice consent and kill switch | RM's written consent before cloning; one switch reverts to a generic voice | FREE-AI "People First" |
| Grounded answers | Only tool results and cited library text; "I won't guess" otherwise | FREE-AI "Understandable by Design" |
| Data minimisation | Customer phone stored as last 4 digits only; connector-level data by default | DPDP Act |
| Tamper-evident logs | Hand-offs, consents, flags and leads are hash-chained | Audit |
""")
    st.subheader("Logs")
    for kind in ["handoffs", "compliance_flags", "consents", "leads_submitted", "tickets", "voice_consents"]:
        rows = store.read(kind)
        with st.expander(f"{kind} ({len(rows)}) · chain {'intact' if store.verify(kind) else 'BROKEN'}"):
            if rows:
                table(pd.DataFrame(rows).drop(columns=["prev_hash"]).tail(20))
    st.subheader("RAFT fine-tuning dataset")
    st.caption("Question + oracle chunk + distractor chunks, so a small in-house model learns to answer only from the right document.")
    if st.button("Build RAFT dataset" + (" with the LLM" if llm else " (template questions)")):
        path = raft.build(use_llm=llm is not None)
        lines = open(path, encoding="utf-8").read().splitlines()
        st.success(f"{len(lines)} training records written to {path}")
        st.json(json.loads(lines[0]))
        st.download_button("Download raft_dataset.jsonl", "\n".join(lines), file_name="raft_dataset.jsonl")

# ---------------------------------------------------------------------------
# 7. Settings
# ---------------------------------------------------------------------------
elif page.startswith("7"):
    st.title("Settings: your API keys")
    if C.CLOUD:
        st.caption("Online demo: keys you type stay in your browser session only and are never saved on the server. "
                   "Without any key the app runs in offline mode.")
    else:
        st.caption("Keys stay in this session's memory. Tick 'Remember on this computer' to save them to .saathi_keys.env in this folder "
                   "(plain text: don't tick on a shared PC). Without any key the app runs in offline mode.")
    st.subheader("1. Brain (LLM)")
    provs = [n for n in C.PROVIDERS if not (C.CLOUD and n.startswith("Ollama"))]
    cur = K.get("LLM_PROVIDER", "Offline (no LLM)")
    prov = st.selectbox("Provider", provs, index=provs.index(cur) if cur in provs else provs.index("Offline (no LLM)"))
    p = C.PROVIDERS[prov]
    if prov != K.get("LLM_PROVIDER"):
        K["LLM_MODEL"], K["LLM_BASE_URL"] = p["default_model"], p["base_url"] or ""
    K["LLM_PROVIDER"] = prov
    if p["kind"] != "offline":
        if p["key_url"]:
            st.caption(f"Get a key: {p['key_url']}")
        K["LLM_API_KEY"] = st.text_input("API key", K.get("LLM_API_KEY", ""), type="password")
        c1, c2 = st.columns([3, 1])
        K["LLM_MODEL"] = c1.text_input("Model", K.get("LLM_MODEL") or p["default_model"])
        if c2.button("List models"):
            try:
                ss.models = LLM(prov, K["LLM_API_KEY"], K["LLM_MODEL"], K.get("LLM_BASE_URL", "")).list_models()
            except Exception as e:
                st.error(f"Could not list models: {e}")
        if ss.get("models"):
            pick = st.selectbox("Models available to your key", ss.models)
            if st.button("Use this model"):
                K["LLM_MODEL"] = pick
                st.rerun()
        if p["kind"] == "openai":
            K["LLM_BASE_URL"] = st.text_input("Base URL (OpenAI-compatible)", K.get("LLM_BASE_URL") or (p["base_url"] or ""))
        if st.button("Test connection"):
            try:
                st.success(test_connection(LLM(prov, K.get("LLM_API_KEY", ""), K["LLM_MODEL"], K.get("LLM_BASE_URL", ""))))
            except Exception as e:
                st.error(str(e))
    st.subheader("2. Voice out (the Digital RM speaks)")
    K["TTS_PROVIDER"] = st.selectbox("Text-to-speech", TTS_OPTIONS, index=TTS_OPTIONS.index(K.get("TTS_PROVIDER", "Off")))
    c1, c2 = st.columns(2)
    K["ELEVENLABS_API_KEY"] = c1.text_input("ElevenLabs API key (voice cloning)", K.get("ELEVENLABS_API_KEY", ""), type="password")
    K["ELEVENLABS_MODEL"] = c1.selectbox("ElevenLabs model", C.ELEVEN_MODELS,
                                         index=C.ELEVEN_MODELS.index(K.get("ELEVENLABS_MODEL") or C.ELEVEN_MODELS[0]))
    K["ELEVENLABS_VOICE_ID"] = c1.text_input("Default ElevenLabs voice ID (or create one in Persona studio)", K.get("ELEVENLABS_VOICE_ID", ""))
    K["SARVAM_API_KEY"] = c2.text_input("Sarvam API key (Indian voices and speech-to-text)", K.get("SARVAM_API_KEY", ""), type="password")
    c2.caption("edge-tts needs no key, only internet.")
    st.subheader("3. Voice in (connector sends a voice note)")
    K["STT_PROVIDER"] = st.selectbox("Speech-to-text", STT_OPTIONS, index=STT_OPTIONS.index(K.get("STT_PROVIDER", "Off")))
    K["OPENAI_STT_KEY"] = st.text_input("OpenAI key for transcription (if not your LLM provider)", K.get("OPENAI_STT_KEY", ""), type="password")
    st.divider()
    remember = False if C.CLOUD else st.checkbox("Remember on this computer", value=C.KEYS_FILE.exists())
    c1, c2 = st.columns(2)
    if c1.button("Save settings", type="primary"):
        if remember:
            keys.save(K)
            st.success(f"Saved to {C.KEYS_FILE.name}")
        else:
            st.success("Applied for this session only")
        ss.pop("llm_sig", None)
        st.rerun()
    if not C.CLOUD and c2.button("Forget saved keys"):
        keys.forget()
        st.success("Deleted the saved keys file")
    st.caption("Status: " + " · ".join(f"{n}: {keys.mask(K.get(n, ''))}" for n in ["LLM_API_KEY", "ELEVENLABS_API_KEY", "SARVAM_API_KEY", "OPENAI_STT_KEY"]))

# ---------------------------------------------------------------------------
# 8. How it is built
# ---------------------------------------------------------------------------
else:
    st.title("How it is built")
    st.markdown("""
| Piece | What it does | Design pattern |
|---|---|---|
| Persona card | RM's call notes become a style card; LLM can enrich it | Persona prompt built from a person's own words |
| Voice clone | Consented RM voice via ElevenLabs; Sarvam / edge-tts Indian voices otherwise | Instant voice cloning (ElevenLabs), consent-gated |
| Knowledge (RAG) | Product, payout, fair-practice and playbook library, cited | Retrieval-augmented generation with citations |
| Standalone question | Rewrites follow-ups using chat history before retrieval | Standalone-question rewrite |
| Router | Lead status / payout / new lead / product / ideas / issue / RM / compliance / chat | Classify-then-route |
| Tools + LLM loop | Leads, payouts, product match, submit lead (consent-gated), tickets, hand-off | Tool-using agent loop |
| Raise hand | Hands the conversation to the human RM with a brief | Human-in-the-loop hand-off |
| RAFT dataset | Oracle + distractor training records for an in-house model | RAFT (oracle + distractor documents) |
| Checks | Route, disclosure, consent, refusal, hand-off | RAGAS-style evaluation |
| Nudges | Behavioural principles, caps, quiet hours, language | Behavioural science |
""")
    st.markdown("""
**Honesty notes**
- Connectors, RMs, leads, payouts, payout rates and tiers are synthetic or illustrative. Product facts marked [SOURCED] in `kb/` come from public IDFC FIRST material; the rest is placeholder.
- No API keys are hard-coded; you type your own in Settings.
- To go live: point `saathi/data.py` at the MyFIRST Partner app, LOS and payout systems; load the bank's product sheets into `kb/`; get each RM's written voice consent.
""")
