"""The Digital RM: one turn of conversation.

Flow:
  1. standalone question from chat history (so "and his payout?" still retrieves well)
  2. route the intent (classify-then-route) and check flags: anger, compliance, ask-for-human
  3. retrieve knowledge (RAG) for the standalone question
  4. LLM mode: persona system prompt + tools, tool loop  |  offline mode: rule-based reply from the same tools
  5. raise_hand: hand over to the human RM when trust is at stake
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config as C, persona, recommend
from .llm import LLMError
from .router import Router
from .tools import SCHEMAS, Toolbox


@dataclass
class Turn:
    text: str
    intent: str
    label: str
    confidence: float
    sources: list = field(default_factory=list)
    tool_trace: list = field(default_factory=list)
    events: list = field(default_factory=list)
    handoff: bool = False
    mode: str = "offline"
    standalone: str = ""
    error: str = ""


class DigitalRM:
    def __init__(self, data, contable, kb, cards: dict, llm=None):
        self.d, self.ct, self.kb, self.cards, self.llm = data, contable, kb, cards, llm
        self.router = Router()

    def reply(self, connector_id: str, message: str, history: list[dict], lang: str | None = None) -> Turn:
        con = self.ct.set_index("connector_id").loc[connector_id]
        lang = lang or con.language
        card = self.cards[con.rm_id]
        tb = Toolbox(self.d, self.ct, self.kb, connector_id, self.llm)
        standalone = self._standalone(message, history)
        r = self.router.route(standalone)
        ctx, srcs = self.kb.context(standalone, k=3)
        first_turn = not any(h["role"] == "assistant" for h in history)

        # trust moments: always bring the human in, whatever the model says
        if "anger" in r["flags"] or r["intent"] == "talk_to_rm":
            tb.handoff_to_rm(f"{'Connector upset' if 'anger' in r['flags'] else 'Asked for RM'}: \"{message[:120]}\"",
                             "high" if "anger" in r["flags"] else "normal")
        if r["intent"] == "compliance":
            tb.flag_compliance(message[:200])

        if self.llm is not None and self.llm.online:
            profile = {k: (con[k].item() if hasattr(con[k], "item") else con[k]) for k in
                       ("name", "archetype", "city", "language", "digital_maturity", "stage", "tier", "streak_weeks")}
            sysmsg = persona.system_prompt(card, profile, lang, ctx)
            msgs = [dict(role=h["role"], content=h["content"]) for h in history[-8:]] + [dict(role="user", content=message)]
            try:
                text, trace = self.llm.chat(msgs, sysmsg, SCHEMAS, tb.call)
                return Turn(text=text, intent=r["intent"], label=r["label"], confidence=r["confidence"], sources=srcs,
                            tool_trace=trace, events=tb.events, handoff=any(e[0] == "handoff" for e in tb.events),
                            mode=f"{self.llm.provider} / {self.llm.model}", standalone=standalone)
            except LLMError as e:
                t = self._offline(r, con, lang, tb, message, srcs, first_turn)
                t.error = f"LLM failed, answered offline: {e}"
                return t
        return self._offline(r, con, lang, tb, message, srcs, first_turn)

    # ------------------------------------------------------------------
    def _standalone(self, message, history):
        if not history or self.llm is None or not self.llm.online:
            return message
        h = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-6:])
        try:
            q = self.llm.complete(persona.STANDALONE_PROMPT.format(history=h, question=message), max_tokens=80)
            return q.strip() or message
        except LLMError:
            return message

    def _offline(self, r, con, lang, tb, message, srcs, first_turn) -> Turn:
        L = lang if lang in OFF else "en"
        T = OFF[L]
        first = con["name"].split()[0]
        rmf = con.rm_name.split()[0]
        i = r["intent"]
        lines = []
        if first_turn:
            lines.append(T["intro"].format(rm=rmf))
        used_srcs = []
        if i == "lead_status":
            res = tb.get_my_leads()
            open_ = res["open_by_stage"]
            if open_:
                lines.append(T["leads"].format(n=sum(open_.values()), stages=", ".join(f"{k} {v}" for k, v in open_.items())))
            else:
                lines.append(T["noleads"])
            pend = [x for x in res["leads"] if x["stage"] == "Documents pending"][:2]
            for x in pend:
                lines.append(T["pending"].format(cust=x["customer"], prod=x["product"], item=x["pending_item"]))
        elif i == "payout_status":
            res = tb.get_my_payouts()
            lines.append(T["payout"].format(paid=f"{res['paid_last_90d']:,.0f}", proc=f"{res['processing']:,.0f}",
                                            hold=f"{res['on_hold']:,.0f}", tier=res["tier"]))
            holds = [x for x in res["recent"] if x["status"] == "On hold"][:1]
            for x in holds:
                lines.append(T["hold"].format(reason=x["hold_reason"]))
        elif i == "new_lead":
            prof = recommend.extract_offline(message)
            m = recommend.match(prof)
            top = m[0]
            amt = f"Rs {prof['amount_needed']:,.0f}" if prof.get("amount_needed") else "-"
            lines.append(T["newlead"].format(prod=top["product"], amt=amt, docs=", ".join(top["documents"][:3])))
            lines.append(T["consent"])
            if prof.get("amount_needed") and prof["amount_needed"] >= C.HANDOFF_AMOUNT:
                tb.handoff_to_rm(f"High-value lead {amt} ({top['product']})", "high")
                lines.append(T["bigticket"].format(rm=rmf))
        elif i == "product_query":
            hits = self.kb.search(message, k=1)
            if hits:
                c = hits[0][0]
                lines.append(T["kb"].format(title=c.title, text=_short(c.text, 330)))
                used_srcs = [c.cite]
            else:
                lines.append(T["dontknow"].format(rm=rmf))
        elif i == "lead_ideas":
            res = tb.get_lead_ideas()
            lines.append(T["idea"].format(idea=res["idea_this_week"]))
        elif i == "issue":
            res = tb.raise_ticket("General", message[:200])
            lines.append(T["ticket"].format(t=res["ticket"], rm=rmf))
        elif i == "progress":
            res = tb.get_my_progress()
            lines.append(T["progress"].format(tier=res["tier"], gap=res["next_tier_gap"], streak=res["streak_weeks"]))
        elif i == "compliance":
            lines.append(T["compliance"])
            used_srcs = ["Fair Practice > Never do this"]
        elif i == "talk_to_rm":
            lines.append(T["handoff"].format(rm=rmf))
        else:
            lines.append(T["hello"].format(name=first))
        if "anger" in r["flags"] and i != "talk_to_rm":
            lines.append(T["handoff"].format(rm=rmf))
        if i in ("lead_status", "payout_status", "progress", "issue"):
            used_srcs = ["Your MyFIRST data (leads, payouts, tickets)"]
        elif i == "new_lead":
            used_srcs = [f"Products > {top['product']}", "Fair Practice > Customer consent"]
        elif i == "lead_ideas":
            used_srcs = [f"Lead Playbook > {con.archetype}"]
        return Turn(text="\n".join(lines), intent=i, label=r["label"], confidence=r["confidence"],
                    sources=used_srcs or ([] if i in ("smalltalk", "talk_to_rm") else srcs[:1]),
                    events=tb.events, handoff=any(e[0] == "handoff" for e in tb.events), mode="offline (no LLM key)")


def _short(t, n):
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "..."


OFF = {
    "en": dict(
        intro="Hi! I'm {rm}'s digital assistant (AI). I can check leads and payouts, match products and add leads. {rm} is one tap away.",
        leads="You have {n} open leads: {stages}.",
        noleads="No open leads right now. Shall I share this week's lead idea?",
        pending="{cust} ({prod}) is waiting for: {item}. Upload it in the app and the file moves to login.",
        payout="Paid in the last 90 days: Rs {paid}. Processing: Rs {proc} (credited Friday). On hold: Rs {hold}. Your tier: {tier}.",
        hold="One payout is on hold because: {reason}. Fix it in the app and it moves to the next Friday cycle.",
        newlead="Looks like a {prod} (amount {amt}). Keep these ready: {docs}. Eligibility is indicative; the bank decides.",
        consent="Before I submit: has the customer agreed to share their details with IDFC FIRST Bank? Reply 'yes, consent' and use Add a lead.",
        bigticket="This is a big-ticket loan, so I've asked {rm} to join personally.",
        kb="{title}: {text}",
        dontknow="I'm not sure about that one, and I won't guess. Shall I ask {rm} to call you?",
        idea="Idea for this week: {idea}. Want a ready-to-send WhatsApp message for your contacts?",
        ticket="Ticket {t} raised. {rm} has been notified; it escalates if still open after 7 days.",
        progress="You're {tier} tier, {gap} disbursal(s) from the next tier, with a {streak}-week lead streak.",
        compliance="I can't help with changing documents or skipping the customer's consent. It hurts the customer and can stop your payouts. Let's submit the real documents; if income is short, I can suggest a smaller amount or a co-applicant.",
        handoff="I've asked {rm} to call you personally. You'll hear back today.",
        hello="Hello {name}! Ask me about your leads, payouts, products or this week's lead idea.",
    ),
    "hi": dict(
        intro="Namaste! Main {rm} ji ka digital assistant (AI) hoon. Lead, payout, product aur naye lead mein madad karunga. {rm} ji bas ek tap door hain.",
        leads="Aapke {n} lead chal rahe hain: {stages}.",
        noleads="Abhi koi lead chalu nahi hai. Is hafte ka lead idea bataun?",
        pending="{cust} ({prod}) ke liye baaki hai: {item}. App mein upload kijiye, file login mein chali jayegi.",
        payout="Pichhle 90 din mein mila: Rs {paid}. Processing: Rs {proc} (Shukravar ko credit). Hold par: Rs {hold}. Aapka tier: {tier}.",
        hold="Ek payout hold par hai, karan: {reason}. App mein theek kijiye, agle Shukravar aa jayega.",
        newlead="Yeh {prod} lagta hai (amount {amt}). Yeh documents ready rakhiye: {docs}. Eligibility sirf andaza hai, faisla bank karega.",
        consent="Submit karne se pehle: kya customer ne IDFC FIRST Bank ko details dene ki permission di hai? 'haan, consent' likhiye aur 'Add a lead' use kijiye.",
        bigticket="Yeh bada loan hai, isliye maine {rm} ji ko khud judne ke liye kaha hai.",
        kb="{title}: {text}",
        dontknow="Iska pakka jawab mere paas nahi hai, andaza nahi lagaunga. {rm} ji se call karwa doon?",
        idea="Is hafte ka idea: {idea}. Contacts ko bhejne ke liye WhatsApp message bana doon?",
        ticket="Ticket {t} ban gaya. {rm} ji ko bata diya hai; 7 din mein band na hua toh escalate hoga.",
        progress="Aap {tier} tier mein hain, agle tier se {gap} disbursal door, aur {streak} hafte ki streak hai.",
        compliance="Documents badalna ya customer ki permission ke bina kuch bhejna, ismein main madad nahi kar sakta. Isse customer ka nuksaan hota hai aur aapka payout bhi ruk sakta hai. Asli documents bhejiye; income kam ho toh chhota amount ya co-applicant sochte hain.",
        handoff="Maine {rm} ji ko aapko khud call karne ko kaha hai. Aaj hi baat hogi.",
        hello="Namaste {name} ji! Lead, payout, product ya is hafte ke idea ke baare mein poochhiye.",
    ),
    "mr": dict(
        intro="Namaskar! Mi {rm} yancha digital assistant (AI) aahe. Lead, payout, product ani navin lead madhe madat karto. {rm} fakt ek tap var aahet.",
        leads="Tumche {n} lead chalu aahet: {stages}.",
        noleads="Sadhya ekahi lead chalu nahi. Ya aathavdyachi lead kalpana sangu ka?",
        pending="{cust} ({prod}) sathi baki aahe: {item}. App madhe upload kara, file login la jail.",
        payout="Gelya 90 divsat milale: Rs {paid}. Processing: Rs {proc} (Shukravari credit). Hold var: Rs {hold}. Tumcha tier: {tier}.",
        hold="Ek payout hold var aahe, karan: {reason}. App madhe durust kara, pudhchya Shukravari yeil.",
        newlead="He {prod} vatat aahe (amount {amt}). He documents tayar theva: {docs}. Eligibility fakt andaj aahe, nirnay bank gheil.",
        consent="Submit karnyaadhi: customer ne IDFC FIRST Bank la mahiti denyachi parvangi dili aahe ka? 'ho, consent' liha ani 'Add a lead' vapara.",
        bigticket="Ha motha loan aahe, mhanun mi {rm} yanna swatah join karayla sangitle aahe.",
        kb="{title}: {text}",
        dontknow="Yache nakki uttar majhyakade nahi, andaj lavnar nahi. {rm} yancha call lavu ka?",
        idea="Ya aathavdyachi kalpana: {idea}. Contacts na pathvayla WhatsApp message banvu ka?",
        ticket="Ticket {t} tayar jhala. {rm} yanna kalavle aahe; 7 divsat band na jhalyas escalate hoil.",
        progress="Tumhi {tier} tier madhe aahat, pudhchya tier pasun {gap} disbursal laamb, ani {streak} aathavdyanchi streak aahe.",
        compliance="Documents badalne kinva customer chya parvangi shivay kahi pathavne, yat mi madat karu shakat nahi. Yane customer che nuksan hote ani tumcha payout pan thambu shakto. Khare documents dya; income kami asel tar lahan amount kinva co-applicant baghuya.",
        handoff="Mi {rm} yanna tumhala swatah call karayla sangitle aahe. Aajach bolne hoil.",
        hello="Namaskar {name}! Lead, payout, product kinva ya aathavdyachi kalpana vichara.",
    ),
}
