# Handles chat with memory + RAG + AI
from langchain_core.prompts import PromptTemplate
from rag.retriever import retrieve_relevant_context
from market_data import get_market_context
import cache

# Lazy LLM — initialized on first use, not at import time
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        from deepseek_client import make_chat_llm
        _llm = make_chat_llm(temperature=1.0)
    return _llm

LANGUAGE_INSTRUCTIONS = {
    "english":  "Respond ONLY in English.",
    "hindi":    "पूरा जवाब केवल हिंदी में दें। Financial terms जैसे SIP, ELSS को वैसे ही रखें।",
    "hinglish": "Respond in Hinglish — natural mix of Hindi and English.",
}

CHAT_PROMPT = PromptTemplate(
    input_variables=["language_instruction", "context", "market_data", "history", "fire_summary", "user_message"],
    template="""
You are a friendly Indian personal finance advisor for a FIRE planning app.
Answer in a conversational but accurate way. Never make up numbers.
If unsure, say so honestly rather than guessing.

LANGUAGE RULE: {language_instruction}

=== USER'S FIRE PLAN (for context) ===
{fire_summary}

=== RELEVANT KNOWLEDGE BASE ===
{context}

{market_data}

=== CONVERSATION SO FAR ===
{history}

=== USER JUST ASKED ===
{user_message}

=== YOUR TASK ===
- Answer clearly and concisely
- Use ₹ symbol where needed
- Give India-specific advice
- Reference current market data above if relevant
- Prefer actionable suggestions
""",
)

_chat_chain = None

def _get_chat_chain():
    global _chat_chain
    if _chat_chain is None:
        _chat_chain = CHAT_PROMPT | _get_llm()
    return _chat_chain


def format_history(messages: list) -> str:
    if not messages:
        return "This is the start of the conversation."
    lines = []
    for msg in messages[-10:]:
        role = "User" if msg["role"] == "user" else "Advisor"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _knowledge_fallback(user_message: str, context: str, fire_summary: str) -> str:
    """
    Rule-based fallback when AI is unavailable.
    Searches the retrieved context for relevant snippets and returns them
    directly — always useful, never shows a quota error to the user.
    """
    msg = user_message.lower()

    # Try to give a relevant answer from the knowledge base context
    if context and len(context) > 100:
        # Return the most relevant snippet from the knowledge base
        lines = [l.strip() for l in context.split("\n") if len(l.strip()) > 40]
        if lines:
            snippet = " ".join(lines[:6])
            return (
                f"Based on our knowledge base: {snippet[:600]}\n\n"
                "For a more detailed personalised answer, please try again in a moment."
            )

    # Topic-specific fallbacks using fire_summary data
    if any(w in msg for w in ["sip", "invest", "mutual fund", "equity"]):
        return (
            "For SIP investments, start with index funds (Nifty 50) for equity exposure. "
            "Use ELSS funds to save tax under Section 80C. "
            "Increase your SIP by 10% every year to beat inflation. "
            "Platforms like Zerodha Coin, Groww, or Kuvera are good for direct mutual funds."
        )
    if any(w in msg for w in ["tax", "80c", "nps", "elss", "ppf"]):
        return (
            "Key tax-saving options in India: "
            "Section 80C (₹1.5L limit) — PPF, ELSS, EPF, NSC. "
            "Section 80CCD(1B) — NPS extra ₹50,000 deduction. "
            "Section 80D — Health insurance premium up to ₹25,000. "
            "Hold equity investments over 1 year for LTCG tax benefit."
        )
    if any(w in msg for w in ["emergency", "fund", "liquid"]):
        return (
            "Keep 6 months of expenses in a liquid fund or high-interest savings account. "
            "Liquid mutual funds give ~6-7% returns and can be redeemed in 1 business day. "
            "Don't invest your emergency fund in equity — it needs to be accessible immediately."
        )
    if any(w in msg for w in ["insurance", "term", "health", "cover"]):
        return (
            "Term insurance: Get 15–20× your annual income as cover. "
            "Pure term plans (no investment component) are cheapest — LIC Tech Term, HDFC Click2Protect. "
            "Health insurance: ₹10–20 lakh family floater is recommended. "
            "Buy both before age 35 for lowest premiums."
        )
    if any(w in msg for w in ["fire", "retire", "corpus", "withdrawal"]):
        return (
            "The 4% withdrawal rule: You can safely withdraw 4% of your corpus annually in retirement. "
            "So if you need ₹60,000/month (₹7.2L/year), you need ₹1.8 Cr corpus. "
            "Indian FIRE planners often use 3.5% to account for higher inflation. "
            "Diversify your retirement corpus across equity, debt, and gold."
        )

    return (
        "I'm here to help with your FIRE planning. "
        "You can ask me about SIP strategy, tax optimization, asset allocation, "
        "emergency funds, insurance, or your retirement timeline. "
        "What would you like to know?"
    )


def get_chat_response(user_message: str, chat_history: list,
                      language: str = "english", fire_summary: str = "") -> tuple:
    context     = retrieve_relevant_context(user_message, k=4)
    market_data = get_market_context()

    # Check cache — same session message within 5 min returns instantly
    ckey   = cache.chat_key(fire_summary[:100], user_message)
    cached = cache.get(ckey)
    if cached:
        print("[chat_advisor] Serving reply from cache.")
        return cached, context

    language_instruction = LANGUAGE_INSTRUCTIONS.get(
        str(language).lower(), LANGUAGE_INSTRUCTIONS["english"]
    )
    history_text = format_history(chat_history)
    fire_summary = fire_summary or "User has not created a FIRE plan yet."

    try:
        response = _get_chat_chain().invoke({
            "language_instruction": language_instruction,
            "context": context,
            "market_data": market_data,
            "history": history_text,
            "fire_summary": fire_summary,
            "user_message": user_message,
        })
        result = response.content
        # Cache for 5 minutes
        cache.set(ckey, result, ttl=300)
        return result, context

    except Exception as e:
        print(f"[chat_advisor] AI unavailable: {str(e)[:200]}")
        # Never show quota/billing errors to the user — serve a helpful fallback
        fallback = _knowledge_fallback(user_message, context, fire_summary)
        return fallback, context
