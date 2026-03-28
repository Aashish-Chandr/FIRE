# Generates a personalized FIRE roadmap using Gemini + RAG + calculator outputs
from langchain_core.prompts import PromptTemplate
from rag.retriever import retrieve_relevant_context
from deepseek_client import make_chat_llm
from market_data import get_market_context
import cache

llm = make_chat_llm(temperature=1.0)


def _fallback_roadmap(calc_results: dict, user_data) -> str:
    """
    Rule-based roadmap when AI quota is exhausted.
    Uses the calculator results to generate useful, accurate advice
    without any API call.
    """
    r = calc_results
    sip_min = r['monthly_sip_range']['min']
    sip_max = r['monthly_sip_range']['max']
    corpus_min = r['fire_corpus_range']['min']
    corpus_max = r['fire_corpus_range']['max']
    years = r['years_to_fire']
    savings_rate = r['savings_rate_percent']
    fire_type = r['fire_type']
    alloc = r['asset_allocation']
    emergency = r['emergency_fund_needed']

    alloc_lines = "\n".join(f"  - {k}: {v}%" for k, v in alloc.items())

    return f"""### Your Personalized FIRE Roadmap

**FIRE Target Summary**
You need ₹{corpus_min:,.0f} – ₹{corpus_max:,.0f} to retire in {years} years ({fire_type}).
Your required monthly SIP: ₹{sip_min:,.0f} – ₹{sip_max:,.0f}

---

### 1. 6-Month Action Plan

**Month 1–2: Foundation**
- Open a dedicated investment account (Zerodha, Groww, or Kuvera)
- Set up auto-debit SIP of ₹{sip_min:,.0f}/month starting immediately
- Build emergency fund of ₹{emergency:,.0f} (6 months expenses) in liquid fund

**Month 3–4: Tax Optimization**
- Maximize Section 80C (₹1.5L limit): PPF, ELSS mutual funds
- Start NPS contribution for additional ₹50,000 deduction under 80CCD(1B)
- Review Form 16 and ensure TDS is correctly deducted

**Month 5–6: Insurance & Review**
- Get term insurance cover of ₹{round(user_data.monthly_income * 12 * 15 / 100000) * 100000:,.0f}
- Get health insurance of ₹10–20 lakh for family
- Review and increase SIP by 10% annually (step-up SIP)

---

### 2. Recommended Asset Allocation

Based on your {user_data.risk_profile.value} risk profile:
{alloc_lines}

---

### 3. Key Milestones

- **Year 1**: Emergency fund complete + SIP running
- **Year {years // 3}**: First major corpus review — rebalance if needed
- **Year {years // 2}**: Shift 5% from equity to debt as you approach FIRE
- **Year {years - 2}**: Final rebalancing — move to conservative allocation
- **Year {years}**: FIRE achieved at age {user_data.fire_target_age}

---

### 4. Risk Warnings

- Market returns are not guaranteed — actual returns may vary ±3%
- Inflation at 6% is assumed — higher inflation extends your timeline
- Healthcare costs in India are rising 12–15% annually — plan accordingly
- Maintain your savings rate of {savings_rate:.1f}% consistently

*Note: This is a rule-based summary. For a full AI-generated narrative, try again in a few minutes.*"""


LANGUAGE_INSTRUCTIONS = {
    "english":  "Respond ONLY in English.",
    "hindi":    "पूरा जवाब केवल हिंदी में दें। SIP, ELSS, PPF जैसे financial terms को वैसे ही रखें।",
    "hinglish": "Respond in Hinglish — natural mix of Hindi and English like educated Indians speak daily."
}


FIRE_PROMPT = PromptTemplate(
    input_variables=["language_instruction", "context", "market_data", "user_profile", "calc_results"],
    template="""
You are an expert Indian financial advisor specializing in FIRE planning.
Use ONLY the context below to give grounded, specific advice. Never make up numbers.

LANGUAGE RULE: {language_instruction}

=== KNOWLEDGE BASE CONTEXT ===
{context}

{market_data}

=== USER PROFILE ===
{user_profile}

=== CALCULATED FIRE RESULTS ===
{calc_results}

=== YOUR TASK ===
Based on the knowledge base and user's numbers, provide:

1. 6-Month Action Plan (monthly steps)
2. SIP Strategy (use calculated SIP range)
3. Tax Optimization Plan (India-specific)
4. Emergency Fund Plan
5. Insurance Plan
6. FIRE Type Analysis
7. Key Risk Warning

Use ₹ symbol. Be India-specific. Be practical and actionable.
"""
)


fire_chain = FIRE_PROMPT | llm


def get_ai_roadmap(user_data, calc_results):
    # Check cache first — same inputs always produce same roadmap
    ckey = cache.roadmap_key(user_data, calc_results)
    cached = cache.get(ckey)
    if cached:
        print("[ai_advisor] Serving roadmap from cache.")
        return cached

    search_query = f"""
    FIRE planning India age {user_data.age}
    {user_data.risk_profile} mutual funds SIP allocation
    tax saving 80C insurance emergency fund
    income {user_data.monthly_income}
    """

    context    = retrieve_relevant_context(search_query, k=6)
    market_data = get_market_context()

    user_profile = f"""
Name: {user_data.name}
Age: {user_data.age} years
Monthly Income: ₹{user_data.monthly_income:,.0f}
Monthly Expenses: ₹{user_data.monthly_expenses:,.0f}
Current Savings: ₹{user_data.current_savings:,.0f}
Existing Investments: ₹{user_data.existing_investments:,.0f}
Target FIRE Age: {user_data.fire_target_age} years
Post-FIRE Monthly Expenses: ₹{user_data.monthly_expenses_post_fire:,.0f}
Risk Profile: {user_data.risk_profile}
"""

    calc_str = f"""
FIRE Corpus Range: ₹{calc_results['fire_corpus_range']['min']:,.0f} - ₹{calc_results['fire_corpus_range']['max']:,.0f}
Monthly SIP Range: ₹{calc_results['monthly_sip_range']['min']:,.0f} - ₹{calc_results['monthly_sip_range']['max']:,.0f}
Years to FIRE: {calc_results['years_to_fire']}
Monthly Savings: ₹{calc_results['monthly_savings']:,.0f}
Savings Rate: {calc_results['savings_rate_percent']}%
Savings Insight: {calc_results['savings_insight']}
Projected Existing Corpus: ₹{calc_results['projected_existing_corpus']:,.0f}
Emergency Fund Needed: ₹{calc_results['emergency_fund_needed']:,.0f}
FIRE Type: {calc_results['fire_type']}
Recommended Asset Allocation: {calc_results['asset_allocation']}
Tax Suggestions: {", ".join(calc_results['tax_suggestions'])}
Insurance Advice:
- Term Insurance: {calc_results['insurance_advice']['term_insurance']}
- Health Insurance: {calc_results['insurance_advice']['health_insurance']}
Warnings: {", ".join(calc_results['warnings'])}
"""

    language_instruction = LANGUAGE_INSTRUCTIONS.get(
        str(user_data.language.value).lower(),
        LANGUAGE_INSTRUCTIONS["english"]
    )

    try:
        response = fire_chain.invoke({
            "language_instruction": language_instruction,
            "context": context,
            "market_data": market_data,
            "user_profile": user_profile,
            "calc_results": calc_str,
        })
        result = response.content
        # Cache for 24 hours — same inputs = same roadmap
        cache.set(ckey, result, ttl=86400)
        return result
    except Exception as e:
        print(f"[ai_advisor] AI unavailable: {str(e)[:200]}")
        return _fallback_roadmap(calc_results, user_data)