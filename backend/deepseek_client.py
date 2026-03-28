"""
AI client — tries DeepSeek first, falls back to Gemini on 402/balance errors.
Returns a standard LangChain Runnable so FIRE_PROMPT | llm works normally.
"""
from dotenv import load_dotenv
import os

load_dotenv()

_deepseek_failed = False  # module-level flag: once balance is gone, skip DeepSeek


def make_chat_llm(temperature: float = 0.8):
    """
    Returns a LangChain chat model.
    Tries DeepSeek; if balance is exhausted falls back to Gemini.
    """
    global _deepseek_failed

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if deepseek_key and not _deepseek_failed:
        # Quick balance probe — cheap single-token call
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=deepseek_key,
                base_url="https://api.deepseek.com",
                temperature=temperature,
                max_tokens=4096,
            )
            llm.invoke([HumanMessage(content="hi")], max_tokens=1)
            print("[deepseek_client] DeepSeek OK — using DeepSeek.")
            return llm
        except Exception as e:
            err = str(e)
            if "402" in err or "Insufficient" in err or "balance" in err.lower():
                print("[deepseek_client] DeepSeek balance exhausted — falling back to Gemini.")
                _deepseek_failed = True
            else:
                print(f"[deepseek_client] DeepSeek error: {err[:200]} — falling back to Gemini.")
                _deepseek_failed = True

    # Gemini fallback
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("[deepseek_client] Using Gemini.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_key,
        temperature=temperature,
    )
