"""Shared Gemini chat model with sane defaults (timeout, model name) for FIRE + chat."""
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# gemini-2.5-* can hang or behave oddly on some keys/networks; 2.0-flash is stable on the Gemini API.
_DEFAULT_MODEL = "gemini-2.0-flash"


def make_chat_llm(temperature: float = 0.8) -> ChatGoogleGenerativeAI:
    model = os.getenv("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temperature,
        timeout=120.0,
        max_retries=2,
    )
