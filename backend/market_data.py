"""
Fetches live Indian market data from Alpha Vantage.
Used to enrich AI advisor context with real current market conditions.

Symbols used:
  - BSE:SENSEX proxy → NIFTYBEES.BSE (Nifty ETF on BSE)
  - Gold proxy       → GOLDBEES.BSE
  - USD/INR forex    → from CURRENCY_EXCHANGE_RATE
  - US 10Y yield     → from TREASURY_YIELD (global risk indicator)

Free tier: 25 requests/day, 5/min — we cache results for 1 hour.
"""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
BASE_URL = "https://www.alphavantage.co/query"

# Simple in-memory cache: {cache_key: (timestamp, data)}
_cache: dict = {}
CACHE_TTL = 3600  # 1 hour


def _get(params: dict) -> dict:
    cache_key = str(sorted(params.items()))
    now = time.time()
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if now - ts < CACHE_TTL:
            return data
    try:
        params["apikey"] = API_KEY
        r = requests.get(BASE_URL, params=params, timeout=8)
        data = r.json()
        _cache[cache_key] = (now, data)
        return data
    except Exception as e:
        print(f"[market_data] Alpha Vantage error: {e}")
        return {}


def get_quote(symbol: str) -> dict:
    """Get latest price quote for a symbol."""
    data = _get({"function": "GLOBAL_QUOTE", "symbol": symbol})
    q = data.get("Global Quote", {})
    if not q or not q.get("05. price"):
        return {}
    return {
        "symbol": symbol,
        "price": float(q.get("05. price", 0)),
        "change_pct": q.get("10. change percent", "0%").replace("%", ""),
        "prev_close": float(q.get("08. previous close", 0)),
    }


def get_usd_inr() -> float:
    """Get current USD/INR exchange rate."""
    data = _get({
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "USD",
        "to_currency": "INR"
    })
    rate = data.get("Realtime Currency Exchange Rate", {}).get("5. Exchange Rate")
    return float(rate) if rate else 0.0


def get_market_context() -> str:
    """
    Returns a formatted string of current Indian market conditions
    to be injected into the AI advisor prompt.
    """
    if not API_KEY:
        return ""

    lines = ["=== LIVE MARKET DATA (Alpha Vantage) ==="]

    # Nifty 50 ETF proxy — try multiple symbols
    nifty = get_quote("NIFTYBEES.BSE")
    if not nifty:
        nifty = get_quote("BSE:NIFTYBEES")
    if nifty:
        lines.append(f"Nifty 50 ETF (NIFTYBEES): ₹{nifty['price']:.2f} ({nifty['change_pct']}% today)")

    # Gold ETF proxy
    gold = get_quote("GOLDBEES.BSE")
    if gold:
        lines.append(f"Gold ETF (GOLDBEES): ₹{gold['price']:.2f} ({gold['change_pct']}% today)")

    # USD/INR
    usd_inr = get_usd_inr()
    if usd_inr:
        lines.append(f"USD/INR: ₹{usd_inr:.2f}")

    if len(lines) == 1:
        return ""  # No data fetched, don't add empty section

    lines.append("Use this data to give more accurate, current advice on equity allocation and currency risk.")
    return "\n".join(lines)
