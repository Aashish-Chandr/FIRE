"""
In-process TTL cache — works like Redis but needs zero extra setup.
Stores AI roadmaps and chat responses keyed by a hash of the inputs.
TTL defaults: roadmaps 24h (same inputs = same result), chat 5min.
"""
import time
import hashlib
import json
import threading
from typing import Any, Optional

_store: dict = {}       # key -> (value, expires_at)
_lock = threading.Lock()


def _make_key(*parts) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def get(key: str) -> Optional[Any]:
    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del _store[key]
            return None
        return value


def set(key: str, value: Any, ttl: int = 3600) -> None:
    with _lock:
        _store[key] = (value, time.time() + ttl)


def delete(key: str) -> None:
    with _lock:
        _store.pop(key, None)


def clear_expired() -> None:
    """Call periodically to free memory."""
    now = time.time()
    with _lock:
        expired = [k for k, (_, exp) in _store.items() if now > exp]
        for k in expired:
            del _store[k]


# ── Convenience helpers ─────────────────────────────────────

def roadmap_key(user_data, calc_results: dict) -> str:
    """Cache key for a FIRE roadmap — based on inputs that affect the output."""
    return _make_key(
        user_data.age,
        user_data.monthly_income,
        user_data.monthly_expenses,
        user_data.current_savings,
        user_data.existing_investments,
        user_data.fire_target_age,
        user_data.monthly_expenses_post_fire,
        user_data.risk_profile.value,
        user_data.language.value,
        round(calc_results["fire_corpus_range"]["min"]),
    )


def chat_key(session_id: str, user_message: str) -> str:
    """Cache key for a chat reply — session + exact message."""
    return _make_key("chat", session_id, user_message.strip().lower())
