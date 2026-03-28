"""
Database layer with in-memory fallback.
Tries Supabase first; if RLS or connection fails, falls back to in-memory storage
so the app works without needing Supabase service key configuration.
"""
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# --- In-memory fallback storage ---
_users = {}        # email -> user dict
_plans = {}        # plan_id -> plan dict
_sessions = {}     # session_id -> session dict
_messages = {}     # session_id -> [message, ...]

# --- Try to init Supabase ---
_supabase = None
try:
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if url and key:
        _supabase = create_client(url, key)
except Exception:
    pass


def _use_db():
    """Quick check: try a lightweight Supabase call to see if it works."""
    if _supabase is None:
        return False
    try:
        _supabase.table("users").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# Cache the DB availability so we don't check every call
_db_available = None

def db_available():
    global _db_available
    if _db_available is None:
        key = (os.getenv("SUPABASE_KEY") or "").strip()
        # Publishable keys are for browser clients; server-side inserts often fail or stall. Use in-memory instead.
        if key.startswith("sb_publishable_"):
            print(
                "[database] In-memory mode: SUPABASE_KEY is a publishable (client) key. "
                "Use the service_role secret in .env for Supabase from the server."
            )
            _db_available = False
        else:
            _db_available = _use_db()
    return _db_available


# ── Public API ──────────────────────────────────────────────

def save_plan(user_data, calc_results, ai_roadmap):
    user_id = str(uuid.uuid4())
    plan_id = str(uuid.uuid4())

    if db_available():
        try:
            # get or create user
            email = user_data.email or f"{user_data.name.replace(' ','_').lower()}@fireapp.com"
            u = _supabase.table("users").upsert(
                {"name": user_data.name, "email": email, "preferred_language": user_data.language.value},
                on_conflict="email"
            ).execute()
            user_id = u.data[0]["id"]

            p = _supabase.table("fire_plans").insert({
                "user_id": user_id,
                "age": user_data.age,
                "monthly_income": user_data.monthly_income,
                "monthly_expenses": user_data.monthly_expenses,
                "current_savings": user_data.current_savings,
                "existing_investments": user_data.existing_investments,
                "fire_target_age": user_data.fire_target_age,
                "monthly_expenses_post_fire": user_data.monthly_expenses_post_fire,
                "risk_profile": user_data.risk_profile.value,
                "language": user_data.language.value
            }).execute()
            plan_id = p.data[0]["id"]

            _supabase.table("fire_results").insert({
                "plan_id": plan_id,
                "fire_corpus_min": calc_results["fire_corpus_range"]["min"],
                "fire_corpus_max": calc_results["fire_corpus_range"]["max"],
                "monthly_sip_min": calc_results["monthly_sip_range"]["min"],
                "monthly_sip_max": calc_results["monthly_sip_range"]["max"],
                "years_to_fire": calc_results["years_to_fire"],
                "monthly_savings": calc_results["monthly_savings"],
                "savings_rate_percent": calc_results["savings_rate_percent"],
                "savings_insight": calc_results["savings_insight"],
                "projected_existing_corpus": calc_results["projected_existing_corpus"],
                "emergency_fund_needed": calc_results["emergency_fund_needed"],
                "fire_type": calc_results["fire_type"],
                "asset_allocation": calc_results["asset_allocation"],
                "tax_suggestions": calc_results["tax_suggestions"],
                "insurance_advice": calc_results["insurance_advice"],
                "warnings": calc_results["warnings"],
                "ai_roadmap": ai_roadmap
            }).execute()
            return plan_id, user_id
        except Exception:
            pass  # fall through to in-memory

    # In-memory fallback
    _plans[plan_id] = {"user_id": user_id, "calc_results": calc_results, "ai_roadmap": ai_roadmap}
    return plan_id, user_id


def create_chat_session(user_id, language="english", plan_id=None, title="New Chat"):
    session_id = str(uuid.uuid4())

    if db_available():
        try:
            # user_id must be a valid UUID for Supabase
            try:
                uuid.UUID(str(user_id))
                uid = str(user_id)
            except ValueError:
                uid = str(uuid.uuid4())

            r = _supabase.table("chat_sessions").insert({
                "user_id": uid,
                "plan_id": plan_id,
                "language": language,
                "session_title": title
            }).execute()
            session_id = r.data[0]["id"]
            _messages[session_id] = []
            return session_id
        except Exception:
            pass

    # In-memory fallback
    _sessions[session_id] = {"user_id": user_id, "language": language, "plan_id": plan_id}
    _messages[session_id] = []
    return session_id


def save_message(session_id, role, content, language="english", context=None):
    if db_available():
        try:
            _supabase.table("chat_messages").insert({
                "session_id": session_id,
                "role": role,
                "content": content,
                "language": language,
                "retrieved_context": context
            }).execute()
            return
        except Exception:
            pass

    # In-memory fallback
    if session_id not in _messages:
        _messages[session_id] = []
    _messages[session_id].append({"role": role, "content": content})


def get_chat_history(session_id, limit=20):
    if db_available():
        try:
            r = _supabase.table("chat_messages")\
                .select("role, content")\
                .eq("session_id", session_id)\
                .order("created_at", desc=False)\
                .limit(limit)\
                .execute()
            return r.data
        except Exception:
            pass

    return _messages.get(session_id, [])[-limit:]


def update_session_activity(session_id):
    if db_available():
        try:
            from datetime import datetime, timezone
            _supabase.table("chat_sessions").update(
                {"last_active": datetime.now(timezone.utc).isoformat()}
            ).eq("id", session_id).execute()
        except Exception:
            pass


def get_user_sessions(user_id):
    if db_available():
        try:
            r = _supabase.table("chat_sessions")\
                .select("*").eq("user_id", user_id)\
                .order("last_active", desc=True).execute()
            return r.data
        except Exception:
            pass
    return []


def delete_user_data(user_id: str):
    """Delete all data for a user (GDPR right to erasure)."""
    if db_available():
        try:
            _supabase.table("users").delete().eq("id", user_id).execute()
            return
        except Exception:
            pass
    # In-memory fallback
    keys_to_delete = [k for k, v in _plans.items() if v.get("user_id") == user_id]
    for k in keys_to_delete:
        _plans.pop(k, None)
    _users.pop(user_id, None)
