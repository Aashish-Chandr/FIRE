"""
Input models with full validation and injection protection.
"""
import re
from pydantic import BaseModel, field_validator, Field
from typing import Optional
from enum import Enum


class Language(str, Enum):
    english  = "english"
    hindi    = "hindi"
    hinglish = "hinglish"


class RiskProfile(str, Enum):
    conservative = "conservative"
    moderate     = "moderate"
    aggressive   = "aggressive"


# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+instructions",
    r"system\s*prompt",
    r"you\s+are\s+now",
    r"act\s+as\s+",
    r"jailbreak",
    r"<\s*script",
    r"javascript:",
    r"data:text",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _check_injection(value: str) -> str:
    if _INJECTION_RE.search(value):
        raise ValueError("Input contains disallowed content.")
    return value


class FIREInput(BaseModel):
    name:                      str       = Field(..., min_length=1, max_length=100)
    email:                     Optional[str] = Field(None, max_length=200)
    age:                       int       = Field(..., ge=18, le=80)
    monthly_income:            float     = Field(..., gt=0, le=10_000_000)
    monthly_expenses:          float     = Field(..., gt=0, le=10_000_000)
    current_savings:           float     = Field(0, ge=0, le=1_000_000_000)
    existing_investments:      float     = Field(0, ge=0, le=1_000_000_000)
    fire_target_age:           int       = Field(..., ge=25, le=80)
    monthly_expenses_post_fire: float    = Field(..., gt=0, le=10_000_000)
    risk_profile:              RiskProfile = RiskProfile.moderate
    language:                  Language    = Language.english

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty.")
        return _check_injection(v)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if v and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email format.")
        return v


class StartChatRequest(BaseModel):
    user_id:      str                    = Field(..., min_length=1, max_length=200)
    language:     Language               = Language.english
    plan_id:      Optional[str]          = Field(None, max_length=200)
    fire_summary: Optional[str]          = Field(None, max_length=2000)


class ChatRequest(BaseModel):
    session_id: str     = Field(..., min_length=1, max_length=200)
    user_id:    str     = Field(..., min_length=1, max_length=200)
    message:    str     = Field(..., min_length=1, max_length=1000)
    language:   Language = Language.english

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty.")
        return _check_injection(v)
