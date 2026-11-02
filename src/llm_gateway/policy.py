"""Gateway policy: which providers and models a product may use, how outbound text is treated,
who may call, and what models cost. One YAML file per deployment (a ConfigMap), so each product
runs the gateway with its own data class and budgets."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DataClass = Literal["general", "payments", "phi"]
RedactionMode = Literal["off", "redact", "block"]


class Provider(BaseModel):
    type: Literal["anthropic", "openai"]
    base_url: str
    api_key_env: str
    baa: bool = False  # a business associate agreement (or DPA) covers this provider
    models: list[str] = Field(default_factory=lambda: ["*"])

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    def serves(self, model: str) -> bool:
        return any(fnmatch.fnmatch(model, p) for p in self.models)


class DataClassRule(BaseModel):
    require_baa: bool = False
    redaction: RedactionMode = "redact"
    detectors: list[str] = Field(
        default_factory=lambda: ["secret", "card_number", "us_ssn", "email", "phone"]
    )


class Caller(BaseModel):
    key_sha256: str
    budget_usd_per_day: float = 25.0
    models: list[str] = Field(default_factory=lambda: ["*"])
    max_tokens: int = 8192

    def may_use(self, model: str) -> bool:
        return any(fnmatch.fnmatch(model, p) for p in self.models)


class Price(BaseModel):
    """US dollars per million tokens."""

    input: float
    output: float
    cache_write: float | None = None  # defaults to 1.25 x input (5-minute cache writes)
    cache_read: float | None = None  # defaults to 0.1 x input


class Policy(BaseModel):
    product: str
    data_class: DataClass
    providers: dict[str, Provider]
    data_classes: dict[str, DataClassRule] = Field(default_factory=dict)
    callers: dict[str, Caller]
    pricing: dict[str, Price] = Field(default_factory=dict)  # model glob -> price

    @field_validator("callers")
    @classmethod
    def _hex_keys(cls, v: dict[str, Caller]) -> dict[str, Caller]:
        for name, c in v.items():
            if len(c.key_sha256) != 64:
                raise ValueError(f"caller {name}: key_sha256 must be a hex SHA-256 of its key")
        return v

    def rule(self) -> DataClassRule:
        return self.data_classes.get(self.data_class, DataClassRule())

    def caller_for(self, key: str) -> tuple[str, Caller] | None:
        digest = hashlib.sha256(key.encode()).hexdigest()
        for name, c in self.callers.items():
            if c.key_sha256 == digest:
                return name, c
        return None

    def provider_for(self, model: str) -> tuple[str, Provider] | None:
        """The first provider that serves the model and is allowed for this data class."""
        rule = self.rule()
        for name, p in self.providers.items():
            if p.serves(model) and (p.baa or not rule.require_baa):
                return name, p
        return None

    def price(self, model: str) -> Price | None:
        for pattern, price in self.pricing.items():
            if fnmatch.fnmatch(model, pattern):
                return price
        return None


def load_policy(path: str | Path) -> Policy:
    return Policy.model_validate(yaml.safe_load(Path(path).read_text()))
