"""Pydantic Settings: load configuration from environment variables and/or YAML config file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


class Settings(BaseSettings):
    """Application-wide settings loaded from env vars (highest priority), then .env file."""

    # ── Neo4j ───────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── Redis ───────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Security ────────────────────────────────────────────────────────────
    secret_key: str = "changeme"

    # ── General ─────────────────────────────────────────────────────────────
    environment: str = "development"

    # ── NLP / Models ────────────────────────────────────────────────────────
    ner_model: str = "en_core_web_trf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gnn_layers: int = 2

    # ── API ─────────────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 100
    max_doc_size_mb: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def _load_yaml_overlay(profile: str = "default") -> dict[str, Any]:
    """Load a YAML config file and flatten its nested structure into Settings-compatible keys."""
    yaml_path = _CONFIG_DIR / f"{profile}.yaml"
    if not yaml_path.exists():
        return {}

    with open(yaml_path) as f:
        raw: dict = yaml.safe_load(f) or {}

    flat: dict[str, Any] = {}
    # Flatten nested YAML: e.g. neo4j.uri → neo4j_uri
    _key_map = {
        ("neo4j", "uri"): "neo4j_uri",
        ("neo4j", "user"): "neo4j_user",
        ("neo4j", "password"): "neo4j_password",
        ("redis", "url"): "redis_url",
        ("api", "rate_limit"): "rate_limit_per_minute",
        ("api", "max_doc_size_mb"): "max_doc_size_mb",
        ("nlp", "ner_model"): "ner_model",
        ("nlp", "embedding_model"): "embedding_model",
        ("nlp", "gnn_layers"): "gnn_layers",
    }
    for (section, key), settings_key in _key_map.items():
        if section in raw and key in raw[section]:
            flat[settings_key] = raw[section][key]

    return flat


@lru_cache(maxsize=1)
def get_settings(profile: str = "default") -> Settings:
    """Return a cached Settings instance.  YAML values are used as defaults
    that environment variables and .env can override."""
    yaml_defaults = _load_yaml_overlay(profile)
    return Settings(**yaml_defaults)
