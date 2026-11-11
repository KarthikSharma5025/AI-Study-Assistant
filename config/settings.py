"""Application configuration and default authorization scope settings."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "generated"

APP_TITLE = "AI STUDY ASSISTANT"
APP_SUBTITLE = "Local AI-Powered Security Study & Reconnaissance Assistant"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

RECON_TIMEOUT = 5.0

# Safe educational / lab targets pre-authorized by default.
# Users can add their own authorized domains, IPs, or CIDRs in the UI.
DEFAULT_AUTHORIZED_SCOPES: list[str] = [
    "example.com",
    "example.org",
    "example.net",
    "localhost",
    "127.0.0.1",
    "::1",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
]

ANALYSIS_TASKS: list[str] = [
    "Full Preliminary Assessment",
    "Domain Information Lookup",
    "DNS Record & Security Analysis",
    "IP Address & Network Lookup",
]

