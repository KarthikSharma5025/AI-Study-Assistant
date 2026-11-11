"""Structured Markdown and JSON report builder for assessment findings."""

from __future__ import annotations

import json
from typing import Any


def build_markdown_report(
    target: str,
    target_type: str,
    task: str,
    timestamp_utc: str,
    model: str,
    recon_data: dict[str, Any],
    observations: list[str],
    ai_analysis: str = "",
    user_question: str = "",
) -> str:
    """Create a well-structured Markdown report from reconnaissance and AI outputs."""
    matched_scope = recon_data.get("matched_scope") or "Authorized"
    lines: list[str] = [
        f"# Security Study & Preliminary Reconnaissance Report: `{target}`",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| **Target** | `{target}` |",
        f"| **Target Type** | `{target_type}` |",
        f"| **Analysis Task** | {task} |",
        f"| **Authorized Scope** | `{matched_scope}` |",
        f"| **Local AI Model** | `{model}` |",
        f"| **Generated At** | {timestamp_utc} |",
        "",
    ]

    if user_question.strip():
        lines.extend(["## Student Prompt / Question", "", user_question.strip(), ""])

    if observations:
        lines.extend(["## Preliminary Security Observations", ""])
        for obs in observations:
            lines.append(f"- {obs}")
        lines.append("")

    domain_info = recon_data.get("domain_lookup")
    if isinstance(domain_info, dict):
        res = domain_info.get("resolution", {})
        rdap = domain_info.get("rdap", {})
        tls = domain_info.get("tls", {})
        lines.extend(["## Domain Information", ""])
        lines.append(f"- **FQDN:** `{res.get('fqdn', target)}`")
        lines.append(
            f"- **Resolved IPv4:** {', '.join(res.get('ipv4_addresses', [])) or 'None'}"
        )
        lines.append(
            f"- **Resolved IPv6:** {', '.join(res.get('ipv6_addresses', [])) or 'None'}"
        )
        if rdap.get("available"):
            lines.append(f"- **Registrar:** {rdap.get('registrar', 'Unknown')}")
            lines.append(f"- **Registered:** {rdap.get('registered_at', 'Unknown')}")
            lines.append(f"- **Expires:** {rdap.get('expires_at', 'Unknown')}")
        elif rdap.get("note"):
            lines.append(f"- **RDAP:** {rdap['note']}")
        if tls.get("available"):
            lines.append(
                f"- **TLS Certificate:** {tls.get('tls_version')} | Issuer: {tls.get('issuer')} | Expires: {tls.get('not_after')}"
            )
        lines.append("")

    dns_info = recon_data.get("dns_analysis")
    if isinstance(dns_info, dict):
        records = dns_info.get("records", {})
        checks = dns_info.get("security_checks", {})
        lines.extend(["## DNS Analysis", ""])
        lines.append(
            f"- **SPF Present:** {'Yes' if checks.get('spf_present') else 'No'} | "
            f"**DMARC Present:** {'Yes' if checks.get('dmarc_present') else 'No'} | "
            f"**CAA Present:** {'Yes' if checks.get('caa_present') else 'No'} | "
            f"**NS Count:** {checks.get('nameserver_count', 0)}"
        )
        lines.append("")
        lines.extend(["| Record Type | Values |", "| --- | --- |"])
        for rtype, values in records.items():
            joined = "; ".join(values) if values else "None"
            lines.append(f"| `{rtype}` | {joined.replace('|', '/')} |")
        lines.append("")

    ip_info = recon_data.get("ip_lookup")
    if isinstance(ip_info, dict):
        cls = ip_info.get("classification", {})
        rdap = ip_info.get("rdap", {})
        lines.extend(["## IP & Network Information", ""])
        lines.append(
            f"- **IP:** `{cls.get('ip', target)}` ({cls.get('version', 'Unknown')})"
        )
        lines.append(
            f"- **Flags:** Private={cls.get('is_private')}, Loopback={cls.get('is_loopback')}, Global={cls.get('is_global')}"
        )
        lines.append(f"- **Reverse DNS (PTR):** {ip_info.get('reverse_dns') or 'None'}")
        if rdap.get("available"):
            lines.append(
                f"- **RDAP Network:** {rdap.get('network_name')} ({rdap.get('country')}) [{rdap.get('start_address')} - {rdap.get('end_address')}]"
            )
        elif rdap.get("note"):
            lines.append(f"- **RDAP:** {rdap['note']}")
        lines.append("")

    if ai_analysis.strip():
        lines.extend(["## Local AI Analysis & Defensive Recommendations", "", ai_analysis.strip(), ""])

    lines.extend(
        [
            "---",
            "*Generated locally via AI Study Assistant (Ollama). For authorized security study and defensive analysis only.*",
        ]
    )
    return "\n".join(lines)


def build_json_report(
    target: str,
    target_type: str,
    task: str,
    timestamp_utc: str,
    model: str,
    recon_data: dict[str, Any],
    observations: list[str],
    ai_analysis: str = "",
    user_question: str = "",
) -> str:
    """Create a structured JSON report string."""
    payload = {
        "target": target,
        "target_type": target_type,
        "task": task,
        "timestamp_utc": timestamp_utc,
        "model": model,
        "user_question": user_question,
        "observations": observations,
        "reconnaissance": recon_data,
        "ai_analysis": ai_analysis,
    }
    return json.dumps(payload, indent=2, default=str)

