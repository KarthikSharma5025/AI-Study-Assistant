"""End-to-end workflow orchestration from input validation through reconnaissance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.input_validator import ValidationResult, validate_target
from config.settings import ANALYSIS_TASKS, DEFAULT_AUTHORIZED_SCOPES, RECON_TIMEOUT
from reconnaissance.dns_analysis import analyze_dns
from reconnaissance.domain_lookup import lookup_domain
from reconnaissance.ip_lookup import lookup_ip


@dataclass
class WorkflowResult:
    """Structured outcome from input validation and reconnaissance execution."""

    success: bool
    validation: ValidationResult
    task: str
    timestamp_utc: str
    recon_data: dict[str, Any] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    error_message: str = ""


def run_recon_step(
    raw_target: str,
    task: str = ANALYSIS_TASKS[0],
    authorized_scopes: list[str] | None = None,
    timeout: float = RECON_TIMEOUT,
) -> WorkflowResult:
    """Validate target & scope, then execute authorized preliminary reconnaissance."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    scopes = authorized_scopes if authorized_scopes is not None else DEFAULT_AUTHORIZED_SCOPES
    validation = validate_target(raw_target, authorized_scopes=scopes)

    if not validation.is_valid:
        return WorkflowResult(
            success=False,
            validation=validation,
            task=task,
            timestamp_utc=timestamp,
            error_message=validation.error_message,
        )

    target = validation.normalized_target
    target_type = validation.target_type
    recon_data: dict[str, Any] = {
        "target": target,
        "target_type": target_type,
        "matched_scope": validation.matched_scope,
        "task": task,
    }
    observations: list[str] = list(validation.notes)

    if target_type in {"ipv4", "ipv6"}:
        if task in {"Domain Information Lookup", "DNS Record & Security Analysis"}:
            observations.append(
                f"Target '{target}' is an IP address; executed IP & network lookup instead of domain-only lookups."
            )
        ip_info = lookup_ip(target, timeout=timeout)
        recon_data["ip_lookup"] = ip_info
        observations.extend(ip_info.get("notes", []))
    else:
        # Domain target
        if task in {"Full Preliminary Assessment", "Domain Information Lookup"}:
            domain_info = lookup_domain(target, include_tls=True, timeout=timeout)
            recon_data["domain_lookup"] = domain_info

            resolved_ips = domain_info.get("resolution", {}).get("ipv4_addresses", [])
            if task == "Full Preliminary Assessment" and resolved_ips:
                recon_data["ip_lookup"] = lookup_ip(resolved_ips[0], timeout=timeout)
                observations.extend(recon_data["ip_lookup"].get("notes", []))

        if task in {"Full Preliminary Assessment", "DNS Record & Security Analysis"}:
            dns_info = analyze_dns(target, timeout=timeout)
            recon_data["dns_analysis"] = dns_info
            observations.extend(dns_info.get("observations", []))

        if task == "IP Address & Network Lookup":
            domain_info = lookup_domain(target, include_tls=False, timeout=timeout)
            recon_data["domain_lookup"] = domain_info
            resolved_ips = domain_info.get("resolution", {}).get("ipv4_addresses", [])
            if resolved_ips:
                ip_info = lookup_ip(resolved_ips[0], timeout=timeout)
                recon_data["ip_lookup"] = ip_info
                observations.extend(ip_info.get("notes", []))
            else:
                observations.append(f"Could not resolve an IPv4 address for '{target}'.")

    return WorkflowResult(
        success=True,
        validation=validation,
        task=task,
        timestamp_utc=timestamp,
        recon_data=recon_data,
        observations=list(dict.fromkeys(observations)),
    )

