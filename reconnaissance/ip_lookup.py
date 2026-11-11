"""IP address classification, reverse DNS, and passive RDAP lookup."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

import requests

from config.settings import RECON_TIMEOUT


def _reverse_dns(ip_str: str) -> str | None:
    try:
        hostname, _, _ = socket.gethostbyaddr(ip_str)
        return hostname
    except (socket.herror, socket.gaierror, OSError):
        return None


def _fetch_rdap_ip(ip_str: str, timeout: float = RECON_TIMEOUT) -> dict[str, Any]:
    """Query public RDAP network registration metadata for globally routable IPs."""
    try:
        resp = requests.get(
            f"https://rdap.org/ip/{ip_str}",
            headers={"Accept": "application/rdap+json, application/json"},
            timeout=timeout,
        )
        if resp.status_code == 404:
            return {"available": False, "note": "No public RDAP record found for this IP."}
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return {"available": False, "note": f"IP RDAP lookup unavailable: {exc}"}
    except ValueError:
        return {"available": False, "note": "IP RDAP returned invalid JSON."}

    return {
        "available": True,
        "network_name": data.get("name", "Unknown"),
        "handle": data.get("handle", "Unknown"),
        "country": data.get("country", "Unknown"),
        "start_address": data.get("startAddress", "Unknown"),
        "end_address": data.get("endAddress", "Unknown"),
        "ip_version": data.get("ipVersion", "Unknown"),
        "type": data.get("type", "Unknown"),
    }


def lookup_ip(ip_str: str, timeout: float = RECON_TIMEOUT) -> dict[str, Any]:
    """Analyze IP address classification, reverse DNS, and public RDAP metadata."""
    clean_ip = ip_str.strip()
    ip_obj = ipaddress.ip_address(clean_ip)

    classification = {
        "ip": str(ip_obj),
        "version": f"IPv{ip_obj.version}",
        "is_private": ip_obj.is_private,
        "is_loopback": ip_obj.is_loopback,
        "is_global": ip_obj.is_global,
        "is_multicast": ip_obj.is_multicast,
        "is_link_local": ip_obj.is_link_local,
        "is_reserved": ip_obj.is_reserved,
    }

    reverse_ptr = _reverse_dns(str(ip_obj))

    if ip_obj.is_global:
        rdap = _fetch_rdap_ip(str(ip_obj), timeout=timeout)
    else:
        rdap = {
            "available": False,
            "note": "Non-public/local IP address — external RDAP lookup skipped.",
        }

    notes: list[str] = []
    if ip_obj.is_loopback:
        notes.append("Loopback address (local machine only).")
    elif ip_obj.is_private:
        notes.append("Private RFC-reserved internal network address.")
    elif ip_obj.is_global:
        notes.append("Publicly routable IP address.")

    if reverse_ptr:
        notes.append(f"Reverse DNS (PTR) resolves to {reverse_ptr}.")
    else:
        notes.append("No reverse DNS (PTR) hostname resolved.")

    return {
        "ip": str(ip_obj),
        "classification": classification,
        "reverse_dns": reverse_ptr,
        "rdap": rdap,
        "notes": notes,
    }

