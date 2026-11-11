"""Passive domain information lookup using socket, RDAP, and TLS metadata."""

from __future__ import annotations

import socket
import ssl
from typing import Any

import requests

from config.settings import RECON_TIMEOUT


def _resolve_addresses(domain: str) -> dict[str, Any]:
    ipv4: list[str] = []
    ipv6: list[str] = []
    error: str | None = None

    try:
        infos = socket.getaddrinfo(domain, None)
        for family, _, _, _, sockaddr in infos:
            ip = sockaddr[0]
            if family == socket.AF_INET and ip not in ipv4:
                ipv4.append(ip)
            elif family == socket.AF_INET6 and ip not in ipv6:
                ipv6.append(ip)
    except socket.gaierror as exc:
        error = f"Hostname resolution failed: {exc}"

    return {
        "fqdn": socket.getfqdn(domain),
        "ipv4_addresses": ipv4,
        "ipv6_addresses": ipv6,
        "resolution_error": error,
    }


def _fetch_rdap_domain(domain: str, timeout: float = RECON_TIMEOUT) -> dict[str, Any]:
    """Query public RDAP metadata for domain registration details."""
    try:
        resp = requests.get(
            f"https://rdap.org/domain/{domain}",
            headers={"Accept": "application/rdap+json, application/json"},
            timeout=timeout,
        )
        if resp.status_code == 404:
            return {"available": False, "note": "No public RDAP record found for this domain."}
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return {"available": False, "note": f"RDAP lookup unavailable: {exc}"}
    except ValueError:
        return {"available": False, "note": "RDAP returned invalid JSON."}

    events = {
        ev.get("eventAction", ""): ev.get("eventDate", "")
        for ev in data.get("events", [])
        if isinstance(ev, dict)
    }
    nameservers = [
        ns.get("ldhName", "")
        for ns in data.get("nameservers", [])
        if isinstance(ns, dict) and ns.get("ldhName")
    ]

    registrar = None
    for entity in data.get("entities", []):
        if not isinstance(entity, dict):
            continue
        if "registrar" in entity.get("roles", []):
            vcard = entity.get("vcardArray", [])
            if isinstance(vcard, list) and len(vcard) > 1 and isinstance(vcard[1], list):
                for entry in vcard[1]:
                    if isinstance(entry, list) and len(entry) >= 4 and entry[0] == "fn":
                        registrar = str(entry[3])
                        break
            if not registrar:
                registrar = entity.get("handle")
            break

    return {
        "available": True,
        "ldh_name": data.get("ldhName", domain),
        "status": data.get("status", []),
        "registrar": registrar or "Not disclosed",
        "registered_at": events.get("registration", "Unknown"),
        "expires_at": events.get("expiration", "Unknown"),
        "last_changed": events.get("last changed", "Unknown"),
        "nameservers": nameservers,
    }


def _fetch_tls_metadata(domain: str, timeout: float = 3.5) -> dict[str, Any]:
    """Read public TLS certificate metadata on port 443 if HTTPS is listening."""
    ctx = ssl.create_default_context()
    try:
        with socket.create_connection((domain, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert() or {}
                protocol = ssock.version()
    except Exception as exc:
        return {"available": False, "note": f"TLS certificate unavailable on port 443 ({exc})"}

    subject = dict(item[0] for item in cert.get("subject", []) if item)
    issuer = dict(item[0] for item in cert.get("issuer", []) if item)
    sans = [val for kind, val in cert.get("subjectAltName", []) if kind == "DNS"]

    return {
        "available": True,
        "tls_version": protocol,
        "common_name": subject.get("commonName", "Unknown"),
        "issuer": issuer.get("organizationName") or issuer.get("commonName", "Unknown"),
        "not_before": cert.get("notBefore", "Unknown"),
        "not_after": cert.get("notAfter", "Unknown"),
        "san_count": len(sans),
        "sample_sans": sans[:10],
    }


def lookup_domain(
    domain: str,
    include_tls: bool = True,
    timeout: float = RECON_TIMEOUT,
) -> dict[str, Any]:
    """Collect preliminary passive information for an authorized domain."""
    clean_domain = domain.strip().lower().rstrip(".")
    resolution = _resolve_addresses(clean_domain)
    rdap = _fetch_rdap_domain(clean_domain, timeout=timeout)
    tls_info = (
        _fetch_tls_metadata(clean_domain, timeout=min(timeout, 3.5))
        if include_tls and clean_domain != "localhost"
        else {"available": False, "note": "TLS metadata check skipped."}
    )

    return {
        "domain": clean_domain,
        "resolution": resolution,
        "rdap": rdap,
        "tls": tls_info,
    }

