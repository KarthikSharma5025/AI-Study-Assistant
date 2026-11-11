"""DNS record analysis and defensive posture checks using dnspython."""

from __future__ import annotations

from typing import Any

import dns.exception
import dns.resolver

from config.settings import RECON_TIMEOUT


RECORD_TYPES: tuple[str, ...] = ("A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA", "CAA")


def _query_record(
    resolver: dns.resolver.Resolver,
    name: str,
    rdtype: str,
) -> list[str]:
    try:
        answers = resolver.resolve(name, rdtype)
        return [rdata.to_text().strip() for rdata in answers]
    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
        dns.resolver.NoNameservers,
        dns.exception.Timeout,
        dns.exception.DNSException,
    ):
        return []


def analyze_dns(domain: str, timeout: float = RECON_TIMEOUT) -> dict[str, Any]:
    """Query common DNS records and evaluate basic defensive email/DNS posture."""
    clean_domain = domain.strip().lower().rstrip(".")
    resolver = dns.resolver.Resolver()
    resolver.timeout = timeout
    resolver.lifetime = timeout

    records: dict[str, list[str]] = {}
    for rdtype in RECORD_TYPES:
        records[rdtype] = _query_record(resolver, clean_domain, rdtype)

    dmarc_records = _query_record(resolver, f"_dmarc.{clean_domain}", "TXT")
    records["DMARC"] = dmarc_records

    txt_records = records.get("TXT", [])
    spf_records = [txt for txt in txt_records if "v=spf1" in txt.lower()]
    has_spf = len(spf_records) > 0
    has_dmarc = any("v=dmarc1" in rec.lower() for rec in dmarc_records)
    has_caa = len(records.get("CAA", [])) > 0
    ns_count = len(records.get("NS", []))

    observations: list[str] = []
    if not has_spf:
        observations.append("No SPF TXT record detected (email spoofing protection may be incomplete).")
    if not has_dmarc:
        observations.append("No DMARC record detected at _dmarc subdomain.")
    if not has_caa:
        observations.append("No CAA record detected (any public CA may issue certificates unless restricted).")
    if ns_count == 1:
        observations.append("Only one authoritative nameserver (NS) record found; redundancy is limited.")

    if not observations and any(records.values()):
        observations.append("Baseline DNS security records (SPF, DMARC, CAA) and redundant NS records are present.")
    elif not any(records.values()):
        observations.append("No public DNS records resolved for this target.")

    return {
        "domain": clean_domain,
        "records": records,
        "security_checks": {
            "spf_present": has_spf,
            "spf_records": spf_records,
            "dmarc_present": has_dmarc,
            "caa_present": has_caa,
            "nameserver_count": ns_count,
        },
        "observations": observations,
    }

