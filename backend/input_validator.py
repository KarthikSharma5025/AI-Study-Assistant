"""Input validation and authorized assessment scope checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import re

from config.settings import DEFAULT_AUTHORIZED_SCOPES


_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)


@dataclass
class ValidationResult:
    """Result of validating format and authorized assessment scope for a target."""

    is_valid: bool
    normalized_target: str
    target_type: str  # "domain", "ipv4", "ipv6", or "invalid"
    authorized: bool
    matched_scope: str | None = None
    error_message: str = ""
    notes: list[str] = field(default_factory=list)


def detect_target_type(target: str) -> tuple[str, str]:
    """Return (target_type, normalized_target) or ('invalid', reason)."""
    raw = target.strip()
    if not raw:
        return "invalid", "Please enter a domain name or IP address."

    if "://" in raw or "/" in raw or any(ch.isspace() for ch in raw):
        return (
            "invalid",
            "Enter only a clean domain or IP address (for example, `example.com` or `127.0.0.1`), without URL schemes or paths.",
        )

    # Check IPv4 / IPv6 first
    try:
        ip_obj = ipaddress.ip_address(raw)
        return ("ipv4" if ip_obj.version == 4 else "ipv6"), str(ip_obj)
    except ValueError:
        pass

    candidate = raw.lower().rstrip(".")
    if candidate == "localhost" or _DOMAIN_RE.match(candidate):
        return "domain", candidate

    return (
        "invalid",
        f"'{raw}' is not a valid domain name, IPv4, or IPv6 address.",
    )


def _match_ip_scope(ip_str: str, scope: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    try:
        if "/" in scope:
            net = ipaddress.ip_network(scope, strict=False)
            return ip_obj in net
        scope_ip = ipaddress.ip_address(scope)
        return ip_obj == scope_ip
    except ValueError:
        return False


def _match_domain_scope(domain: str, scope: str) -> bool:
    clean_scope = scope.strip().lower().rstrip(".")
    if not clean_scope:
        return False

    if clean_scope.startswith("*."):
        base = clean_scope[2:]
        return domain == base or domain.endswith(f".{base}")

    return domain == clean_scope or domain.endswith(f".{clean_scope}")


def check_authorized_scope(
    normalized_target: str,
    target_type: str,
    authorized_scopes: list[str] | None = None,
) -> tuple[bool, str | None]:
    """Check whether a normalized target falls within the authorized scope list."""
    scopes = authorized_scopes if authorized_scopes is not None else DEFAULT_AUTHORIZED_SCOPES

    for raw_scope in scopes:
        scope = raw_scope.strip()
        if not scope:
            continue
        if target_type in {"ipv4", "ipv6"} and _match_ip_scope(normalized_target, scope):
            return True, scope
        if target_type == "domain" and _match_domain_scope(normalized_target, scope):
            return True, scope

    return False, None


def validate_scope_entry(entry: str) -> tuple[bool, str]:
    """Validate a user-supplied scope entry (domain, wildcard domain, IP, or CIDR)."""
    clean = entry.strip().lower()
    if not clean:
        return False, "Scope entry cannot be empty."

    if "/" in clean:
        try:
            net = ipaddress.ip_network(clean, strict=False)
            return True, str(net)
        except ValueError:
            return False, f"Invalid CIDR network '{entry}'."

    if clean.startswith("*."):
        base = clean[2:]
        t_type, norm = detect_target_type(base)
        if t_type == "domain":
            return True, f"*.{norm}"
        return False, f"Invalid wildcard domain '{entry}'."

    t_type, norm_or_err = detect_target_type(clean)
    if t_type in {"domain", "ipv4", "ipv6"}:
        return True, norm_or_err
    return False, norm_or_err


def validate_target(
    raw_target: str,
    authorized_scopes: list[str] | None = None,
) -> ValidationResult:
    """Validate format and authorized assessment scope for a target domain or IP."""
    target_type, norm_or_err = detect_target_type(raw_target)
    if target_type == "invalid":
        return ValidationResult(
            is_valid=False,
            normalized_target="",
            target_type="invalid",
            authorized=False,
            error_message=norm_or_err,
        )

    authorized, matched_scope = check_authorized_scope(
        normalized_target=norm_or_err,
        target_type=target_type,
        authorized_scopes=authorized_scopes,
    )
    if not authorized:
        return ValidationResult(
            is_valid=False,
            normalized_target=norm_or_err,
            target_type=target_type,
            authorized=False,
            error_message=(
                f"Target '{norm_or_err}' is outside the authorized assessment scope. "
                "Add it to the Authorized Scope list in the sidebar only if you have permission to assess it."
            ),
        )

    return ValidationResult(
        is_valid=True,
        normalized_target=norm_or_err,
        target_type=target_type,
        authorized=True,
        matched_scope=matched_scope,
        notes=[f"Authorized via scope rule `{matched_scope}`."],
    )

