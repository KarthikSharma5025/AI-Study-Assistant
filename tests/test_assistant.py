"""Unit and Streamlit AppTest suite for the AI Security Study & Reconnaissance Assistant."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_engine import OllamaEngine, OllamaError
from backend.input_validator import detect_target_type, validate_scope_entry, validate_target
from backend.workflow import run_recon_step
from reconnaissance.dns_analysis import analyze_dns
from reconnaissance.domain_lookup import lookup_domain
from reconnaissance.ip_lookup import lookup_ip
from reports.report_generator import build_json_report, build_markdown_report


# ---------------------------------------------------------------------------
# Input validation & scope checks
# ---------------------------------------------------------------------------

def test_target_detection_and_scope_validation() -> None:
    assert detect_target_type("Example.COM.") == ("domain", "example.com")
    assert detect_target_type("192.168.1.10") == ("ipv4", "192.168.1.10")
    assert detect_target_type("::1") == ("ipv6", "::1")
    assert detect_target_type("https://example.com/login")[0] == "invalid"
    assert detect_target_type("not a valid host")[0] == "invalid"

    ok_domain = validate_target("sub.example.com")
    assert ok_domain.is_valid and ok_domain.authorized

    ok_cidr = validate_target("192.168.10.25")
    assert ok_cidr.is_valid and ok_cidr.authorized

    blocked = validate_target("unauthorized-domain.org")
    assert not blocked.is_valid
    assert not blocked.authorized
    assert "outside the authorized assessment scope" in blocked.error_message

    assert validate_scope_entry("*.corp.local") == (True, "*.corp.local")
    assert validate_scope_entry("10.20.0.0/16") == (True, "10.20.0.0/16")
    assert validate_scope_entry("http://bad.com")[0] is False


# ---------------------------------------------------------------------------
# Reconnaissance modules
# ---------------------------------------------------------------------------

def test_ip_lookup_private_and_public() -> None:
    local_res = lookup_ip("127.0.0.1")
    assert local_res["classification"]["is_loopback"] is True
    assert local_res["rdap"]["available"] is False

    mock_rdap = MagicMock()
    mock_rdap.status_code = 200
    mock_rdap.raise_for_status.return_value = None
    mock_rdap.json.return_value = {
        "name": "EXAMPLE-NET",
        "handle": "NET-93-184-216-0-1",
        "country": "US",
        "startAddress": "93.184.216.0",
        "endAddress": "93.184.216.255",
        "ipVersion": "v4",
        "type": "DIRECT ALLOCATION",
    }
    with (
        patch("reconnaissance.ip_lookup.requests.get", return_value=mock_rdap),
        patch("reconnaissance.ip_lookup.socket.gethostbyaddr", return_value=("example.com", [], ["93.184.216.34"])),
    ):
        pub_res = lookup_ip("93.184.216.34")

    assert pub_res["classification"]["is_global"] is True
    assert pub_res["reverse_dns"] == "example.com"
    assert pub_res["rdap"]["network_name"] == "EXAMPLE-NET"


def test_dns_analysis_posture_checks() -> None:
    def _fake_resolve(name: str, rdtype: str):
        item = MagicMock()
        if rdtype == "A":
            item.to_text.return_value = "93.184.216.34"
            return [item]
        if rdtype == "NS":
            n1, n2 = MagicMock(), MagicMock()
            n1.to_text.return_value = "ns1.example.com."
            n2.to_text.return_value = "ns2.example.com."
            return [n1, n2]
        if rdtype == "TXT" and not name.startswith("_dmarc."):
            item.to_text.return_value = '"v=spf1 -all"'
            return [item]
        if rdtype == "TXT" and name.startswith("_dmarc."):
            item.to_text.return_value = '"v=DMARC1; p=reject;"'
            return [item]
        import dns.resolver

        raise dns.resolver.NoAnswer()

    mock_resolver = MagicMock()
    mock_resolver.resolve.side_effect = _fake_resolve

    with patch("reconnaissance.dns_analysis.dns.resolver.Resolver", return_value=mock_resolver):
        res = analyze_dns("example.com")

    assert res["security_checks"]["spf_present"] is True
    assert res["security_checks"]["dmarc_present"] is True
    assert res["security_checks"]["caa_present"] is False
    assert res["security_checks"]["nameserver_count"] == 2
    assert any("No CAA record" in obs for obs in res["observations"])


def test_domain_lookup_mocked() -> None:
    mock_rdap = MagicMock()
    mock_rdap.status_code = 200
    mock_rdap.raise_for_status.return_value = None
    mock_rdap.json.return_value = {
        "ldhName": "example.com",
        "status": ["client transfer prohibited"],
        "events": [{"eventAction": "registration", "eventDate": "1995-08-14"}],
        "nameservers": [{"ldhName": "a.iana-servers.net"}],
        "entities": [
            {
                "roles": ["registrar"],
                "vcardArray": ["vcard", [["fn", {}, "text", "RESERVED-Internet Assigned Numbers Authority"]]],
            }
        ],
    }

    with (
        patch(
            "reconnaissance.domain_lookup.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("93.184.216.34", 0))],
        ),
        patch("reconnaissance.domain_lookup.socket.getfqdn", return_value="example.com"),
        patch("reconnaissance.domain_lookup.requests.get", return_value=mock_rdap),
    ):
        res = lookup_domain("example.com", include_tls=False)

    assert res["resolution"]["ipv4_addresses"] == ["93.184.216.34"]
    assert res["rdap"]["available"] is True
    assert "Internet Assigned Numbers Authority" in res["rdap"]["registrar"]


# ---------------------------------------------------------------------------
# Workflow, AI Engine & Reports
# ---------------------------------------------------------------------------

def test_workflow_blocks_unauthorized_and_runs_authorized() -> None:
    blocked = run_recon_step("evil-external-site.com", task="DNS Record & Security Analysis")
    assert blocked.success is False
    assert "outside the authorized assessment scope" in blocked.error_message

    allowed = run_recon_step("127.0.0.1", task="IP Address & Network Lookup")
    assert allowed.success is True
    assert allowed.recon_data["ip_lookup"]["classification"]["is_loopback"] is True


def test_ollama_engine_cpu_fallback_and_errors() -> None:
    import requests as req_lib

    engine = OllamaEngine(default_model="llama3:8b", auto_cpu_fallback=True)

    cuda_resp = MagicMock()
    cuda_resp.status_code = 500
    cuda_resp.json.return_value = {"error": "CUDA error: shared object initialization failed"}

    first_mock = MagicMock()
    first_mock.raise_for_status.side_effect = req_lib.HTTPError("500", response=cuda_resp)

    second_mock = MagicMock()
    second_mock.raise_for_status.return_value = None
    second_mock.json.return_value = {"message": {"content": "Analysis complete"}}

    with patch("backend.ai_engine.requests.post", side_effect=[first_mock, second_mock]) as post_mock:
        reply = engine.chat([{"role": "user", "content": "Summarize"}])

    assert reply == "Analysis complete"
    assert engine.using_cpu_fallback is True
    assert post_mock.call_args_list[1].kwargs["json"]["options"]["num_gpu"] == 0

    with patch("backend.ai_engine.requests.post", side_effect=req_lib.ConnectionError("refused")):
        with pytest.raises(OllamaError, match="Could not connect to Ollama"):
            engine.chat([{"role": "user", "content": "Hi"}])


def test_report_generators() -> None:
    recon_data = {
        "matched_scope": "127.0.0.1",
        "ip_lookup": {
            "classification": {"ip": "127.0.0.1", "version": "IPv4", "is_private": True, "is_loopback": True, "is_global": False},
            "reverse_dns": "localhost",
            "rdap": {"available": False, "note": "Local IP skipped."},
        },
    }
    md = build_markdown_report(
        target="127.0.0.1",
        target_type="ipv4",
        task="IP Address & Network Lookup",
        timestamp_utc="2026-11-12 02:00:00 UTC",
        model="llama3:8b",
        recon_data=recon_data,
        observations=["Loopback address."],
        ai_analysis="This is a local loopback IP.",
        user_question="What is 127.0.0.1?",
    )
    assert "# Security Study & Preliminary Reconnaissance Report: `127.0.0.1`" in md
    assert "This is a local loopback IP." in md

    raw_json = build_json_report(
        target="127.0.0.1",
        target_type="ipv4",
        task="IP Address & Network Lookup",
        timestamp_utc="2026-11-12 02:00:00 UTC",
        model="llama3:8b",
        recon_data=recon_data,
        observations=["Loopback address."],
        ai_analysis="This is a local loopback IP.",
    )
    parsed = json.loads(raw_json)
    assert parsed["target"] == "127.0.0.1"
    assert parsed["ai_analysis"] == "This is a local loopback IP."


# ---------------------------------------------------------------------------
# Streamlit AppTest end-to-end UI flow
# ---------------------------------------------------------------------------

def test_streamlit_ui_validation_recon_and_qa() -> None:
    mock_engine = MagicMock(spec=OllamaEngine)
    mock_engine.check_connection.return_value = (True, "Connected (3 model(s) available)")
    mock_engine.list_models.return_value = ["llama3:8b", "phi3:latest"]
    mock_engine.using_cpu_fallback = False
    mock_engine.stream_analyze_recon_findings.side_effect = lambda **kw: iter(
        ["### Plain-English Summary\n", "127.0.0.1 is the local loopback address."]
    )
    mock_engine.stream_study_qa.side_effect = lambda **kw: iter(
        ["Loopback traffic never leaves the host network stack."]
    )

    with patch("frontend.interface.OllamaEngine", return_value=mock_engine):
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15)
        at.run()
        assert not at.exception

        # 1. Test scope validation error on unauthorized target
        target_box = next(t for t in at.text_input if t.key == "target_input")
        target_box.set_value("unlisted-example.org")
        run_btn = next(b for b in at.button if b.label == "Run Authorized Analysis")
        run_btn.click().run()
        assert not at.exception
        assert any("Validation Failed" in err.value for err in at.error)

        # 2. Test authorized target 127.0.0.1 completes recon + AI analysis + report
        target_box = next(t for t in at.text_input if t.key == "target_input")
        target_box.set_value("127.0.0.1")
        run_btn = next(b for b in at.button if b.label == "Run Authorized Analysis")
        run_btn.click().run()
        assert not at.exception
        assert at.session_state["latest_run"]["target"] == "127.0.0.1"
        assert "loopback address" in at.session_state["latest_run"]["ai_analysis"]
        assert len(at.session_state["saved_reports"]) == 1

        # 3. Test Security Study Q&A page
        qa_nav = next(b for b in at.sidebar.button if b.key == "nav_Security Study Q&A")
        qa_nav.click().run()
        assert not at.exception
        at.chat_input[0].set_value("Why is 127.0.0.1 safe for local testing?").run()
        assert not at.exception
        assert len(at.session_state["study_chat_messages"]) == 2
        assert "network stack" in at.session_state["study_chat_messages"][1]["content"]

        # 4. Test Reports History page
        rep_nav = next(b for b in at.sidebar.button if b.key == "nav_Reports History")
        rep_nav.click().run()
        assert not at.exception
        assert any("127.0.0.1" in md.value for md in at.markdown)

