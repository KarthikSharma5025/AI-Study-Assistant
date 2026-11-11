"""Streamlit user interface for the AI Security Study & Reconnaissance Assistant."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from backend.ai_engine import OllamaEngine, OllamaError
from backend.input_validator import validate_scope_entry
from backend.workflow import run_recon_step
from config.settings import (
    ANALYSIS_TASKS,
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_AUTHORIZED_SCOPES,
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
)
from reports.report_generator import build_json_report, build_markdown_report


PAGES = [
    "Recon & Study Analysis",
    "Security Study Q&A",
    "Reports History",
]

COMPUTE_MODES = [
    "Auto (GPU w/ CPU fallback)",
    "CPU Only",
    "GPU Default",
]

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    max-width: 1180px;
}
.asa-header {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(16, 185, 129, 0.10));
    border: 1px solid rgba(59, 130, 246, 0.30);
    border-radius: 14px;
    padding: 1.05rem 1.4rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}
.asa-title {
    font-size: 1.45rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    margin: 0;
}
.asa-subtitle {
    font-size: 0.9rem;
    opacity: 0.82;
    margin: 0.2rem 0 0 0;
}
.asa-pills {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.asa-pill {
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.28rem 0.72rem;
    border-radius: 999px;
    border: 1px solid rgba(128, 128, 128, 0.28);
    background: rgba(128, 128, 128, 0.08);
}
.asa-pill.online {
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
}
.asa-pill.offline {
    border-color: rgba(239, 68, 68, 0.45);
    background: rgba(239, 68, 68, 0.12);
    color: #ef4444;
}
.workflow-strip {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.workflow-step {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.28rem 0.65rem;
    border-radius: 8px;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.22);
}
.sidebar-heading {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    opacity: 0.7;
    margin: 0.35rem 0 0.45rem 0;
}
section[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start;
    text-align: left;
    border-radius: 8px;
}
</style>
"""


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "current_page": PAGES[0],
        "ollama_url": OLLAMA_BASE_URL,
        "selected_model": DEFAULT_MODEL,
        "temperature": 0.4,
        "compute_mode": COMPUTE_MODES[0],
        "cpu_fallback_active": False,
        "conn_status": None,
        "available_models": None,
        "authorized_scopes": list(DEFAULT_AUTHORIZED_SCOPES),
        "latest_run": None,
        "followup_messages": [],
        "study_chat_messages": [],
        "saved_reports": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _build_engine() -> OllamaEngine:
    mode = st.session_state.compute_mode
    if mode == "CPU Only":
        num_gpu: int | None = 0
        auto_fallback = False
    elif mode == "GPU Default":
        num_gpu = None
        auto_fallback = False
    else:
        num_gpu = None
        auto_fallback = True

    engine = OllamaEngine(
        base_url=st.session_state.ollama_url,
        default_model=st.session_state.selected_model,
        num_gpu=num_gpu,
        auto_cpu_fallback=auto_fallback,
    )
    if mode == COMPUTE_MODES[0] and st.session_state.cpu_fallback_active:
        engine.using_cpu_fallback = True
    return engine


def _refresh_status(engine: OllamaEngine) -> tuple[bool, str, list[str]]:
    ok, msg = engine.check_connection()
    models = engine.list_models()
    st.session_state.conn_status = (ok, msg)
    st.session_state.available_models = models
    return ok, msg, models


def _get_status(engine: OllamaEngine) -> tuple[bool, str, list[str]]:
    if st.session_state.conn_status is None or st.session_state.available_models is None:
        return _refresh_status(engine)
    ok, msg = st.session_state.conn_status
    return ok, msg, st.session_state.available_models


def _render_header(conn_ok: bool) -> None:
    safe_model = html.escape(st.session_state.selected_model)
    status_cls = "online" if conn_ok else "offline"
    status_txt = "Ollama Online" if conn_ok else "Ollama Offline"
    st.markdown(
        f"""
        <div class="asa-header">
            <div>
                <div class="asa-title">{html.escape(APP_TITLE)}</div>
                <p class="asa-subtitle">{html.escape(APP_SUBTITLE)}</p>
            </div>
            <div class="asa-pills">
                <span class="asa-pill">Model: {safe_model}</span>
                <span class="asa-pill {status_cls}">{status_txt}</span>
                <span class="asa-pill">Local &amp; Authorized Mode</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(engine: OllamaEngine, conn_ok: bool, conn_msg: str, models: list[str]) -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-heading">Navigation</div>', unsafe_allow_html=True)
        for page in PAGES:
            active = st.session_state.current_page == page
            label = f"> {page}" if active else page
            if st.button(
                label,
                key=f"nav_{page}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.current_page = page
                st.rerun()

        st.divider()
        st.markdown('<div class="sidebar-heading">Ollama Settings</div>', unsafe_allow_html=True)

        if st.session_state.selected_model not in models:
            models = [st.session_state.selected_model] + models

        st.session_state.selected_model = st.selectbox(
            "Model",
            options=models,
            index=models.index(st.session_state.selected_model),
        )
        if conn_ok:
            st.caption(f"Status: {conn_msg}")
        else:
            st.warning(conn_msg)

        if st.button("Refresh Ollama", use_container_width=True):
            _refresh_status(engine)
            st.rerun()

        with st.expander("Model Preferences", expanded=False):
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.temperature),
                step=0.05,
            )
            st.session_state.compute_mode = st.selectbox(
                "Compute Mode",
                options=COMPUTE_MODES,
                index=COMPUTE_MODES.index(st.session_state.compute_mode),
            )
            new_url = st.text_input("Ollama Base URL", value=st.session_state.ollama_url).strip()
            if new_url and new_url != st.session_state.ollama_url:
                st.session_state.ollama_url = new_url
                st.session_state.conn_status = None
                st.session_state.available_models = None

        st.divider()
        st.markdown('<div class="sidebar-heading">Authorized Scope</div>', unsafe_allow_html=True)
        st.caption("Only targets matching this scope list are permitted for reconnaissance.")

        new_scope = st.text_input(
            "Add Domain, IP, or CIDR",
            placeholder="e.g., scanme.nmap.org or 192.168.1.0/24",
            key="new_scope_input",
        )
        if st.button("Add to Authorized Scope", use_container_width=True):
            valid_entry, msg_or_norm = validate_scope_entry(new_scope)
            if not valid_entry:
                st.error(msg_or_norm)
            elif msg_or_norm in st.session_state.authorized_scopes:
                st.info(f"`{msg_or_norm}` is already authorized.")
            else:
                st.session_state.authorized_scopes.append(msg_or_norm)
                st.success(f"Added `{msg_or_norm}` to authorized scope.")
                st.rerun()

        with st.expander(f"Active Scope Rules ({len(st.session_state.authorized_scopes)})", expanded=False):
            for idx, rule in enumerate(list(st.session_state.authorized_scopes)):
                c1, c2 = st.columns([3.2, 1])
                c1.code(rule, language=None)
                if c2.button("X", key=f"rm_scope_{idx}", help=f"Remove {rule}"):
                    st.session_state.authorized_scopes.pop(idx)
                    st.rerun()
            if st.button("Reset Defaults", use_container_width=True):
                st.session_state.authorized_scopes = list(DEFAULT_AUTHORIZED_SCOPES)
                st.rerun()


def _render_recon_findings(run: dict[str, Any]) -> None:
    recon = run["recon_data"]
    observations = run["observations"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target", run["target"])
    m2.metric("Target Type", run["target_type"].upper())
    m3.metric("Matched Scope", recon.get("matched_scope") or "-")
    m4.metric("Task", run["task"].split()[0])

    if observations:
        st.markdown("#### Preliminary Security Observations")
        for obs in observations:
            st.markdown(f"- {obs}")

    domain_info = recon.get("domain_lookup")
    if isinstance(domain_info, dict):
        with st.expander("Domain Lookup Details", expanded=True):
            res = domain_info.get("resolution", {})
            rdap = domain_info.get("rdap", {})
            tls = domain_info.get("tls", {})

            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Hostname Resolution**")
                st.write(f"- **FQDN:** `{res.get('fqdn', run['target'])}`")
                st.write(f"- **IPv4:** {', '.join(res.get('ipv4_addresses', [])) or 'None'}")
                st.write(f"- **IPv6:** {', '.join(res.get('ipv6_addresses', [])) or 'None'}")
                if res.get("resolution_error"):
                    st.caption(res["resolution_error"])
            with d2:
                st.markdown("**Registration (RDAP) & TLS**")
                if rdap.get("available"):
                    st.write(f"- **Registrar:** {rdap.get('registrar')}")
                    st.write(f"- **Registered:** {rdap.get('registered_at')}")
                    st.write(f"- **Expires:** {rdap.get('expires_at')}")
                else:
                    st.write(f"- **RDAP:** {rdap.get('note', 'Unavailable')}")
                if tls.get("available"):
                    st.write(
                        f"- **TLS:** {tls.get('tls_version')} | Issuer: {tls.get('issuer')} | Expires: {tls.get('not_after')}"
                    )
                else:
                    st.write(f"- **TLS:** {tls.get('note', 'Unavailable')}")

    dns_info = recon.get("dns_analysis")
    if isinstance(dns_info, dict):
        with st.expander("DNS Records & Security Checks", expanded=True):
            checks = dns_info.get("security_checks", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SPF", "Present" if checks.get("spf_present") else "Missing")
            c2.metric("DMARC", "Present" if checks.get("dmarc_present") else "Missing")
            c3.metric("CAA", "Present" if checks.get("caa_present") else "Missing")
            c4.metric("Nameservers", checks.get("nameserver_count", 0))

            records = dns_info.get("records", {})
            rows = [
                {"Record Type": rtype, "Values": ", ".join(vals) if vals else "—"}
                for rtype, vals in records.items()
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    ip_info = recon.get("ip_lookup")
    if isinstance(ip_info, dict):
        with st.expander("IP Address & Network Details", expanded=True):
            cls = ip_info.get("classification", {})
            rdap = ip_info.get("rdap", {})
            i1, i2 = st.columns(2)
            with i1:
                st.markdown("**Classification**")
                st.write(f"- **IP:** `{cls.get('ip')}` ({cls.get('version')})")
                st.write(f"- **Global / Public:** {cls.get('is_global')}")
                st.write(f"- **Private:** {cls.get('is_private')} | **Loopback:** {cls.get('is_loopback')}")
                st.write(f"- **Reverse DNS (PTR):** {ip_info.get('reverse_dns') or 'None'}")
            with i2:
                st.markdown("**Network Allocation (RDAP)**")
                if rdap.get("available"):
                    st.write(f"- **Network Name:** {rdap.get('network_name')}")
                    st.write(f"- **Country:** {rdap.get('country')}")
                    st.write(f"- **Range:** {rdap.get('start_address')} – {rdap.get('end_address')}")
                    st.write(f"- **Handle:** `{rdap.get('handle')}`")
                else:
                    st.write(rdap.get("note", "Unavailable"))


def _render_recon_workflow(engine: OllamaEngine) -> None:
    st.markdown(
        """
        <div class="workflow-strip">
            <span class="workflow-step">1. Enter Domain / IP</span>
            <span class="workflow-step">2. Validate &amp; Scope Check</span>
            <span class="workflow-step">3. Preliminary Recon</span>
            <span class="workflow-step">4. Local Ollama Analysis</span>
            <span class="workflow-step">5. Findings &amp; Advice</span>
            <span class="workflow-step">6. Export Report</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_target, col_task = st.columns([1.4, 1.2])
    with col_target:
        raw_target = st.text_input(
            "Target Domain Name or IP Address",
            placeholder="e.g., example.com, 127.0.0.1, 192.168.1.1",
            key="target_input",
        )
    with col_task:
        selected_task = st.selectbox(
            "Analysis Task",
            options=ANALYSIS_TASKS,
            index=0,
            key="analysis_task",
        )

    user_question = st.text_area(
        "Study Question or Analysis Prompt (Optional)",
        placeholder="e.g., Explain what these DNS records reveal about email spoofing defenses and hosting.",
        height=90,
        key="recon_question_input",
    )

    run_clicked = st.button("Run Authorized Analysis", type="primary")

    just_streamed_ai = False
    if run_clicked:
        with st.spinner("Validating target scope and gathering preliminary information..."):
            wf_result = run_recon_step(
                raw_target=raw_target,
                task=selected_task,
                authorized_scopes=st.session_state.authorized_scopes,
            )

        if not wf_result.success:
            st.error(f"Validation Failed: {wf_result.error_message}")
            return

        run_record: dict[str, Any] = {
            "target": wf_result.validation.normalized_target,
            "target_type": wf_result.validation.target_type,
            "task": wf_result.task,
            "timestamp_utc": wf_result.timestamp_utc,
            "model": st.session_state.selected_model,
            "user_question": user_question.strip(),
            "observations": wf_result.observations,
            "recon_data": wf_result.recon_data,
            "ai_analysis": "",
            "markdown_report": "",
            "json_report": "",
        }

        st.success(
            f"Authorized target `{run_record['target']}` verified "
            f"(scope rule `{wf_result.validation.matched_scope}`). Reconnaissance complete."
        )

        st.markdown("### Local AI Interpretation & Recommendations")
        try:
            stream = engine.stream_analyze_recon_findings(
                target=run_record["target"],
                target_type=run_record["target_type"],
                task_name=run_record["task"],
                recon_data=run_record["recon_data"],
                user_question=run_record["user_question"],
                model=st.session_state.selected_model,
                temperature=float(st.session_state.temperature),
            )
            ai_text = str(st.write_stream(stream)).strip()
            if engine.using_cpu_fallback:
                st.session_state.cpu_fallback_active = True
            run_record["ai_analysis"] = ai_text
            just_streamed_ai = True
        except OllamaError as exc:
            st.error(f"Ollama Analysis Error: {exc}")

        run_record["markdown_report"] = build_markdown_report(
            target=run_record["target"],
            target_type=run_record["target_type"],
            task=run_record["task"],
            timestamp_utc=run_record["timestamp_utc"],
            model=run_record["model"],
            recon_data=run_record["recon_data"],
            observations=run_record["observations"],
            ai_analysis=run_record["ai_analysis"],
            user_question=run_record["user_question"],
        )
        run_record["json_report"] = build_json_report(
            target=run_record["target"],
            target_type=run_record["target_type"],
            task=run_record["task"],
            timestamp_utc=run_record["timestamp_utc"],
            model=run_record["model"],
            recon_data=run_record["recon_data"],
            observations=run_record["observations"],
            ai_analysis=run_record["ai_analysis"],
            user_question=run_record["user_question"],
        )

        st.session_state.latest_run = run_record
        st.session_state.followup_messages = []
        st.session_state.saved_reports.insert(0, run_record)

    latest = st.session_state.latest_run
    if not latest:
        st.info(
            "Enter an authorized domain or IP (such as `example.com` or `127.0.0.1`) above and click **Run Authorized Analysis**."
        )
        return

    st.divider()
    tab_findings, tab_ai, tab_followup, tab_report = st.tabs(
        [
            "Target & Recon Findings",
            "AI Explanation & Recommendations",
            "Ask Follow-Up Questions",
            "Structured Report",
        ]
    )

    with tab_findings:
        _render_recon_findings(latest)

    with tab_ai:
        if just_streamed_ai:
            st.caption("AI interpretation streamed above; switch back here anytime while reviewing findings.")
        if latest["ai_analysis"]:
            st.markdown(latest["ai_analysis"])
        else:
            st.warning("AI interpretation is unavailable for this run (check Ollama connection).")

    with tab_followup:
        st.caption(f"Ask follow-up questions grounded in the collected data for `{latest['target']}`.")
        for msg in st.session_state.followup_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        followup_prompt = st.chat_input(
            f"Ask a follow-up question about {latest['target']}...",
            key="followup_chat_input",
        )
        if followup_prompt and followup_prompt.strip():
            q_text = followup_prompt.strip()
            history_before = list(st.session_state.followup_messages)
            st.session_state.followup_messages.append({"role": "user", "content": q_text})
            with st.chat_message("user"):
                st.markdown(q_text)
            with st.chat_message("assistant"):
                try:
                    stream = engine.stream_study_qa(
                        question=q_text,
                        history=history_before,
                        recon_context=latest["recon_data"],
                        model=st.session_state.selected_model,
                        temperature=float(st.session_state.temperature),
                    )
                    reply = str(st.write_stream(stream)).strip()
                    if engine.using_cpu_fallback:
                        st.session_state.cpu_fallback_active = True
                    if reply:
                        st.session_state.followup_messages.append({"role": "assistant", "content": reply})
                except OllamaError as exc:
                    st.error(str(exc))

    with tab_report:
        dl1, dl2 = st.columns(2)
        safe_slug = latest["target"].replace(":", "_")
        with dl1:
            st.download_button(
                "Download Markdown Report (.md)",
                data=latest["markdown_report"],
                file_name=f"recon_report_{safe_slug}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Download JSON Report (.json)",
                data=latest["json_report"],
                file_name=f"recon_report_{safe_slug}.json",
                mime="application/json",
                use_container_width=True,
            )
        st.markdown(latest["markdown_report"])


def _render_study_qa(engine: OllamaEngine) -> None:
    st.subheader("Security Study Q&A")
    st.caption("Ask local Ollama about networking, DNS, IP addressing, TLS, or defensive security concepts.")

    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        use_context = st.checkbox(
            "Include latest reconnaissance findings as context (if available)",
            value=bool(st.session_state.latest_run),
            disabled=not bool(st.session_state.latest_run),
        )
    with top_c2:
        if st.button("Clear Q&A", use_container_width=True):
            st.session_state.study_chat_messages = []
            st.rerun()

    for msg in st.session_state.study_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a security or networking study question...")
    if question and question.strip():
        clean_q = question.strip()
        prior = list(st.session_state.study_chat_messages)
        st.session_state.study_chat_messages.append({"role": "user", "content": clean_q})
        with st.chat_message("user"):
            st.markdown(clean_q)

        recon_ctx = (
            st.session_state.latest_run["recon_data"]
            if use_context and st.session_state.latest_run
            else None
        )
        with st.chat_message("assistant"):
            try:
                stream = engine.stream_study_qa(
                    question=clean_q,
                    history=prior,
                    recon_context=recon_ctx,
                    model=st.session_state.selected_model,
                    temperature=float(st.session_state.temperature),
                )
                ans = str(st.write_stream(stream)).strip()
                if engine.using_cpu_fallback:
                    st.session_state.cpu_fallback_active = True
                if ans:
                    st.session_state.study_chat_messages.append({"role": "assistant", "content": ans})
            except OllamaError as exc:
                st.error(str(exc))


def _render_reports_history() -> None:
    st.subheader("Reports History")
    reports = st.session_state.saved_reports
    if not reports:
        st.info("No reports generated in this session yet. Run an authorized analysis to generate a report.")
        return

    for idx, rep in enumerate(reports):
        with st.expander(
            f"{rep['target']} — {rep['task']} ({rep['timestamp_utc']})",
            expanded=(idx == 0),
        ):
            safe_slug = rep["target"].replace(":", "_")
            b1, b2 = st.columns(2)
            b1.download_button(
                "Download Markdown (.md)",
                data=rep["markdown_report"],
                file_name=f"report_{idx}_{safe_slug}.md",
                mime="text/markdown",
                key=f"hist_md_{idx}",
                use_container_width=True,
            )
            b2.download_button(
                "Download JSON (.json)",
                data=rep["json_report"],
                file_name=f"report_{idx}_{safe_slug}.json",
                mime="application/json",
                key=f"hist_json_{idx}",
                use_container_width=True,
            )
            st.markdown(rep["markdown_report"])


def render_app() -> None:
    """Main Streamlit UI entry point."""
    st.set_page_config(
        page_title="AI Study Assistant",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    engine = _build_engine()
    conn_ok, conn_msg, models = _get_status(engine)

    _render_header(conn_ok=conn_ok)
    _render_sidebar(engine=engine, conn_ok=conn_ok, conn_msg=conn_msg, models=models)

    page = st.session_state.current_page
    if page == "Recon & Study Analysis":
        _render_recon_workflow(engine=engine)
    elif page == "Security Study Q&A":
        _render_study_qa(engine=engine)
    elif page == "Reports History":
        _render_reports_history()

