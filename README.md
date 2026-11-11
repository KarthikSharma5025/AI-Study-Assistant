# AI Study Assistant

A local AI-powered **security study and preliminary reconnaissance assistant** built with **Python**, **Streamlit**, and **Ollama**. It validates domain and IP inputs against an authorized assessment scope, runs passive domain/DNS/IP lookups, interprets findings locally with an LLM (such as `llama3:8b`), and exports structured reports.

## Features

- **Input & Authorized Scope Validation (`backend/input_validator.py`)**: Validates domain, IPv4, and IPv6 formats and verifies targets against the configured authorized scope list before running any lookups.
- **Preliminary Reconnaissance (`reconnaissance/`)**:
  - **Domain Lookup**: Hostname resolution (`socket`), passive registration metadata via RDAP, and public TLS certificate inspection.
  - **DNS Analysis**: `A`, `AAAA`, `MX`, `NS`, `TXT`, `CNAME`, `SOA`, `CAA`, and `_dmarc` lookups using `dnspython`, plus automated SPF/DMARC/CAA posture checks.
  - **IP Lookup**: IPv4/IPv6 classification (`ipaddress`), reverse DNS (`PTR`), and public RDAP network info for globally routable addresses.
- **Local AI Analysis & Follow-Up Q&A (`backend/ai_engine.py`)**: Uses a local Ollama server (`llama3:8b` by default) to explain findings, highlight security observations, suggest defensive recommendations, and answer follow-up questions.
- **Structured Report Generation (`reports/report_generator.py`)**: Generates downloadable Markdown (`.md`) and JSON (`.json`) assessment reports.

## Project Structure

```text
AI-Study-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
├── frontend/
│   └── interface.py
│
├── backend/
│   ├── ai_engine.py
│   ├── input_validator.py
│   └── workflow.py
│
├── reconnaissance/
│   ├── domain_lookup.py
│   ├── dns_analysis.py
│   └── ip_lookup.py
│
├── reports/
│   └── report_generator.py
│
├── config/
│   └── settings.py
│
└── screenshots/
```

## Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Backend logic and workflow orchestration |
| Streamlit | Interactive web user interface |
| Ollama (`llama3:8b`) | Local LLM execution and analysis |
| Requests | Ollama REST API & public RDAP HTTP requests |
| `dnspython`, `socket`, `ipaddress`, `ssl` | DNS queries, IP classification, hostname resolution, TLS metadata |
| HTML / CSS | Custom Streamlit layout and theme styling |

## Quick Start

1. Ensure [Ollama](https://ollama.com/) is installed and running locally with a model pulled:
   ```powershell
   ollama pull llama3:8b
   ```
2. Install Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```powershell
   streamlit run app.py
   ```

> **Note:** Only assess domains and IP addresses you own or have explicit authorization to evaluate.

