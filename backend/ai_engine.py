"""Local Ollama AI engine for security study and reconnaissance interpretation."""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests

from config.settings import DEFAULT_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


class OllamaEngine:
    """HTTP client communicating with a local Ollama server."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
        num_gpu: int | None = None,
        auto_cpu_fallback: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.auto_cpu_fallback = auto_cpu_fallback
        self.using_cpu_fallback = False

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}{endpoint}"

    def check_connection(self) -> tuple[bool, str]:
        """Check whether the Ollama server responds cleanly."""
        try:
            resp = requests.get(self._url("/api/tags"), timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return True, f"Connected ({len(models)} model(s) available)"
        except requests.ConnectionError:
            return False, f"Cannot reach Ollama at {self.base_url}. Is Ollama running?"
        except requests.Timeout:
            return False, f"Timed out connecting to Ollama at {self.base_url}."
        except requests.RequestException as exc:
            return False, f"Ollama connection error: {exc}"

    def list_models(self) -> list[str]:
        """List installed Ollama models, keeping default_model first when available."""
        try:
            resp = requests.get(self._url("/api/tags"), timeout=5)
            resp.raise_for_status()
            raw = resp.json().get("models", [])
        except requests.RequestException:
            return [self.default_model]

        names = list(dict.fromkeys(m.get("name", "") for m in raw if m.get("name")))
        if not names:
            return [self.default_model]
        if self.default_model in names:
            names.remove(self.default_model)
            names.insert(0, self.default_model)
        return names

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.4,
        stream: bool = False,
        num_gpu: int | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": float(temperature)}
        effective_gpu = num_gpu if num_gpu is not None else self.num_gpu
        if effective_gpu is None and self.using_cpu_fallback:
            effective_gpu = 0
        if effective_gpu is not None:
            options["num_gpu"] = effective_gpu

        return {
            "model": model or self.default_model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }

    @staticmethod
    def _is_cuda_init_error(detail: str) -> bool:
        low = detail.lower()
        return "cuda error" in low or "shared object initialization failed" in low

    @staticmethod
    def _extract_error(response: requests.Response | None) -> str:
        if response is None:
            return "No response received"
        try:
            body = response.json()
            if isinstance(body, dict) and "error" in body:
                return str(body["error"])
        except ValueError:
            pass
        return f"HTTP {response.status_code}: {response.text[:200]}"

    def _chat_once(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.4,
        num_gpu: int | None = None,
    ) -> str:
        payload = self._build_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=False,
            num_gpu=num_gpu,
        )
        try:
            resp = requests.post(self._url("/api/chat"), json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError as exc:
            raise OllamaError(f"Could not connect to Ollama at {self.base_url}.") from exc
        except requests.Timeout as exc:
            raise OllamaError("Ollama request timed out. Try again or reduce prompt size.") from exc
        except requests.HTTPError as exc:
            raise OllamaError(f"Ollama HTTP error: {self._extract_error(exc.response)}") from exc
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Ollama communication error: {exc}") from exc

        if "error" in data:
            raise OllamaError(str(data["error"]))
        content = data.get("message", {}).get("content", "")
        if not content:
            raise OllamaError("Ollama returned an empty response.")
        return content

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.4,
    ) -> str:
        """Send a synchronous chat request with automatic CPU fallback on CUDA init failure."""
        try:
            return self._chat_once(messages=messages, model=model, temperature=temperature)
        except OllamaError as exc:
            if (
                self.auto_cpu_fallback
                and self.num_gpu is None
                and not self.using_cpu_fallback
                and self._is_cuda_init_error(str(exc))
            ):
                self.using_cpu_fallback = True
                return self._chat_once(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    num_gpu=0,
                )
            raise

    def _chat_stream_once(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.4,
        num_gpu: int | None = None,
    ) -> Iterator[str]:
        payload = self._build_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=True,
            num_gpu=num_gpu,
        )
        try:
            with requests.post(
                self._url("/api/chat"),
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(str(chunk["error"]))
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except OllamaError:
            raise
        except requests.ConnectionError as exc:
            raise OllamaError(f"Could not connect to Ollama at {self.base_url}.") from exc
        except requests.Timeout as exc:
            raise OllamaError("Ollama stream timed out. Please try again.") from exc
        except requests.HTTPError as exc:
            raise OllamaError(f"Ollama HTTP error: {self._extract_error(exc.response)}") from exc
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Ollama streaming error: {exc}") from exc

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.4,
    ) -> Iterator[str]:
        """Stream chat tokens with automatic CPU fallback if CUDA fails before first token."""
        emitted = False
        try:
            for token in self._chat_stream_once(
                messages=messages,
                model=model,
                temperature=temperature,
            ):
                emitted = True
                yield token
        except OllamaError as exc:
            if (
                not emitted
                and self.auto_cpu_fallback
                and self.num_gpu is None
                and not self.using_cpu_fallback
                and self._is_cuda_init_error(str(exc))
            ):
                self.using_cpu_fallback = True
                yield from self._chat_stream_once(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    num_gpu=0,
                )
                return
            raise

    @staticmethod
    def build_recon_analysis_messages(
        target: str,
        target_type: str,
        task_name: str,
        recon_data: dict[str, Any],
        user_question: str = "",
    ) -> list[dict[str, str]]:
        """Construct structured prompts to interpret reconnaissance findings safely."""
        system_prompt = (
            "You are a defensive cybersecurity educator and AI Study Assistant. "
            "Your job is to interpret authorized preliminary reconnaissance data accurately, "
            "explain networking and DNS/IP concepts in clear beginner-friendly language, "
            "highlight defensive security observations strictly supported by the collected data, "
            "and recommend practical defensive hardening steps. "
            "Never invent findings that are not present in the provided data, and never provide offensive exploitation instructions."
        )

        findings_json = json.dumps(recon_data, indent=2, default=str)[:9000]
        question_block = (
            f"\nStudent's specific question:\n{user_question.strip()}\n"
            if user_question.strip()
            else ""
        )

        user_prompt = (
            f"Target: {target} (Type: {target_type})\n"
            f"Analysis Task: {task_name}\n"
            f"{question_block}\n"
            f"Collected Preliminary Reconnaissance Data:\n```json\n{findings_json}\n```\n\n"
            "Provide a clear, structured response with these sections:\n"
            "1. **Plain-English Summary** — What this target and the collected records represent.\n"
            "2. **Key Findings Explained** — Break down the DNS, domain, or IP details simply.\n"
            "3. **Potential Security Observations** — Note concrete posture signals from the data (e.g., SPF/DMARC/CAA presence, DNS redundancy, TLS status, public vs private IP).\n"
            "4. **Suggested Defensive Recommendations** — Actionable security hygiene steps.\n"
            + ("5. **Direct Answer to Your Question** — Address the student's question clearly.\n" if user_question.strip() else "")
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def analyze_recon_findings(
        self,
        target: str,
        target_type: str,
        task_name: str,
        recon_data: dict[str, Any],
        user_question: str = "",
        model: str | None = None,
        temperature: float = 0.4,
    ) -> str:
        messages = self.build_recon_analysis_messages(
            target=target,
            target_type=target_type,
            task_name=task_name,
            recon_data=recon_data,
            user_question=user_question,
        )
        return self.chat(messages=messages, model=model, temperature=temperature)

    def stream_analyze_recon_findings(
        self,
        target: str,
        target_type: str,
        task_name: str,
        recon_data: dict[str, Any],
        user_question: str = "",
        model: str | None = None,
        temperature: float = 0.4,
    ) -> Iterator[str]:
        messages = self.build_recon_analysis_messages(
            target=target,
            target_type=target_type,
            task_name=task_name,
            recon_data=recon_data,
            user_question=user_question,
        )
        yield from self.chat_stream(messages=messages, model=model, temperature=temperature)

    @staticmethod
    def build_study_qa_messages(
        question: str,
        history: list[dict[str, str]] | None = None,
        recon_context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """Construct messages for general security study Q&A or follow-up questions."""
        system_prompt = (
            "You are a helpful, accurate AI Security Study Assistant. "
            "Explain cybersecurity, networking, DNS, IP addressing, and defensive concepts clearly for students. "
            "Stay strictly defensive and educational."
        )
        if recon_context:
            context_snippet = json.dumps(recon_context, indent=2, default=str)[:6000]
            system_prompt += (
                "\n\nLatest authorized reconnaissance context for reference:\n"
                f"```json\n{context_snippet}\n```"
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            for item in history[-12:]:
                role = item.get("role", "")
                content = item.get("content", "")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question.strip()})
        return messages

    def stream_study_qa(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        recon_context: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.5,
    ) -> Iterator[str]:
        messages = self.build_study_qa_messages(
            question=question,
            history=history,
            recon_context=recon_context,
        )
        yield from self.chat_stream(messages=messages, model=model, temperature=temperature)

