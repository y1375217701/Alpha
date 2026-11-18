from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

DEEPSEEK_PROVIDER = "deepseek"
DEEPSEEK_BACKEND_URL = "https://api.deepseek.com"
DEFAULT_QUICK_MODEL = "deepseek-flash"
DEFAULT_DEEP_MODEL = "deepseek-v4-pro"
OPENAI_PROVIDER = "openai"
OPENAI_BACKEND_URL = "https://api.openai.com/v1"
_RUNNER = Path(__file__).with_name("tradingagents_runner.py").resolve()
_LEGACY_MODELS = {
    "deepseek-chat": DEFAULT_QUICK_MODEL,
    "deepseek-v4-flash": DEFAULT_QUICK_MODEL,
    "deepseek-reasoner": DEFAULT_DEEP_MODEL,
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def resolve_tradingagents_models(
    quick_model: str = DEFAULT_QUICK_MODEL,
    deep_model: str = DEFAULT_DEEP_MODEL,
) -> tuple[str, str, str]:
    provider = os.getenv("TRADINGAGENTS_PROVIDER", DEEPSEEK_PROVIDER).strip().lower()
    if provider == OPENAI_PROVIDER:
        model = os.getenv("HERMES_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        return provider, model, model
    quick = quick_model.strip() or DEFAULT_QUICK_MODEL
    deep = deep_model.strip() or DEFAULT_DEEP_MODEL
    return DEEPSEEK_PROVIDER, _LEGACY_MODELS.get(quick, quick), _LEGACY_MODELS.get(deep, deep)


class TradingAgentsAdapter:
    def __init__(
        self,
        repo_path: str | Path | None = None,
        runtime_root: str | Path = "data/tradingagents_runtime",
    ) -> None:
        self.repo_path = Path(
            repo_path or os.getenv("TRADINGAGENTS_REPO_PATH", "").strip()
            or Path(__file__).resolve().parents[3] / "TradingAgents"
        ).expanduser().resolve()
        self.uv_bin = os.getenv("TRADINGAGENTS_UV_BIN", "").strip() or "uv"
        self.runtime_root = Path(runtime_root).resolve()
        self.runtime_root.mkdir(parents=True, exist_ok=True)

    def describe_runtime(self) -> dict[str, str]:
        provider, quick_model, deep_model = resolve_tradingagents_models()
        backend_url = DEEPSEEK_BACKEND_URL
        if provider == OPENAI_PROVIDER:
            backend_url = os.getenv("OPENAI_BASE_URL", OPENAI_BACKEND_URL).strip().rstrip("/")
        return {
            "repo_path": str(self.repo_path),
            "provider": provider,
            "backend_url": backend_url,
            "quick_model": quick_model,
            "deep_model": deep_model,
        }

    @staticmethod
    def to_vendor_symbol(symbol: str) -> str:
        raw = str(symbol or "").strip()
        if raw.startswith(("60", "68")):
            return f"{raw}.SS"
        if raw.startswith(("00", "30")):
            return f"{raw}.SZ"
        if raw.startswith(("43", "83", "87", "92")):
            return f"{raw}.BJ"
        return raw

    def _check_runtime(self) -> None:
        if not (self.repo_path / "pyproject.toml").is_file():
            raise FileNotFoundError(f"TradingAgents repo not found: {self.repo_path}")
        if shutil.which(self.uv_bin) is None:
            raise FileNotFoundError(f"uv executable not found: {self.uv_bin}")

    @staticmethod
    def _decision_bonus(decision: str) -> float:
        mapping = {
            "BUY": 2.0,
            "OVERWEIGHT": 1.0,
            "HOLD": 0.0,
            "UNDERWEIGHT": -1.0,
            "SELL": -2.0,
        }
        return mapping.get(str(decision or "").strip().upper(), 0.0)

    @staticmethod
    def _decision_action(decision: str) -> tuple[str, str]:
        value = str(decision or "").strip().upper()
        mapping = {
            "BUY": ("buy", "买入"),
            "OVERWEIGHT": ("buy", "买入"),
            "HOLD": ("watch", "观望"),
            "UNDERWEIGHT": ("sell", "卖出"),
            "SELL": ("sell", "卖出"),
        }
        return mapping.get(value, ("watch", "观望"))

    def analyze(
        self,
        symbol: str,
        trade_date: str,
        *,
        provider: str = DEEPSEEK_PROVIDER,
        quick_model: str = DEFAULT_QUICK_MODEL,
        deep_model: str = DEFAULT_DEEP_MODEL,
        selected_analysts: list[str] | None = None,
        output_language: str = "Chinese",
        timeout_seconds: float = 600,
    ) -> dict[str, Any]:
        provider, quick_model, deep_model = resolve_tradingagents_models(quick_model, deep_model)
        key_name = "OPENAI_API_KEY" if provider == OPENAI_PROVIDER else "DEEPSEEK_API_KEY"
        if not os.getenv(key_name, "").strip():
            raise RuntimeError(f"{key_name} 未设置，无法调用 TradingAgents 分析")
        self._check_runtime()
        vendor_symbol = self.to_vendor_symbol(symbol)
        analyst_csv = ",".join(selected_analysts or ["market", "news", "fundamentals"])
        with tempfile.NamedTemporaryFile(prefix="tradingagents_", suffix=".json", dir=self.runtime_root, delete=False) as fh:
            output_path = Path(fh.name)
        command = [
            self.uv_bin, "run", "--no-sync", "python", "-X", "utf8", str(_RUNNER),
            "--ticker", vendor_symbol,
            "--date", trade_date,
            "--provider", provider,
            "--quick-model", quick_model,
            "--deep-model", deep_model,
            "--analysts", analyst_csv,
            "--output-language", output_language,
            "--output-file", str(output_path),
        ]
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.repo_path),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_seconds,
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout or "").strip() or f"TradingAgents CLI failed with code {proc.returncode}")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        finally:
            output_path.unlink(missing_ok=True)

        decision = str(payload.get("decision") or "").strip().upper()
        decision_text = _normalize_whitespace(payload.get("final_trade_decision", ""))
        investment_plan = _normalize_whitespace(payload.get("investment_plan", ""))
        trader_plan = _normalize_whitespace(payload.get("trader_investment_plan", ""))
        market_report = _normalize_whitespace(payload.get("market_report", ""))
        news_report = _normalize_whitespace(payload.get("news_report", ""))
        fundamentals_report = _normalize_whitespace(payload.get("fundamentals_report", ""))
        summary_source = decision_text or trader_plan or investment_plan or market_report
        summary = summary_source[:360]
        action, action_text = self._decision_action(decision)

        return {
            "ok": True,
            "symbol": symbol,
            "vendor_symbol": vendor_symbol,
            "trade_date": trade_date,
            "provider": provider,
            "backend_url": self.describe_runtime()["backend_url"],
            "decision": decision,
            "decision_action": action,
            "decision_action_text": action_text,
            "score_bonus": self._decision_bonus(decision),
            "summary": summary,
            "discussion": decision_text[:4000],
            "command": " ".join(command),
            "reports": {
                "market": market_report[:1200],
                "news": news_report[:1200],
                "fundamentals": fundamentals_report[:1200],
                "investment_plan": investment_plan[:1200],
                "trader_plan": trader_plan[:1200],
            },
        }
