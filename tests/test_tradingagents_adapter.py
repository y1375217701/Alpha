from __future__ import annotations

import json
import subprocess

import pytest

from app.services.tradingagents_adapter import (
    DEEPSEEK_BACKEND_URL,
    DEEPSEEK_PROVIDER,
    DEFAULT_DEEP_MODEL,
    TradingAgentsAdapter,
)

@pytest.mark.parametrize("provider", [DEEPSEEK_PROVIDER, "openai"])
def test_tradingagents_adapter_uses_configured_backend(monkeypatch, tmp_path, provider):
    monkeypatch.setenv("TRADINGAGENTS_PROVIDER", provider)
    if provider == "openai":
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8317/v1")
        monkeypatch.setenv("HERMES_MODEL", "gpt-6.1-sol")
        backend_url = "http://127.0.0.1:8317/v1"
        quick_model = deep_model = "gpt-6.1-sol"
    else:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        backend_url = DEEPSEEK_BACKEND_URL
        quick_model, deep_model = "deepseek-flash", DEFAULT_DEEP_MODEL
    monkeypatch.delenv("TRADINGAGENTS_UV_BIN", raising=False)
    monkeypatch.setattr("app.services.tradingagents_adapter.shutil.which", lambda _: "uv")

    adapter = TradingAgentsAdapter(repo_path=tmp_path / "TradingAgents", runtime_root=tmp_path / "runtime")
    adapter.repo_path.mkdir(parents=True, exist_ok=True)
    (adapter.repo_path / "pyproject.toml").touch()
    captured = {}

    def _fake_run(cmd, cwd, capture_output, encoding, errors, check, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        output_file = cmd[cmd.index("--output-file") + 1]
        payload = {
            "decision": "BUY",
            "final_trade_decision": "BUY with strong conviction",
            "investment_plan": "Accumulate on strength.",
            "trader_investment_plan": "Enter in two tranches.",
            "market_report": "Market report",
            "news_report": "News report",
            "fundamentals_report": "Fundamentals report",
        }
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = adapter.analyze("600001", "2026-04-22")

    assert captured["cwd"] == str(adapter.repo_path)
    assert captured["cmd"][:6] == ["uv", "run", "--no-sync", "python", "-X", "utf8"]
    assert captured["cmd"][6].endswith("tradingagents_runner.py")
    assert "--provider" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--provider") + 1] == provider
    assert captured["cmd"][captured["cmd"].index("--ticker") + 1] == "600001.SS"
    assert captured["cmd"][captured["cmd"].index("--quick-model") + 1] == quick_model
    assert captured["cmd"][captured["cmd"].index("--deep-model") + 1] == deep_model
    assert result["provider"] == provider
    assert result["backend_url"] == backend_url
    assert result["decision"] == "BUY"
    assert result["decision_action"] == "buy"
    assert result["decision_action_text"] == "买入"
