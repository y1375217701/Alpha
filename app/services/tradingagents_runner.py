"""Run TradingAgents headlessly inside its own Python environment."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--quick-model", required=True)
    parser.add_argument("--deep-model", required=True)
    parser.add_argument("--analysts", required=True)
    parser.add_argument("--output-language", required=True)
    parser.add_argument("--output-file", required=True, type=Path)
    args = parser.parse_args()

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    class DeepSeekTradingAgentsGraph(TradingAgentsGraph):
        def _get_provider_kwargs(self):
            # The generic OpenAI client drops reasoning_content between tool calls.
            return {**super()._get_provider_kwargs(), "reasoning_effort": "none"}

    config = {
        **DEFAULT_CONFIG,
        "llm_provider": args.provider,
        "quick_think_llm": args.quick_model,
        "deep_think_llm": args.deep_model,
        "output_language": args.output_language,
        "results_dir": str(args.output_file.parent / "logs"),
        "data_cache_dir": str(args.output_file.parent / "cache"),
    }
    if args.provider == "openai":
        config["backend_url"] = os.getenv("OPENAI_BASE_URL") or config["backend_url"]
    graph_type = DeepSeekTradingAgentsGraph if args.provider == "deepseek" else TradingAgentsGraph
    graph = graph_type(selected_analysts=args.analysts.split(","), config=config)
    state, decision = graph.propagate(args.ticker, args.date)
    fields = (
        "final_trade_decision", "investment_plan", "trader_investment_plan",
        "market_report", "news_report", "fundamentals_report",
    )
    payload = {"decision": decision, **{field: state.get(field, "") for field in fields}}
    args.output_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
