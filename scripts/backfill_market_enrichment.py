"""Backfill limit boards, Dragon–Tiger lists and Zhitu moneyflow for cached trade dates."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from app.container import build_container


async def backfill(args: argparse.Namespace) -> int:
    container = build_container(args.data_dir)
    cache = container.kline_cache_service
    board = container.limit_board_service
    dates = [
        day
        for day in cache.store.get_trade_dates_from_db()
        if args.start <= day <= args.end and cache._date_settled(day)
    ]
    if not dates:
        print("No settled trade dates in the local K-line cache.", flush=True)
        return 1
    print(f"Cached trade dates: {dates[0]} to {dates[-1]} ({len(dates)})", flush=True)
    incomplete = False
    if args.dataset != "moneyflow":
        for day in dates:
            result = await board.sync_trade_date(
                day,
                datasets=(
                    "limit_board",
                    "limit_ladder",
                    "dragon_tiger",
                    "dragon_tiger_seat",
                ),
                trigger_mode="manual",
            )
            items = result["datasets"]
            incomplete |= not items or any(
                item["status"] != "success"
                for item in items
                if item["dataset"] != "limit_ladder"
            )
            print(
                day,
                {i["dataset"]: (i["rows"], i["status"], i["reason"]) for i in items},
                flush=True,
            )
    if args.dataset != "boards":
        if not cache.market_data_client.zhitu:
            print("ZHITU_TOKEN is required for moneyflow backfill.", flush=True)
            return 1

        def progress(done: int, total: int, rows: int, errors: set[str]) -> None:
            if done == total or done % 320 == 0 or errors:
                print(
                    f"Moneyflow {done}/{total} symbols, {rows} rows, errors={sorted(errors)}",
                    flush=True,
                )

        start = date.fromisoformat(dates[0])
        last = date.fromisoformat(dates[-1])
        while start <= last:
            end = min(start + timedelta(days=364), last)
            result = await cache.sync_zhitu_moneyflow(
                start.isoformat(), end.isoformat(), progress=progress
            )
            print(start, end, result, flush=True)
            incomplete |= result["status"] != "success"
            start = end + timedelta(days=1)
    return int(incomplete)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--start",
        type=lambda value: date.fromisoformat(value).isoformat(),
        default="0001-01-01",
    )
    parser.add_argument(
        "--end",
        type=lambda value: date.fromisoformat(value).isoformat(),
        default=date.today().isoformat(),
    )
    parser.add_argument(
        "--dataset", choices=("all", "boards", "moneyflow"), default="all"
    )
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    load_dotenv()
    return asyncio.run(backfill(args))


if __name__ == "__main__":
    raise SystemExit(main())
