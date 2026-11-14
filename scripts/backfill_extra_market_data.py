"""Backfill Alpha's daily enrichment using AxData, Zhitu and Fuyao."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from app.config import POOL_BUY
from app.container import build_container
from app.services.market_data_sources import read_fuyao_raw_turnover
from app.services.time_utils import now_cn
from app.services.tushare_market_data_client import _BENCHMARK_CODES


async def backfill(args: argparse.Namespace) -> None:
    container = build_container()
    cache = container.kline_cache_service
    client = cache.market_data_client
    store = cache.store
    dates = [
        day
        for day in store.get_trade_dates_from_db()
        if args.start <= day <= args.end
        and day <= now_cn().date().isoformat()
        and cache._date_settled(day)
    ]
    if not dates:
        raise SystemExit("No settled K-line dates in range")
    start, end = dates[0], dates[-1]
    now = now_cn().isoformat()
    codes = store.get_all_symbols()
    print(f"{args.dataset}: {start} to {end}, {len(codes)} symbols", flush=True)

    if args.dataset == "limits":
        if not client.zhitu:
            raise SystemExit("ZHITU_TOKEN is required")

        def report(done, total, rows, errors):
            if done % 320 == 0 or done == total:
                print(
                    f"limits {done}/{total}: {rows} rows, errors={dict(errors)}",
                    flush=True,
                )

        pending = store.get_daily_extra_gaps(start, end, "limits")
        print(
            await cache.sync_zhitu_price_limits(start, end, pending, progress=report),
            flush=True,
        )

    if args.dataset == "valuation":
        if not client.axdata:
            raise SystemExit("AXDATA_BASE_URL is required")
        pending = store.get_daily_extra_gaps(start, end, "valuation")

        def report(done, total, rows, error):
            if done % 4 == 0 or done == total or error:
                print(f"valuation {done}/{total}: {rows} rows {error}", flush=True)

        await cache.sync_axdata_valuations(
            start,
            end,
            pending,
            batch_size=args.batch_size,
            workers=args.workers,
            progress=report,
        )

    if args.dataset == "boards":
        board = container.limit_board_service
        for day in dates:
            result = await board.sync_trade_date(
                day, datasets=["limit_board_theme", "hot_money_seat"]
            )
            print(
                day,
                [(r["dataset"], r["status"], r["rows"]) for r in result["datasets"]],
                flush=True,
            )

    if args.dataset == "indices-connect":
        for dataset, fetch, write, state_date in (
            (
                "index_daily",
                client.fuyao.fetch_index_range(start, end, list(_BENCHMARK_CODES)),
                store.upsert_index_daily,
                end,
            ),
            (
                "hsgt",
                client.zhitu.connect_flows("2014-11-17", end),
                store.upsert_market_capital_flow,
                end,
            ),
        ):
            result = await cache._run_enrichment_dataset(
                dataset, lambda: fetch, write, state_date, now
            )
            print(result, flush=True)
        rows = await client.zhitu.connect_top10(start, end)
        count = store.upsert_hsgt_top10(rows, now)
        latest = sum(row["trade_date"] == end for row in rows)
        store.set_enrichment_state(
            "hsgt_top10",
            trade_date=end,
            status="success" if latest else "partial",
            row_count=latest,
            message="source=zhitu.connect_top10",
            updated_at=now,
        )
        print(f"hsgt_top10: {count} rows", flush=True)

    if args.dataset == "minutes":
        symbols = await container.funnel_service.get_pool_symbols(POOL_BUY)
        for service in (
            container.predict_funnel_service,
            container.hot_stock_ai_service,
        ):
            symbols += [
                row["symbol"]
                for row in service.get_snapshot().get("pools", {}).get("buy", [])
            ]
        symbols += args.symbol
        if not symbols:
            print("Buy pool is empty", flush=True)
            return
        result = await cache.backfill_axdata_minutes(sorted(set(symbols)), start, end)
        print(result, flush=True)

    if args.dataset == "factors":
        result = await cache.sync_fuyao_adjustments(
            start, end, local_dumps=args.local_dumps
        )
        print(result, flush=True)

    if args.dataset == "daily":
        print(await cache.sync_alternative_daily_extras(end), flush=True)
        ratios = await asyncio.to_thread(
            store.derive_daily_volume_ratios, start, end, now
        )
        print(f"Derived {ratios} daily volume ratios", flush=True)
        path = Path(store.db_path).parent / "fuyao-daily-k.parquet"
        if args.local_dumps and path.is_file():
            raw = await asyncio.to_thread(read_fuyao_raw_turnover, path, start, end)
            count = 0
            for offset in range(0, len(raw), 20_000):
                batch = list(
                    raw.iloc[offset : offset + 20_000].itertuples(
                        index=False, name=None
                    )
                )
                count += await asyncio.to_thread(
                    store.derive_daily_turnover, batch, now
                )
            print(f"Derived {count} daily turnover ratios", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=(
            "limits",
            "valuation",
            "boards",
            "indices-connect",
            "minutes",
            "factors",
            "daily",
        ),
    )
    parser.add_argument(
        "--start",
        type=lambda x: date.fromisoformat(x).isoformat(),
        default="2000-01-01",
    )
    parser.add_argument(
        "--end",
        type=lambda x: date.fromisoformat(x).isoformat(),
        default=now_cn().date().isoformat(),
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument(
        "--batch-size", type=int, choices=range(1, 65), default=24, metavar="1-64"
    )
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--local-dumps", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    asyncio.run(backfill(args))


if __name__ == "__main__":
    main()
