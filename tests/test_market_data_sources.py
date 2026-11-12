import asyncio

import pandas as pd

from app.services.market_data_sources import (
    MarketDataSourceError,
    ZhituMarketData,
    _date_ms,
    _read_dump,
)
from app.services.tushare_market_data_client import TushareFirstMarketDataClient


def test_fuyao_dump_keeps_trade_dates_units_and_adjustment_boundary(tmp_path):
    start = _date_ms("2026-11-10")
    end = _date_ms("2026-11-11")
    daily = tmp_path / "daily.parquet"
    events = tmp_path / "actions.parquet"
    pd.DataFrame(
        {
            "thscode": ["600519.SH", "000001.SZ", "000001.SZ"],
            "date_ms": [start, end, end + 86_400_000],
            "open_price": [100.0, 10.0, 11.0],
            "high_price": [110.0, 11.0, 12.0],
            "low_price": [90.0, 9.0, 10.0],
            "close_price": [100.0, 10.0, 11.0],
            "volume": [1200, 3500, 1300],
            "turnover": [120_000.0, 35_000.0, 14_300.0],
        }
    ).to_parquet(daily)
    pd.DataFrame(
        {"thscode": ["600519.SH"], "ex_date_ms": [end + 86_400_000]}
    ).to_parquet(events)

    raw = _read_dump(daily, None, start, end)
    adjusted = _read_dump(daily, events, start, end)

    assert raw["日期"].tolist() == ["2026-11-10", "2026-11-11"]
    assert adjusted[["代码", "成交量", "成交额"]].values.tolist() == [
        ["000001", 35.0, 35_000.0]
    ]
    assert _read_dump(daily, None, start, end + 172_800_000).empty


def test_zhitu_failure_keeps_legacy_source_and_reason():
    class UnavailableZhitu:
        async def fetch_spot(self):
            raise MarketDataSourceError("zhitu_http_403")

    class Fallback:
        async def fetch_spot(self):
            return pd.DataFrame({"代码": ["000001"]})

    client = TushareFirstMarketDataClient(fallback=Fallback(), zhitu=UnavailableZhitu())
    snapshot = asyncio.run(client.fetch_spot())

    assert snapshot.attrs["data_source"] == "eastmoney"
    assert snapshot.attrs["fallback_reason"] == "zhitu_http_403"


def test_zhitu_normalizes_exchange_suffixes_for_lookup_and_spot(monkeypatch):
    client = ZhituMarketData("fixture-token")

    async def items(path):
        if path == "/hs/list/all":
            return [{"dm": "000001.SZ", "mc": "平安银行"}]
        if path == "/hs/public/realall":
            return [{"dm": "000001", "p": 11.0, "t": "2026-11-12 15:30:00"}]
        return []

    monkeypatch.setattr(client, "_items", items)

    async def check():
        assert await client.fetch_symbol_names() == {"000001": "平安银行"}
        spot = await client.fetch_spot()
        assert spot[["代码", "名称", "行情日期"]].values.tolist() == [
            ["000001", "平安银行", "2026-11-12"]
        ]

    asyncio.run(check())
