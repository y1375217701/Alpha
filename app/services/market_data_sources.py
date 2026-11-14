from __future__ import annotations

import asyncio
from datetime import date, timedelta
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

import httpx
import pandas as pd

from app.services.market_data_client import _normalize_symbol
from app.services.time_utils import now_cn

_ZHITU = "https://api.zhituapi.com"
_FUYAO = "https://fuyao.aicubes.cn"
_EASTMONEY_DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_ZHITU_BOARD_POOLS = {"U": "ztgc", "D": "dtgc", "Z": "zbgc"}
_ZHITU_MONEYFLOW_FIELDS = {
    "buy_sm_amount": "zmbxdcje",
    "sell_sm_amount": "zmsxdcje",
    "buy_md_amount": "zmbzdcje",
    "sell_md_amount": "zmszdcje",
    "buy_lg_amount": "zmbddcje",
    "sell_lg_amount": "zmsddcje",
    "buy_elg_amount": "zmbtdcje",
    "sell_elg_amount": "zmstdcje",
}
_EASTMONEY_DRAGON_TIGER_FIELDS = {
    "ts_code": "SECUCODE",
    "trade_date": "TRADE_DATE",
    "reason": "EXPLANATION",
    "name": "SECURITY_NAME_ABBR",
    "close": "CLOSE_PRICE",
    "pct_change": "CHANGE_RATE",
    "turnover_rate": "TURNOVERRATE",
    "amount": "ACCUM_AMOUNT",
    "l_buy": "BILLBOARD_BUY_AMT",
    "l_sell": "BILLBOARD_SELL_AMT",
    "l_amount": "BILLBOARD_DEAL_AMT",
    "net_amount": "BILLBOARD_NET_AMT",
    "net_rate": "DEAL_NET_RATIO",
    "amount_rate": "DEAL_AMOUNT_RATIO",
    "float_values": "FREE_MARKET_CAP",
}
_EASTMONEY_SEAT_FIELDS = {
    "ts_code": "SECUCODE",
    "trade_date": "TRADE_DATE",
    "reason": "EXPLANATION",
    "seat_name": "OPERATEDEPT_NAME",
    "buy_amount": "BUY",
    "sell_amount": "SELL",
    "net_amount": "NET",
    "buy_rate": "TOTAL_BUYRIO",
    "sell_rate": "TOTAL_SELLRIO",
}
_BAR_COLUMNS = {
    "open_price": "开盘",
    "high_price": "最高",
    "low_price": "最低",
    "close_price": "收盘",
    "volume": "成交量",
    "turnover": "成交额",
}
_SPOT_COLUMNS = {
    "dm": "代码",
    "p": "最新价",
    "yc": "昨收",
    "o": "今开",
    "h": "最高",
    "l": "最低",
    "pc": "涨跌幅",
    "ud": "涨跌额",
    "zf": "振幅",
    "hs": "换手率",
    "lb": "量比",
    "v": "成交量",
    "cje": "成交额",
    "pe": "市盈率-动态",
    "sjl": "市净率",
    "sz": "总市值",
    "lt": "流通市值",
    "zs": "涨速",
}


class MarketDataSourceError(RuntimeError):
    pass


async def _json(client: httpx.AsyncClient, source: str, url: str, **kwargs: Any) -> Any:
    try:
        response = await client.get(url, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise MarketDataSourceError(
            f"{source}_http_{exc.response.status_code}"
        ) from None
    except (httpx.RequestError, ValueError) as exc:
        raise MarketDataSourceError(f"{source}_{type(exc).__name__}") from None


class EastmoneyDragonTiger:
    async def _report(
        self, client: httpx.AsyncClient, report: str, trade_date: str
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 1
        trade_date = date.fromisoformat(trade_date).isoformat()
        while True:
            body = await _json(
                client,
                "eastmoney",
                _EASTMONEY_DATACENTER,
                params={
                    "reportName": report,
                    "columns": "ALL",
                    "pageNumber": page,
                    "pageSize": 500,
                    "filter": f"(TRADE_DATE='{trade_date}')",
                    "source": "WEB",
                    "client": "WEB",
                    "sortColumns": "TRADE_ID"
                    if report == "RPT_DAILYBILLBOARD_DETAILS"
                    else "TRADE_ID,OPERATEDEPT_CODE,BUY,SELL",
                    "sortTypes": "1"
                    if report == "RPT_DAILYBILLBOARD_DETAILS"
                    else "1,1,1,1",
                },
            )
            if not isinstance(body, dict):
                raise MarketDataSourceError("eastmoney_invalid_report")
            if page == 1 and body.get("code") == 9201:
                return []
            result = body.get("result")
            if (
                body.get("code") != 0
                or not isinstance(result, dict)
                or not isinstance(result.get("data"), list)
            ):
                raise MarketDataSourceError("eastmoney_invalid_report")
            rows.extend(result["data"])
            if page >= result["pages"]:
                if len(rows) != result["count"]:
                    raise MarketDataSourceError("eastmoney_incomplete_report")
                return rows
            page += 1

    async def fetch_daily(self, trade_date: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=12.0) as client:
            raw = await self._report(client, "RPT_DAILYBILLBOARD_DETAILS", trade_date)
        return [
            {
                field: row.get(column)
                for field, column in _EASTMONEY_DRAGON_TIGER_FIELDS.items()
            }
            for row in raw
        ]

    async def fetch_seats(self, trade_date: str) -> list[dict[str, Any]]:
        sides = ("buy", "sell")
        async with httpx.AsyncClient(timeout=12.0) as client:
            reports = await asyncio.gather(
                *(
                    self._report(
                        client, f"RPT_BILLBOARD_DAILYDETAILS{side.upper()}", trade_date
                    )
                    for side in sides
                ),
                return_exceptions=True,
            )
        rows = []
        for side, report in zip(sides, reports, strict=True):
            if isinstance(report, BaseException):
                raise report
            for row in report:
                rows.append(
                    {
                        **{
                            field: row.get(column)
                            for field, column in _EASTMONEY_SEAT_FIELDS.items()
                        },
                        "side": side,
                    }
                )
        return rows


class ZhituMarketData:
    def __init__(self, token: str) -> None:
        self._token = token
        self._names: dict[str, str] = {}
        self._names_until = 0.0
        self._spot = pd.DataFrame()
        self._spot_until = 0.0
        self._spot_lock = asyncio.Lock()
        self._names_lock = asyncio.Lock()
        try:
            requests_per_minute = int(
                os.getenv("ZHITU_MONEYFLOW_REQUESTS_PER_MINUTE", "240")
            )
        except ValueError:
            requests_per_minute = 240
        self._moneyflow_interval = 60 / max(1, min(requests_per_minute, 6000))
        self._moneyflow_lock = asyncio.Lock()
        self._next_moneyflow_request = 0.0
        self._moneyflow_blocked_until = 0.0

    async def _items(self, path: str, **params: Any) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=8.0) as client:
            data = await _json(
                client, "zhitu", _ZHITU + path, params={**params, "token": self._token}
            )
        if not isinstance(data, list):
            raise MarketDataSourceError("zhitu_invalid_response")
        return data

    async def fetch_symbol_names(self) -> dict[str, str]:
        async with self._names_lock:
            if monotonic() >= self._names_until:
                sh_sz = await self._items("/hs/list/all")
                try:
                    beijing = await self._items("/bj/list/all")
                except MarketDataSourceError:
                    beijing = []
                names = {
                    _normalize_symbol(row["dm"]): row["mc"]
                    for row in sh_sz + beijing
                    if isinstance(row.get("dm"), str) and row.get("mc")
                }
                names.pop("", None)
                if names:
                    self._names = names
                    self._names_until = monotonic() + 3600
            return self._names.copy()

    async def fetch_limit_pool(
        self, flag: str, trade_date: str
    ) -> list[dict[str, Any]]:
        pool = _ZHITU_BOARD_POOLS[flag]
        raw = await self._items(f"/hs/pool/{pool}/{trade_date}")
        return [
            {
                "symbol": row["dm"],
                "trade_date": trade_date,
                "name": row.get("mc"),
                "industry": row.get("hy"),
                "limit": flag,
                "first_time": row.get("fbt"),
                "last_time": row.get("lbt"),
                "open_times": row.get("zbc"),
                "limit_times": row.get("lbc"),
                "up_stat": row.get("tj") if flag == "U" else None,
                "fd_amount": row.get("zj"),
                "amount": row.get("cje"),
                "limit_amount": row.get("fba"),
                "turnover_ratio": row.get("hs"),
                "float_mv": row.get("lt"),
                "total_mv": row.get("zsz"),
                "close": row.get("p"),
                "pct_chg": row.get("zdf", row.get("zf")),
                "source": f"zhitu.{pool}",
            }
            for row in raw
            if row.get("dm")
        ]

    async def fetch_moneyflow(
        self, client: httpx.AsyncClient, ts_code: str, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        async with self._moneyflow_lock:
            await asyncio.sleep(max(0, self._next_moneyflow_request - monotonic()))
            if self._moneyflow_blocked_until > monotonic():
                raise MarketDataSourceError("zhitu_http_429")
            self._next_moneyflow_request = monotonic() + self._moneyflow_interval
        try:
            raw = await _json(
                client,
                "zhitu",
                f"{_ZHITU}/hs/history/transaction/{ts_code}",
                params={"token": self._token, "st": start_date, "et": end_date},
            )
        except MarketDataSourceError as exc:
            if str(exc) == "zhitu_http_404":
                return []
            if str(exc) == "zhitu_http_429":
                self._moneyflow_blocked_until = monotonic() + 60
            raise
        if not isinstance(raw, list):
            raise MarketDataSourceError("zhitu_invalid_moneyflow")
        rows = []
        for row in raw:
            trade_date = str(row.get("t", ""))[:10]
            if not start_date <= trade_date <= end_date:
                continue
            try:
                amounts = {
                    key: float(row[field])
                    for key, field in _ZHITU_MONEYFLOW_FIELDS.items()
                }
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(amount) for amount in amounts.values()):
                continue
            rows.append(
                {
                    "symbol": _normalize_symbol(ts_code),
                    "trade_date": trade_date,
                    **amounts,
                    "net_mf_amount": round(
                        sum(
                            amount if key.startswith("buy_") else -amount
                            for key, amount in amounts.items()
                        ),
                        2,
                    ),
                    "source": "zhitu.transaction",
                }
            )
        return rows

    async def fetch_spot(self) -> pd.DataFrame:
        async with self._spot_lock:
            if monotonic() >= self._spot_until:
                # The bulk endpoint allows about three requests per minute.
                self._spot = pd.DataFrame()
                self._spot_until = monotonic() + 30
                raw = pd.DataFrame(await self._items("/hs/public/realall"))
                if not raw.empty:
                    try:
                        names = await self.fetch_symbol_names()
                    except MarketDataSourceError:
                        names = self._names
                    spot = raw.rename(columns=_SPOT_COLUMNS).reindex(
                        columns=list(_SPOT_COLUMNS.values())
                    )
                    spot["代码"] = spot["代码"].fillna("").map(_normalize_symbol)
                    spot["名称"] = spot["代码"].map(names).fillna("")
                    spot["行情日期"] = raw.get(
                        "t", pd.Series(index=raw.index, dtype="str")
                    ).str[:10]
                    prices = ["最新价", "今开", "最高", "最低", "成交量", "成交额"]
                    spot[prices] = (
                        spot[prices].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                    )
                    self._spot = spot.loc[spot["代码"].ne("") & spot["最新价"].gt(0)]
            return self._spot.copy()

    async def fetch_hist(
        self,
        thscode: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        rights = {"": "n", "none": "n", "qfq": "f", "hfq": "b"}
        if adjust not in rights:
            return pd.DataFrame()
        market = "bj" if thscode.endswith(".BJ") else "hs"
        raw = pd.DataFrame(
            await self._items(
                f"/{market}/history/{thscode}/d/{rights[adjust]}",
                st=start_date.replace("-", ""),
                et=end_date.replace("-", ""),
            )
        )
        if raw.empty:
            return raw
        bars = raw.rename(
            columns={
                "t": "日期",
                "o": "开盘",
                "h": "最高",
                "l": "最低",
                "c": "收盘",
                "v": "成交量",
                "a": "成交额",
            }
        )
        return bars.dropna(subset=["日期", *_BAR_COLUMNS.values()])


def _date_ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="Asia/Shanghai").timestamp() * 1000)


def _read_dump(
    path: Path, events_path: Path | None, start: int, end: int
) -> pd.DataFrame:
    raw = pd.read_parquet(
        path,
        columns=["thscode", "date_ms", *_BAR_COLUMNS],
        filters=[("date_ms", ">=", start), ("date_ms", "<=", end)],
    )
    if raw.empty or raw["date_ms"].max() != end:
        return pd.DataFrame()
    if events_path is not None:
        # Bulk prices are raw. A later corporate action can change qfq even
        # when the action falls outside the requested history window.
        events = pd.read_parquet(
            events_path,
            columns=["thscode"],
            filters=[("ex_date_ms", ">", start)],
        )
        raw = raw.loc[~raw["thscode"].isin(events["thscode"])].copy()
    raw["代码"] = raw["thscode"].str.split(".", n=1).str[0]
    raw["日期"] = (
        pd.to_datetime(raw["date_ms"], unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%d")
    )
    raw["volume"] = raw["volume"] / 100.0  # Fuyao shares → Alpha lots.
    columns = ["代码", "日期", *_BAR_COLUMNS.values()]
    return raw.rename(columns=_BAR_COLUMNS)[columns].dropna()


class FuyaoMarketData:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._calendar = pd.DataFrame()
        self._calendar_day: date | None = None
        self._calendar_lock = asyncio.Lock()

    async def _get(
        self, client: httpx.AsyncClient, path: str, **params: Any
    ) -> dict[str, Any]:
        body = await _json(
            client,
            "fuyao",
            _FUYAO + path,
            params=params,
            headers={"X-api-key": self._api_key},
        )
        if (
            not isinstance(body, dict)
            or body.get("code") != 0
            or not isinstance(body.get("data"), dict)
        ):
            raise MarketDataSourceError("fuyao_invalid_response")
        return body["data"]

    async def fetch_trade_days(self, min_days: int = 0) -> pd.DataFrame:
        if min_days > 365:
            return pd.DataFrame()
        async with self._calendar_lock:
            today = now_cn().date()
            if self._calendar_day != today:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    data = await self._get(client, "/api/a-share/calendar/trading-days")
                dates = pd.to_datetime(
                    [row["date"] for row in data["item"]], format="%Y%m%d"
                )
                if dates.empty:
                    return pd.DataFrame()
                self._calendar = pd.DataFrame(
                    {"trade_date": dates.strftime("%Y-%m-%d")}
                )
                self._calendar_day = today
            return (
                self._calendar.copy()
                if len(self._calendar) >= min_days
                else pd.DataFrame()
            )

    async def fetch_limit_breaks(self, trade_date: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 1
        async with httpx.AsyncClient(timeout=8.0) as client:
            while True:
                data = await self._get(
                    client,
                    "/api/a-share/special-data/limit-break-pool",
                    date_ms=_date_ms(trade_date),
                    page=page,
                    size=200,
                )
                rows.extend(
                    {
                        "ts_code": row["thscode"],
                        "trade_date": trade_date,
                        "name": row.get("name"),
                        "limit": "Z",
                        # Fuyao publishes minute precision as HH:mm.
                        "first_time": str(row.get("first_seal_time") or "").replace(":", ""),
                        "last_time": str(row.get("last_seal_time") or "").replace(":", ""),
                        "open_times": row.get("open_times"),
                        "amount": row.get("turnover"),
                        "turnover_ratio": row.get("turnover_ratio_pct"),
                        "close": row.get("last_price"),
                        "pct_chg": row.get("price_change_ratio_pct"),
                        "source": "fuyao.limit_break_pool",
                    }
                    for row in data["item"]
                )
                pagination = data["pagination"]
                if page >= pagination["pages"]:
                    if len(rows) != pagination["total"]:
                        raise MarketDataSourceError("fuyao_incomplete_limit_breaks")
                    return rows
                page += 1

    async def fetch_index_daily(
        self, trade_date: str, codes: list[str]
    ) -> pd.DataFrame:
        day = pd.Timestamp(trade_date)
        end = _date_ms(day.date().isoformat())
        start = _date_ms((day - timedelta(days=14)).date().isoformat())
        async with httpx.AsyncClient(timeout=8.0) as client:
            data = await self._get(
                client,
                "/api/a-share-index/prices/historical-batch",
                thscodes=",".join(codes),
                interval="1d",
                start=start,
                end=end,
            )
        rows = []
        for group in data["item"]:
            bars = group["item"]
            if not bars or bars[-1]["date_ms"] != end:
                continue
            bar = bars[-1]
            prev = bars[-2]["close_price"] if len(bars) > 1 else None
            close = bar["close_price"]
            change = close - prev if prev and close else None
            rows.append(
                {
                    "ts_code": group["thscode"],
                    "trade_date": day.date().isoformat(),
                    "open": bar["open_price"],
                    "high": bar["high_price"],
                    "low": bar["low_price"],
                    "close": close,
                    "pre_close": prev,
                    "change": change,
                    "pct_chg": 100 * change / prev if change is not None else None,
                    "volume": bar["volume"] / 100.0,
                    "amount": bar["turnover"],
                }
            )
        if {row["ts_code"] for row in rows} != set(codes):
            return pd.DataFrame()
        return pd.DataFrame(rows)

    async def _download_dump(
        self, client: httpx.AsyncClient, kind: str, path: Path
    ) -> None:
        data = await self._get(client, f"/api/dump/market-dumps/{kind}/download-url")
        url = data.get("presigned_url", "")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "o.thsi.cn":
            raise MarketDataSourceError("fuyao_unexpected_download_origin")
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                size = 0
                with path.open("wb") as file:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        size += len(chunk)
                        if size > 512 * 1024 * 1024:
                            raise MarketDataSourceError("fuyao_dump_too_large")
                        file.write(chunk)
        except (httpx.HTTPError, OSError) as exc:
            raise MarketDataSourceError(
                f"fuyao_download_{type(exc).__name__}"
            ) from None

    async def fetch_market_history(
        self, start_date: str, end_date: str, *, adjusted: bool
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date).date()
        kind = (
            "daily-k-10d"
            if start >= now_cn().date() - timedelta(days=10)
            else "daily-k"
        )
        with TemporaryDirectory(prefix="alpha-fuyao-") as scratch:
            path = Path(scratch) / "daily.parquet"
            events_path = Path(scratch) / "actions.parquet" if adjusted else None
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0)
            ) as client:
                await self._download_dump(client, kind, path)
                if events_path is not None:
                    await self._download_dump(client, "adjustment-factors", events_path)
            reader = asyncio.create_task(
                asyncio.to_thread(
                    _read_dump,
                    path,
                    events_path,
                    _date_ms(start_date),
                    _date_ms(end_date),
                )
            )
            try:
                return await asyncio.shield(reader)
            except asyncio.CancelledError:
                await asyncio.gather(reader, return_exceptions=True)
                raise
