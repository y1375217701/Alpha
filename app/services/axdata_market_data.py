from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from app.services.market_data_sources import MarketDataSourceError
from app.services.time_utils import now_cn


class AxDataMarketData:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def request(self, interface: str, **params: Any) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(75, connect=3)
            ) as client:
                response = await client.post(
                    f"{self.base_url}/v1/request/{interface}", json={"params": params}
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise MarketDataSourceError(
                f"axdata_{interface}_http_{exc.response.status_code}"
            ) from None
        except (httpx.RequestError, ValueError) as exc:
            raise MarketDataSourceError(
                f"axdata_{interface}_{type(exc).__name__}"
            ) from None
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            data = data.get("rows", data.get("items"))
        if not isinstance(data, list):
            raise MarketDataSourceError(f"axdata_{interface}_invalid_response")
        return data

    async def prediction_boards(
        self, limit: int, trade_date: str
    ) -> tuple[pd.DataFrame, str]:
        try:
            rows = await self.request(
                "eastmoney_sector_realtime", sector_type="concept", limit=200
            )
            if rows:
                frame = pd.DataFrame(rows).rename(
                    columns={
                        "sector_code": "board_code",
                        "sector_name": "板块名称",
                        "change_pct": "涨跌幅",
                    }
                )
                frame["涨跌幅"] = pd.to_numeric(frame["涨跌幅"], errors="coerce")
                return frame.nlargest(limit, "涨跌幅"), "axdata_eastmoney"
        except MarketDataSourceError:
            pass
        return await self.historical_prediction_boards(limit, trade_date)

    async def historical_prediction_boards(
        self, limit: int, trade_date: str
    ) -> tuple[pd.DataFrame, str]:
        rows = await self.request(
            "kph_sector_ranking", trade_date=trade_date, sector_type="selected"
        )
        if not rows:
            return pd.DataFrame(), "axdata_kph"
        frame = pd.DataFrame(rows).rename(
            columns={
                "plate_id": "board_code",
                "plate_name": "板块名称",
                "change_pct": "涨跌幅",
            }
        )
        frame["涨跌幅"] = pd.to_numeric(frame["涨跌幅"], errors="coerce")
        return frame.nlargest(limit, "涨跌幅"), "axdata_kph"

    async def prediction_members(
        self,
        source: str,
        code: str,
        trade_date: str,
        limit: int,
    ) -> pd.DataFrame:
        if source == "axdata_kph":
            rows = await self.request(
                "kph_sector_constituents_history", plate_id=code, trade_date=trade_date
            )
        else:
            rows = await self.request(
                "eastmoney_sector_constituents",
                sector_code=code,
                limit=min(200, max(20, limit * 2)),
            )
        return pd.DataFrame(rows).rename(
            columns={
                "symbol": "代码",
                "name": "名称",
                "last_price": "最新价",
                "change_pct": "涨跌幅",
            }
        )

    async def valuation_history(
        self,
        codes: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        rows = await self.request(
            "stock_valuation_metrics_tdx",
            code=codes,
            start_date=start,
            end_date=end,
            count=(pd.Timestamp(end) - pd.Timestamp(start)).days + 1,
        )
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).rename(
            columns={
                "date": "trade_date",
                "pb_mrq": "pb",
                "float_market_cap": "circ_mv",
                "total_market_cap": "total_mv",
            }
        )
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"], format="%Y%m%d"
        ).dt.strftime("%Y-%m-%d")
        return frame[
            ["symbol", "trade_date", "pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv"]
        ]

    async def minute_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        rows = await self.request(
            "stock_kline_minute_tdx", code=symbol, period="1m", adjust="none"
        )
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        clock = pd.to_datetime(
            frame["trade_time"], utc=True, errors="coerce"
        ).dt.tz_convert("Asia/Shanghai")
        frame["bar_time"] = clock.dt.strftime("%Y-%m-%d %H:%M:%S")
        frame["trade_date"] = clock.dt.strftime("%Y-%m-%d")
        frame["source"] = "axdata.tdx_1m"
        return frame.loc[
            frame["trade_date"].between(start, end),
            [
                "symbol",
                "trade_date",
                "bar_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "source",
            ],
        ]

    async def price_limits(self, trade_date: str) -> pd.DataFrame:
        rows = await self.request("stock_daily_price_limit_tdx", scope="all")
        return pd.DataFrame(
            [
                {
                    "symbol": r["symbol"],
                    "trade_date": trade_date,
                    "up_limit": r.get("limit_up_price"),
                    "down_limit": r.get("limit_down_price"),
                }
                for r in rows
                if r["trade_date"].replace("-", "") == trade_date.replace("-", "")
                and r.get("limit_status") == "normal"
            ]
        )

    async def free_float_shares(self, trade_date: str) -> dict[str, float]:
        rows = await self.request("stock_daily_share_tdx", scope="all")
        target = trade_date.replace("-", "")
        return {
            row["symbol"]: float(row["free_float_share_z"])
            for row in rows
            if str(row.get("trade_date") or "").replace("-", "") == target
            and (row.get("free_float_share_z") or 0) > 0
        }

    async def limit_themes(self, trade_date: str) -> list[dict[str, Any]]:
        params = (
            {}
            if trade_date == now_cn().date().isoformat()
            else {"trade_date": trade_date.replace("-", "")}
        )
        rows = await self.request("kph_limit_resumption_history", **params)
        by_symbol: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not row.get("symbol") or str(row.get("trade_date", "")).replace(
                "-", ""
            ) != trade_date.replace("-", ""):
                continue
            item = by_symbol.setdefault(
                row["symbol"],
                {
                    "symbol": row["symbol"],
                    "trade_date": trade_date,
                    "board_type": row.get("limit_tag"),
                    "theme": "",
                    "limit_reason": row.get("reason_short") or row.get("reason"),
                },
            )
            themes = [
                *item["theme"].split("、"),
                *str(row.get("themes") or row.get("plate_name") or "").split("、"),
            ]
            item["theme"] = "、".join(
                dict.fromkeys(value.strip() for value in themes if value.strip())
            )
            item["limit_reason"] = row.get("reason_detail") or item["limit_reason"]
        return list(by_symbol.values())
