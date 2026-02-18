from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .analyst_panel import compute_analyst_weekly_dataset
from .config import config_hash, ensure_dir, load_config, parse_date, resolve_api_key
from .consensus import compute_weekly_dataset
from .export import export_analyst_csv, export_csv as export_weekly_csv
from .fmp_client import FMPRequestConfig, fetch_fmp_price_targets
from .prices import fetch_prices_for_universe
from .reporting import generate_spot_check_report
from .universe import build_universe
from .yahoo_events import choose_provider, empty_events, fetch_yahoo_events

app = typer.Typer(help="Weekly analyst target reconstruction pipeline")


def _read_universe(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"ticker_norm", "ticker_fmp", "ticker_yahoo"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Universe file missing required columns: {sorted(missing)}")
    return frame


def _resolve_fmp_key(api_key: Optional[str], config_data: dict) -> Optional[str]:
    direct = resolve_api_key(api_key)
    if direct:
        return direct
    env_name = config_data.get("providers", {}).get("fmp", {}).get("api_key_env")
    if env_name:
        return resolve_api_key(f"ENV:{env_name}")
    return None


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out
    except Exception:
        return None


@app.command("build-universe")
def build_universe_cmd(
    sources: str = typer.Option("wikipedia", help="Universe source set"),
    out: str = typer.Option("data/universe/", help="Output directory"),
) -> None:
    if sources.lower() != "wikipedia":
        raise typer.BadParameter("Only wikipedia source is currently supported")
    union = build_universe(out)
    typer.echo(f"Built universe with {union['ticker_norm'].nunique()} unique tickers at {out}")


@app.command("plan-fmp-fallback")
def plan_fmp_fallback_cmd(
    universe: str = typer.Option(..., help="Path to universe_union.csv"),
    daily_limit: int = typer.Option(250, help="Daily call limit"),
    reserve_calls: int = typer.Option(10, help="Reserved calls per day"),
    out: str = typer.Option("data/plans/fmp_fallback_batches.csv", help="Output CSV path"),
    yahoo_dir: str = typer.Option("data/raw/yahoo_target_events", help="Yahoo cache dir"),
) -> None:
    universe_df = _read_universe(universe)
    capacity = max(1, daily_limit - reserve_calls)

    rows = []
    fallback_tickers = []
    for ticker in sorted(universe_df["ticker_norm"].astype(str).unique()):
        path = Path(yahoo_dir) / f"{ticker}.parquet"
        valid_count = 0
        if path.exists():
            try:
                ydf = pd.read_parquet(path)
                valid_count = int(len(ydf))
            except Exception:
                valid_count = 0
        if valid_count == 0:
            fallback_tickers.append(ticker)

    for idx, ticker in enumerate(fallback_tickers):
        rows.append(
            {
                "ticker": ticker,
                "batch_id": int(idx // capacity) + 1,
                "position": idx + 1,
                "daily_capacity": capacity,
            }
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    typer.echo(f"Planned {len(rows)} fallback tickers across {max((len(rows)-1)//capacity + 1, 0)} batch(es)")


@app.command("fetch-target-events")
def fetch_target_events_cmd(
    universe: str = typer.Option(..., help="Path to universe_union.csv"),
    provider_priority: str = typer.Option("yahoo,fmp", help="Ordered provider list"),
    api_key: Optional[str] = typer.Option(None, help="FMP key or ENV:VARNAME"),
    cache: str = typer.Option("data/raw/", help="Raw cache root"),
    workers: int = typer.Option(8, help="Thread workers"),
    force_refresh: bool = typer.Option(False, help="Ignore cache and refresh"),
    config: str = typer.Option("config.yaml", help="Config path"),
) -> None:
    config_data = load_config(config)
    universe_df = _read_universe(universe)

    priorities = [p.strip().lower() for p in provider_priority.split(",") if p.strip()]
    if not priorities:
        raise typer.BadParameter("provider-priority must include at least one provider")

    raw_root = Path(cache)
    yahoo_dir = ensure_dir(raw_root / "yahoo_target_events")
    fmp_dir = ensure_dir(raw_root / "fmp_price_target")
    derived_dir = ensure_dir("data/derived")

    fmp_key = _resolve_fmp_key(api_key=api_key, config_data=config_data)
    fmp_cfg = None
    if "fmp" in priorities and fmp_key:
        fmp_defaults = config_data.get("providers", {}).get("fmp", {})
        fmp_cfg = FMPRequestConfig(
            base_url=fmp_defaults.get("base_url", "https://financialmodelingprep.com"),
            api_key=fmp_key,
            timeout_seconds=int(fmp_defaults.get("timeout_seconds", 25)),
            max_retries=int(fmp_defaults.get("max_retries", 5)),
            backoff_seconds=float(fmp_defaults.get("backoff_seconds", 1.0)),
        )

    universe_records = universe_df[["ticker_norm", "ticker_fmp", "ticker_yahoo"]].to_dict("records")

    def process_one(record: dict) -> tuple[dict, pd.DataFrame]:
        ticker = str(record["ticker_norm"])
        ticker_yahoo = str(record["ticker_yahoo"])
        ticker_fmp = str(record["ticker_fmp"])

        yahoo_events = empty_events()
        fmp_events = empty_events()
        errors = []

        if "yahoo" in priorities:
            try:
                yahoo_events = fetch_yahoo_events(
                    ticker=ticker_yahoo,
                    raw_dir=yahoo_dir,
                    force_refresh=force_refresh,
                )
                if not yahoo_events.empty:
                    yahoo_events = yahoo_events.copy()
                    yahoo_events["ticker"] = ticker
            except Exception as exc:
                errors.append(f"yahoo:{exc}")

        if len(yahoo_events) == 0 and "fmp" in priorities and fmp_cfg is not None:
            try:
                fmp_events = fetch_fmp_price_targets(
                    ticker_fmp=ticker_fmp,
                    ticker_norm=ticker,
                    raw_dir=fmp_dir,
                    cfg=fmp_cfg,
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                errors.append(f"fmp:{exc}")

        provider = choose_provider(len(yahoo_events), len(fmp_events))
        selected = yahoo_events if provider == "yahoo" else fmp_events if provider == "fmp" else empty_events()

        status = "ok" if provider != "none" else "no_events"
        if errors and provider == "none":
            status = "error"

        row = {
            "ticker": ticker,
            "provider": provider,
            "yahoo_valid_events": int(len(yahoo_events)),
            "fmp_valid_events": int(len(fmp_events)),
            "selected_events": int(len(selected)),
            "status": status,
            "errors": " | ".join(errors),
        }
        return row, selected

    summaries = []
    selected_frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, r) for r in universe_records]
        for future in as_completed(futures):
            summary, selected = future.result()
            summaries.append(summary)
            if not selected.empty:
                selected_frames.append(selected)

    provider_df = pd.DataFrame(summaries).sort_values("ticker").reset_index(drop=True)
    provider_df.to_csv(derived_dir / "provider_selection.csv", index=False)

    if selected_frames:
        events_df = pd.concat(selected_frames, ignore_index=True).sort_values(["ticker", "event_ts"])
    else:
        events_df = empty_events()
    events_df.to_parquet(derived_dir / "target_events.parquet", index=False)

    failures = provider_df[provider_df["status"] == "error"]
    if not failures.empty:
        failures.to_csv(derived_dir / "target_event_errors.csv", index=False)

    typer.echo(
        f"Target fetch complete: {len(provider_df)} tickers, "
        f"{(provider_df['provider'] == 'yahoo').sum()} yahoo, "
        f"{(provider_df['provider'] == 'fmp').sum()} fmp fallback"
    )


@app.command("fetch-prices")
def fetch_prices_cmd(
    universe: str = typer.Option(..., help="Path to universe_union.csv"),
    provider: str = typer.Option("yfinance", help="Price provider"),
    out: str = typer.Option("data/raw/prices/", help="Price output dir"),
    start: str = typer.Option("2024-02-18", help="Start date"),
    end: str = typer.Option("2026-02-18", help="End date"),
    workers: int = typer.Option(8, help="Thread workers"),
) -> None:
    if provider.lower() != "yfinance":
        raise typer.BadParameter("Only yfinance provider is currently supported")

    universe_df = _read_universe(universe)
    tickers = sorted(universe_df["ticker_norm"].astype(str).unique())

    price_dir = Path(out)
    split_dir = Path("data/raw/splits")

    results = fetch_prices_for_universe(
        tickers=tickers,
        out_price_dir=price_dir,
        out_split_dir=split_dir,
        start=parse_date(start),
        end=parse_date(end),
        workers=workers,
    )

    report = pd.DataFrame([r.__dict__ for r in results])
    ensure_dir("data/derived")
    report.to_csv("data/derived/price_fetch_report.csv", index=False)
    typer.echo(f"Fetched prices for {len(results)} tickers; errors={(report['status'] == 'error').sum()}")


@app.command("compute-weekly")
def compute_weekly_cmd(
    start: str = typer.Option("2024-02-18", help="Backtest start date"),
    end: str = typer.Option("2026-02-18", help="Backtest end date"),
    ttl_days: int = typer.Option(365, help="Target TTL"),
    min_analysts: int = typer.Option(3, help="Minimum analyst threshold"),
    calendar: str = typer.Option("XNYS", help="Trading calendar"),
    horizon_days: int = typer.Option(252, help="Horizon in trading days"),
    universe: str = typer.Option("data/universe/universe_union.csv", help="Universe CSV path"),
    events: str = typer.Option("data/derived/target_events.parquet", help="Selected events parquet"),
    provider_selection: str = typer.Option("data/derived/provider_selection.csv", help="Provider selection CSV"),
    price_dir: str = typer.Option("data/raw/prices", help="Raw prices directory"),
    split_dir: str = typer.Option("data/raw/splits", help="Raw splits directory"),
    out: str = typer.Option("data/derived/weekly_targets.parquet", help="Output parquet"),
) -> None:
    universe_df = _read_universe(universe)
    events_df = pd.read_parquet(events) if Path(events).exists() else pd.DataFrame()
    provider_df = pd.read_csv(provider_selection) if Path(provider_selection).exists() else pd.DataFrame()

    weekly = compute_weekly_dataset(
        universe=universe_df,
        events=events_df,
        provider_selection=provider_df,
        price_dir=price_dir,
        split_dir=split_dir,
        start=parse_date(start),
        end=parse_date(end),
        ttl_days=ttl_days,
        min_analysts=min_analysts,
        calendar=calendar,
        horizon_days=horizon_days,
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(out_path, index=False)

    manifest_payload = {
        "computed_at": date.today().isoformat(),
        "start": start,
        "end": end,
        "ttl_days": ttl_days,
        "min_analysts": min_analysts,
        "calendar": calendar,
        "horizon_days": horizon_days,
        "rows": int(len(weekly)),
        "git_commit": _git_commit(),
    }
    manifest_payload["config_hash"] = config_hash(manifest_payload)
    Path("data/derived/run_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )

    typer.echo(f"Computed weekly dataset with {len(weekly)} rows")


@app.command("compute-analyst-weekly")
def compute_analyst_weekly_cmd(
    pred_start: str = typer.Option("2023-02-18", help="Prediction window start date"),
    pred_end: str = typer.Option("2025-02-18", help="Prediction window end date"),
    actual_start: str = typer.Option("2024-02-18", help="Actual window start date"),
    actual_end: str = typer.Option("2026-02-18", help="Actual window end date"),
    ttl_days: int = typer.Option(365, help="Target TTL"),
    calendar: str = typer.Option("XNYS", help="Trading calendar"),
    horizon_days: int = typer.Option(252, help="Horizon in trading days"),
    universe: str = typer.Option("data/universe/universe_union.csv", help="Universe CSV path"),
    events: str = typer.Option("data/derived/target_events.parquet", help="Selected events parquet"),
    provider_selection: str = typer.Option("data/derived/provider_selection.csv", help="Provider selection CSV"),
    price_dir: str = typer.Option("data/raw/prices", help="Raw prices directory"),
    split_dir: str = typer.Option("data/raw/splits", help="Raw splits directory"),
    out: str = typer.Option("data/derived/analyst_weekly_targets.parquet", help="Output parquet"),
    manifest_out: str = typer.Option("data/derived/analyst_run_manifest.json", help="Run manifest output path"),
) -> None:
    universe_df = _read_universe(universe)
    events_df = pd.read_parquet(events) if Path(events).exists() else pd.DataFrame()
    provider_df = pd.read_csv(provider_selection) if Path(provider_selection).exists() else pd.DataFrame()

    analyst_weekly = compute_analyst_weekly_dataset(
        universe=universe_df,
        events=events_df,
        provider_selection=provider_df,
        price_dir=price_dir,
        split_dir=split_dir,
        pred_start=parse_date(pred_start),
        pred_end=parse_date(pred_end),
        actual_start=parse_date(actual_start),
        actual_end=parse_date(actual_end),
        ttl_days=ttl_days,
        calendar=calendar,
        horizon_days=horizon_days,
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    analyst_weekly.to_parquet(out_path, index=False)

    manifest_payload = {
        "computed_at": date.today().isoformat(),
        "pred_start": pred_start,
        "pred_end": pred_end,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "ttl_days": ttl_days,
        "calendar": calendar,
        "horizon_days": horizon_days,
        "rows": int(len(analyst_weekly)),
        "git_commit": _git_commit(),
    }
    manifest_payload["config_hash"] = config_hash(manifest_payload)
    manifest_path = Path(manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    typer.echo(f"Computed analyst weekly dataset with {len(analyst_weekly)} rows")


@app.command("spot-check-report")
def spot_check_report_cmd(
    tickers: str = typer.Option(..., help="Comma-separated tickers"),
    weeks: int = typer.Option(10, help="Number of weeks per ticker"),
    source: str = typer.Option("data/derived/weekly_targets.parquet", help="Weekly parquet source"),
    out: str = typer.Option("data/derived/spot_check.md", help="Markdown output path"),
) -> None:
    frame = pd.read_parquet(source)
    wanted = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    generate_spot_check_report(frame, wanted, weeks=weeks, out_path=out)
    typer.echo(f"Spot-check report written to {out}")


@app.command("export-csv")
def export_csv_cmd(
    out: str = typer.Option(..., help="Output CSV path"),
    source: str = typer.Option("data/derived/weekly_targets.parquet", help="Weekly parquet source"),
    required_only: bool = typer.Option(False, help="Export only required columns"),
) -> None:
    output = export_weekly_csv(source, out, required_only=required_only)
    typer.echo(f"Exported {len(output)} rows to {out}")


@app.command("export-analyst-csv")
def export_analyst_csv_cmd(
    out: str = typer.Option(
        "exports/analyst_targets_backtest_2023-02-18_2025-02-18.csv",
        help="Output CSV path",
    ),
    source: str = typer.Option(
        "data/derived/analyst_weekly_targets.parquet",
        help="Analyst weekly parquet source",
    ),
) -> None:
    output = export_analyst_csv(source, out)
    typer.echo(f"Exported {len(output)} analyst rows to {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
