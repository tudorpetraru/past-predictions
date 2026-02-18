# Past Predictions Pipeline

Yahoo-first analyst target reconstruction for the union of current S&P 500 and Nasdaq-100 constituents, with FMP fallback when Yahoo has zero valid target events for a ticker.

## What it builds

A deterministic weekly backtest dataset with:

- `ticker`
- `date`
- `predicted_min`
- `predicted_avg`
- `predicted_max`
- `actual`
- `actual_12m`
- `error_avg_12m`
- `hit_within_range_12m`
- `n_analysts`
- `data_quality_flags`

Default window is `2024-02-18` to `2026-02-18`.

## Install

```bash
python3 -m pip install -e .
```

Optional dev/test deps:

```bash
python3 -m pip install -e '.[dev]'
```

## Run end-to-end

1. Build universe:

```bash
past-predictions build-universe --sources wikipedia --out data/universe/
```

2. Fetch target events (Yahoo first, FMP fallback):

```bash
past-predictions fetch-target-events \
  --universe data/universe/universe_union.csv \
  --provider-priority yahoo,fmp \
  --api-key ENV:FMP_KEY \
  --cache data/raw/ \
  --workers 8
```

3. Fetch prices and splits:

```bash
past-predictions fetch-prices \
  --universe data/universe/universe_union.csv \
  --provider yfinance \
  --out data/raw/prices/ \
  --start 2024-02-18 \
  --end 2026-02-18
```

4. Compute weekly dataset:

```bash
past-predictions compute-weekly \
  --start 2024-02-18 \
  --end 2026-02-18 \
  --ttl-days 365 \
  --min-analysts 3 \
  --calendar XNYS
```

5. Spot-check report:

```bash
past-predictions spot-check-report \
  --tickers AAPL,MSFT,NVDA,AMZN,META \
  --weeks 10 \
  --out data/derived/spot_check.md
```

6. Export final CSV:

```bash
past-predictions export-csv \
  --out exports/targets_weekly_backtest_2024-02-18_2026-02-18.csv
```

## FMP fallback planning (free tier)

```bash
past-predictions plan-fmp-fallback \
  --universe data/universe/universe_union.csv \
  --daily-limit 250 \
  --reserve-calls 10 \
  --out data/plans/fmp_fallback_batches.csv
```

## Storage layout

- `data/universe/sp500.csv`
- `data/universe/nasdaq100.csv`
- `data/universe/universe_union.csv`
- `data/universe/metadata.json`
- `data/raw/yahoo_target_events/{ticker}.parquet`
- `data/raw/fmp_price_target/{ticker}.json`
- `data/raw/prices/{ticker}.parquet`
- `data/raw/splits/{ticker}.parquet`
- `data/derived/provider_selection.csv`
- `data/derived/target_events.parquet`
- `data/derived/weekly_targets.parquet`
- `data/derived/run_manifest.json`
- `exports/targets_weekly_backtest_{start}_{end}.csv`

## Data quality flags

- `NO_TARGETS`
- `LOW_COVERAGE`
- `NO_PRICE`
- `NO_12M_PRICE`
- `SPLIT_ADJUSTED`
- `SOURCE_YAHOO`
- `SOURCE_FMP_FALLBACK`

## Caveats

- **Survivorship bias**: universe uses current constituents, not historical index membership.
- **Coverage bias**: analyst target event density varies by ticker.
- **Target semantics**: assumed 12-month horizon with TTL 365 days.
- **Vendor dependency**: Yahoo schema can change; FMP fallback depends on API key and free-tier limits.
- **Consensus reconstruction**: weekly consensus is reconstructed from event streams, not downloaded as native weekly historical consensus snapshots.

## Sources

- [S&P 500 constituents](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- [Nasdaq-100 constituents](https://en.wikipedia.org/wiki/Nasdaq-100)
- [FMP price target API](https://site.financialmodelingprep.com/developer/docs/price-target-api)
