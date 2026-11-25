import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # headless backend for GitHub Actions
import matplotlib.pyplot as plt
from tradingview_screener import Query, col

CACHE_FILE = 'fallback_prices.csv'
CHART_FILE = 'monte_carlo_chart.png'
LOOKBACK_PERIOD = '3y'
BATCH_SIZE = 250

BACKUP_TICKERS = [
    'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'AVGO', 'LLY', 'BRK-B',
    'WMT', 'JPM', 'V', 'MU', 'AMD', 'JNJ', 'XOM', 'MA', 'ABBV', 'COST',
    'KO', 'CVX', 'PG', 'PLTR', 'CSCO', 'HD', 'MRK', 'BAC', 'ORCL', 'GE',
    'UNH', 'INTC', 'WFC', 'CAT', 'IBM', 'CRM', 'TMUS', 'PM', 'MCD', 'AXP',
    'LIN', 'NOW', 'QCOM', 'ABT', 'GS', 'RTX', 'TXN', 'UBER', 'ISRG', 'DIS'
]

LABELS = {'equal': 'Equal-Weighted (1/N)', 'market_cap': 'Market-Cap Weighted'}


def parse_args():
    parser = argparse.ArgumentParser(description="US Mega-Cap EMA50 Monte Carlo Simulator")
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial portfolio capital ($ USD)')
    parser.add_argument('--sims', type=int, default=10000, help='Number of Monte Carlo simulations')
    parser.add_argument('--days', type=int, default=252, help='Horizon in trading days (252 = 1 year)')
    parser.add_argument('--confidence', type=float, default=95.0, help='VaR confidence level (e.g., 95 or 99)')
    parser.add_argument('--weighting', type=str, default='both', choices=['both', 'equal', 'market_cap'],
                        help='Weighting strategy')
    parser.add_argument('--rf', type=float, default=0.0425, help='Annual risk-free rate as decimal (default: 0.0425)')
    args = parser.parse_args()

    if args.capital <= 0:
        parser.error("--capital must be positive")
    if args.sims < 100:
        parser.error("--sims must be at least 100")
    if args.days < 2:
        parser.error("--days must be at least 2")
    if not 50 < args.confidence < 100:
        parser.error("--confidence must be between 50 and 100 (e.g. 95)")
    if args.rf > 1:  # allow "4.25" to mean 4.25%
        args.rf = args.rf / 100.0
    return args


def fetch_top50_ema50_data():
    """Queries TradingView for Top 50 US Mega-Caps > EMA50 and returns tickers + market caps."""
    try:
        print("Querying TradingView screener (US Mega-Caps with Price > EMA50)...")
        _, tv_df = (
            Query()
            .select('name', 'close', 'EMA50', 'market_cap_basic')
            .where(
                col('exchange').isin(['NYSE', 'NASDAQ']),
                col('type') == 'stock',
                col('is_primary') == True,
                col('close') > col('EMA50')
            )
            .order_by('market_cap_basic', ascending=False)
            .limit(50)
            .get_scanner_data()
        )
        tv_df['yf_ticker'] = tv_df['ticker'].apply(lambda t: t.split(':')[-1].replace('.', '-'))
        mcap_map = dict(zip(tv_df['yf_ticker'], tv_df['market_cap_basic']))
        tickers = tv_df['yf_ticker'].tolist()
        if len(tickers) >= 10:
            print(f"Successfully retrieved {len(tickers)} tickers from TradingView.")
            return tickers, mcap_map
        print(f"[Warning] TradingView returned only {len(tickers)} tickers. Using backup list.")
    except Exception as e:
        print(f"[Warning] TradingView query failed ({e}). Using backup mega-cap list.")

    print("[Warning] Backup list has no market caps: market-cap weights will equal 1/N weights.")
    return BACKUP_TICKERS, {t: 1.0 for t in BACKUP_TICKERS}


def load_market_data(tickers):
    try:
        print(f"Downloading {LOOKBACK_PERIOD} of historical prices from Yahoo Finance...")
        raw_data = yf.download(
            tickers,
            period=LOOKBACK_PERIOD,
            auto_adjust=True,
            progress=False,
            threads=False
        )['Close']
        if isinstance(raw_data, pd.Series):
            raw_data = raw_data.to_frame()
        valid_data = raw_data.dropna(axis=1, thresh=int(len(raw_data) * 0.90)).ffill().dropna()

        if valid_data.shape[1] >= 10 and len(valid_data) > 100:
            valid_data.to_csv(CACHE_FILE)
            print(f"Cached clean dataset ({valid_data.shape[1]} stocks) to {CACHE_FILE}.")
            return valid_data
        print("[Warning] Downloaded data was insufficient.")
    except Exception as e:
        print(f"[Warning] yfinance download failed ({e}).")

    if os.path.exists(CACHE_FILE):
        print(f"Loading historical data from local fallback cache: {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)

    raise RuntimeError("Failed to fetch live data and no fallback CSV cache was found.")


def run_monte_carlo(price_df, mcap_map, args):
    tickers = price_df.columns.tolist()
    num_assets = len(tickers)

    equal_weights = np.ones(num_assets) / num_assets

    known = [float(v) for v in mcap_map.values() if v]
    default_mcap = float(np.median(known)) if known else 1.0
    raw_mcaps = np.array([float(mcap_map.get(t) or default_mcap) for t in tickers])
    mcap_weights = raw_mcaps / raw_mcaps.sum()

    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    mean_returns = log_returns.mean().values  # mean of LOG returns: no extra -0.5*sigma^2 needed
    cov_matrix = log_returns.cov().values + np.eye(num_assets) * 1e-8
    L = np.linalg.cholesky(cov_matrix)

    weights = {'equal': equal_weights, 'market_cap': mcap_weights}
    strategies = ['equal', 'market_cap'] if args.weighting == 'both' else [args.weighting]
    paths_dict = {s: np.zeros((args.days, args.sims)) for s in strategies}

    rng = np.random.default_rng(42)
    for start in range(0, args.sims, BATCH_SIZE):
        n = min(BATCH_SIZE, args.sims - start)
        Z = rng.standard_normal((n, args.days, num_assets))
        shocks = Z @ L.T                                  # (n, days, assets)
        daily_growth = np.exp(mean_returns + shocks)
        asset_paths = np.cumprod(daily_growth, axis=1)    # (n, days, assets)
        for s in strategies:
            paths_dict[s][:, start:start + n] = args.capital * (asset_paths @ weights[s]).T

    return paths_dict, tickers


def compute_metrics(paths, capital, confidence, rf_rate):
    ending_values = paths[-1, :]
    returns_pct = (ending_values - capital) / capital
    tail_percentile = 100.0 - confidence

    median_val = np.median(ending_values)
    median_return = ((median_val / capital) - 1) * 100
    var_cutoff = np.percentile(ending_values, tail_percentile)
    var_dollar = capital - var_cutoff
    var_pct = np.percentile(returns_pct, tail_percentile) * 100

    worst_tail = ending_values[ending_values <= var_cutoff]
    cvar_dollar = capital - worst_tail.mean()
    cvar_pct = ((worst_tail.mean() - capital) / capital) * 100
    prob_loss = np.mean(ending_values < capital) * 100

    full_paths = np.vstack([np.full((1, paths.shape[1]), capital), paths])
    daily_rets = (full_paths[1:] - full_paths[:-1]) / full_paths[:-1]
    daily_rf = rf_rate / 252.0

    excess_daily = daily_rets - daily_rf
    ann_return = np.mean(daily_rets, axis=0) * 252
    ann_vol = np.std(daily_rets, axis=0, ddof=1) * np.sqrt(252)
    sharpe_ratio = np.median((ann_return - rf_rate) / np.where(ann_vol == 0, 1e-8, ann_vol))

    downside_diff = np.minimum(0.0, excess_daily)
    downside_dev = np.sqrt(np.mean(downside_diff ** 2, axis=0)) * np.sqrt(252)
    sortino_ratio = np.median((ann_return - rf_rate) / np.where(downside_dev == 0, 1e-8, downside_dev))

    running_max = np.maximum.accumulate(full_paths, axis=0)
    drawdowns = (full_paths - running_max) / running_max
    median_max_dd = np.median(np.min(drawdowns, axis=0)) * 100

    return {
        'median_val': median_val,
        'median_return': median_return,
        'prob_loss': prob_loss,
        'var_cutoff': var_cutoff,
        'var_dollar': var_dollar,
        'var_pct': var_pct,
        'cvar_dollar': cvar_dollar,
        'cvar_pct': cvar_pct,
        'sharpe': sharpe_ratio,
        'sortino': sortino_ratio,
        'max_dd': median_max_dd
    }


def write_github_summary(metrics_map, active_tickers, args):
    summary_path = os.getenv('GITHUB_STEP_SUMMARY')
    if not summary_path:
        return
    conf = f"{args.confidence:.0f}%"
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"## 📊 Monte Carlo Risk Report ({len(active_tickers)} US Mega-Caps > EMA50)\n\n")
        f.write(f"**Parameters:** `${args.capital:,.2f}` Capital | `{args.days}` Days | "
                f"`{args.sims:,}` Runs | `{conf}` Confidence | `{args.rf * 100:.2f}%` Risk-Free Rate\n\n")

        keys = list(metrics_map.keys())
        header = "| Risk & Performance Metric | " + " | ".join(LABELS[k] for k in keys) + " |\n"
        sep = "| :--- | " + " | ".join(":---" for _ in keys) + " |\n"
        f.write(header)
        f.write(sep)

        rows = [
            ("Median Ending Value", lambda m: f"`${m['median_val']:,.2f}` (`{m['median_return']:+.2f}%`)"),
            ("Probability of Loss", lambda m: f"`{m['prob_loss']:.2f}%`"),
            (f"{conf} Value at Risk (VaR)", lambda m: f"`-${m['var_dollar']:,.2f}` (`{m['var_pct']:.2f}%`)"),
            (f"{conf} Conditional VaR (CVaR)", lambda m: f"`-${m['cvar_dollar']:,.2f}` (`{m['cvar_pct']:.2f}%`)"),
            ("Median Max Drawdown", lambda m: f"`{m['max_dd']:.2f}%`"),
            ("Annualized Sharpe Ratio", lambda m: f"`{m['sharpe']:.2f}`"),
            ("Annualized Sortino Ratio", lambda m: f"`{m['sortino']:.2f}`"),
        ]
        for name, fmt in rows:
            f.write(f"| **{name}** | " + " | ".join(fmt(metrics_map[k]) for k in keys) + " |\n")
        f.write("\n")


def plot_results(paths_dict, metrics_map, args):
    keys = list(paths_dict.keys())
    fig, axes = plt.subplots(2, len(keys), figsize=(8 * len(keys), 10), squeeze=False)
    x = np.arange(0, args.days + 1)
    tail = 100.0 - args.confidence

    for i, key in enumerate(keys):
        paths = paths_dict[key]
        full = np.vstack([np.full((1, paths.shape[1]), args.capital), paths])
        m = metrics_map[key]

        ax = axes[0, i]
        for j in range(min(50, full.shape[1])):
            ax.plot(x, full[:, j], color='steelblue', alpha=0.08, linewidth=0.8)
        p5, p25, p50, p75, p95 = np.percentile(full, [5, 25, 50, 75, 95], axis=1)
        ax.fill_between(x, p5, p95, color='steelblue', alpha=0.15, label='5th-95th pct')
        ax.fill_between(x, p25, p75, color='steelblue', alpha=0.25, label='25th-75th pct')
        ax.plot(x, p50, color='navy', linewidth=2, label='Median')
        ax.axhline(args.capital, color='gray', linestyle='--', linewidth=1, label='Initial capital')
        ax.set_title(f"{LABELS[key]}: Simulated Paths")
        ax.set_xlabel("Trading days")
        ax.set_ylabel("Portfolio value ($)")
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)

        ax2 = axes[1, i]
        ending = paths[-1, :]
        ax2.hist(ending, bins=80, color='steelblue', alpha=0.75)
        ax2.axvline(args.capital, color='gray', linestyle='--', label='Initial capital')
        ax2.axvline(m['median_val'], color='navy', linewidth=2, label=f"Median ${m['median_val']:,.0f}")
        ax2.axvline(m['var_cutoff'], color='red', linewidth=2,
                    label=f"{args.confidence:.0f}% VaR cutoff ${m['var_cutoff']:,.0f}")
        ax2.set_title(f"{LABELS[key]}: Ending Value Distribution")
        ax2.set_xlabel("Ending portfolio value ($)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(alpha=0.3)

    fig.suptitle(f"Monte Carlo: {args.sims:,} sims, {args.days} days, ${args.capital:,.0f} start", fontsize=14)
    fig.tight_layout()
    fig.savefig(CHART_FILE, dpi=130)
    plt.close(fig)
    print(f"Saved chart to {CHART_FILE}")


def generate_report(paths_dict, active_tickers, args):
    metrics_map = {k: compute_metrics(v, args.capital, args.confidence, args.rf) for k, v in paths_dict.items()}

    print("\n" + "=" * 72)
    print(f" MONTE CARLO RISK COMPARISON ({len(active_tickers)} US Mega-Caps > EMA50)")
    print("=" * 72)
    for key, m in metrics_map.items():
        print(f"[{LABELS[key]}]")
        print(f"  Median Value: ${m['median_val']:,.2f} ({m['median_return']:+.2f}%) | Loss Prob: {m['prob_loss']:.2f}%")
        print(f"  {args.confidence:.0f}% VaR: -${m['var_dollar']:,.2f} ({m['var_pct']:.2f}%) | "
              f"{args.confidence:.0f}% CVaR: -${m['cvar_dollar']:,.2f} ({m['cvar_pct']:.2f}%)")
        print(f"  Sharpe: {m['sharpe']:.2f} | Sortino: {m['sortino']:.2f} | Median Max Drawdown: {m['max_dd']:.2f}%")
    print("=" * 72)

    write_github_summary(metrics_map, active_tickers, args)
    plot_results(paths_dict, metrics_map, args)


def main():
    args = parse_args()
    tickers, mcap_map = fetch_top50_ema50_data()
    price_df = load_market_data(tickers)
    paths_dict, active_tickers = run_monte_carlo(price_df, mcap_map, args)
    generate_report(paths_dict, active_tickers, args)


if __name__ == "__main__":
    main()
