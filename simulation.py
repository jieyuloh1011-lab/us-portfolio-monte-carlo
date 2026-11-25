import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tradingview_screener import Query, col

CACHE_FILE = 'fallback_prices.csv'
LOOKBACK_PERIOD = '3y'

BACKUP_TICKERS = [
    'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'AVGO', 'LLY', 'BRK-B',
    'WMT', 'JPM', 'V', 'MU', 'AMD', 'JNJ', 'XOM', 'MA', 'ABBV', 'COST',
    'KO', 'CVX', 'PG', 'PLTR', 'CSCO', 'HD', 'MRK', 'BAC', 'ORCL', 'GE',
    'UNH', 'INTC', 'WFC', 'CAT', 'IBM', 'CRM', 'TMUS', 'PM', 'MCD', 'AXP',
    'LIN', 'NOW', 'QCOM', 'ABT', 'GS', 'RTX', 'TXN', 'UBER', 'ISRG', 'DIS'
]

def parse_args():
    parser = argparse.ArgumentParser(description="US Mega-Cap EMA50 Monte Carlo Simulator")
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial portfolio capital ($ USD)')
    parser.add_argument('--sims', type=int, default=10000, help='Number of Monte Carlo simulations')
    parser.add_argument('--days', type=int, default=252, help='Horizon in trading days (252 = 1 year)')
    parser.add_argument('--confidence', type=float, default=95.0, help='VaR confidence level (e.g., 95 or 99)')
    return parser.parse_args()

def fetch_top50_ema50_tickers():
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
        tickers = [t.split(':')[-1].replace('.', '-') for t in tv_df['ticker']]
        if len(tickers) >= 10:
            print(f"Successfully retrieved {len(tickers)} tickers from TradingView.")
            return tickers
    except Exception as e:
        print(f"[Warning] TradingView query failed ({e}). Using backup mega-cap list.")
    return BACKUP_TICKERS

def load_market_data(tickers):
    try:
        print(f"Downloading {LOOKBACK_PERIOD} of historical prices from Yahoo Finance...")
        # threads=False prevents SQLite 'database is locked' errors on GitHub Actions
        raw_data = yf.download(
            tickers, 
            period=LOOKBACK_PERIOD, 
            auto_adjust=True, 
            progress=False, 
            threads=False
        )['Close']
        
        valid_data = raw_data.dropna(axis=1, thresh=int(len(raw_data) * 0.90)).ffill().dropna()
        
        # Fixed: check valid_data.shape[1] (number of stocks) instead of valid_data.shape
        if valid_data.shape[1] >= 10 and len(valid_data) > 100:
            valid_data.to_csv(CACHE_FILE)
            print(f"Cached clean dataset ({valid_data.shape[1]} stocks, {len(valid_data)} days) to {CACHE_FILE}.")
            return valid_data
    except Exception as e:
        print(f"[Warning] yfinance download failed ({e}).")

    if os.path.exists(CACHE_FILE):
        print(f"Loading historical data from local fallback cache: {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    
    raise RuntimeError("Failed to fetch live data and no fallback CSV cache was found.")



def run_monte_carlo(price_df, initial_capital, num_sims, horizon_days):
    tickers = price_df.columns.tolist()
    num_assets = len(tickers)
    weights = np.ones(num_assets) / num_assets

    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values

    cov_matrix += np.eye(num_assets) * 1e-8
    L = np.linalg.cholesky(cov_matrix)

    drift = mean_returns - 0.5 * np.diag(cov_matrix)
    portfolio_paths = np.zeros((horizon_days, num_sims))

    np.random.seed(42)
    for sim in range(num_sims):
        Z = np.random.normal(size=(horizon_days, num_assets))
        correlated_shocks = Z @ L.T
        daily_returns = np.exp(drift + correlated_shocks)
        asset_paths = np.cumprod(daily_returns, axis=0)
        portfolio_paths[:, sim] = initial_capital * (asset_paths @ weights)

    return portfolio_paths, tickers

def generate_report(portfolio_paths, active_tickers, args):
    ending_values = portfolio_paths[-1, :]
    returns_pct = (ending_values - args.capital) / args.capital
    tail_percentile = 100.0 - args.confidence

    median_val = np.median(ending_values)
    var_cutoff = np.percentile(ending_values, tail_percentile)
    var_dollar = args.capital - var_cutoff
    var_pct = np.percentile(returns_pct, tail_percentile) * 100
    
    worst_tail = ending_values[ending_values <= var_cutoff]
    cvar_dollar = args.capital - worst_tail.mean()
    prob_loss = np.mean(ending_values < args.capital) * 100

    print("\n" + "=" * 54)
    print(f" MONTE CARLO RISK REPORT ({len(active_tickers)} US Mega-Caps > EMA50)")
    print("=" * 54)
    print(f"Initial Capital:          ${args.capital:,.2f}")
    print(f"Horizon / Simulations:    {args.days} days / {args.sims:,} runs")
    print(f"Median Ending Value:      ${median_val:,.2f} ({((median_val/args.capital)-1)*100:+.2f}%)")
    print(f"Probability of Loss:      {prob_loss:.2f}%")
    print(f"{args.confidence:.0f}% VaR (Max Loss):       ${var_dollar:,.2f} ({var_pct:.2f}%)")
    print(f"{args.confidence:.0f}% CVaR (Tail Avg):      ${cvar_dollar:,.2f}")
    print("=" * 54)

    plt.figure(figsize=(11, 5.5))
    plt.plot(portfolio_paths[:, :150], color='#4A90E2', alpha=0.12)
    plt.plot(np.median(portfolio_paths, axis=1), color='#FF8C00', linewidth=2.2, label='Median Outcome (50th Pct)')
    plt.plot(np.percentile(portfolio_paths, 95, axis=1), color='#2ECC71', linestyle='--', linewidth=1.8, label='95th Percentile (Bull Case)')
    plt.plot(np.percentile(portfolio_paths, tail_percentile, axis=1), color='#E74C3C', linestyle='--', linewidth=2.0, label=f'{tail_percentile:.0f}th Percentile ({args.confidence:.0f}% VaR)')
    plt.axhline(args.capital, color='#2C3E50', linestyle=':', linewidth=1.5, label=f'Initial Capital (${args.capital:,.0f})')

    plt.title(f'Monte Carlo Simulation: Top {len(active_tickers)} US Mega-Caps > EMA50 ({args.sims:,} Runs, {args.days} Days)')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($ USD)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('monte_carlo_chart.png', dpi=300)
    print("Saved visualization to monte_carlo_chart.png")

if __name__ == '__main__':
    args = parse_args()
    tickers = fetch_top50_ema50_tickers()
    prices = load_market_data(tickers)
    paths, final_tickers = run_monte_carlo(prices, args.capital, args.sims, args.days)
    generate_report(paths, final_tickers, args)

