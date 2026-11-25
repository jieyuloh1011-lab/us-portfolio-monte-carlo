import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tradingview_screener import Query, col

# =========================================================
# 1. CONFIGURATION
# =========================================================
INITIAL_CAPITAL = 100_000   # $100,000 USD starting portfolio
NUM_SIMULATIONS = 10_000    # Number of Monte Carlo scenarios
HORIZON_DAYS = 252          # 252 trading days = 1 Year
LOOKBACK_PERIOD = '3y'      # Historical window for covariance & drift
CACHE_FILE = 'fallback_prices.csv'

# Backup list in case TradingView's screener endpoint is temporarily unreachable
BACKUP_TICKERS = [
    'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'AVGO', 'LLY', 'BRK-B',
    'WMT', 'JPM', 'V', 'MU', 'AMD', 'JNJ', 'XOM', 'MA', 'ABBV', 'COST',
    'KO', 'CVX', 'PG', 'PLTR', 'CSCO', 'HD', 'MRK', 'BAC', 'ORCL', 'GE',
    'UNH', 'INTC', 'WFC', 'CAT', 'IBM', 'CRM', 'TMUS', 'PM', 'MCD', 'AXP',
    'LIN', 'NOW', 'QCOM', 'ABT', 'GS', 'RTX', 'TXN', 'UBER', 'ISRG', 'DIS'
]

# =========================================================
# 2. DATA LOADER (TradingView Screener + yfinance + Fallback)
# =========================================================
def fetch_top50_ema50_tickers():
    """Queries TradingView for Top 50 US Mega-Caps trading above their 50-day EMA."""
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
    """Downloads historical prices via yfinance with automatic CSV caching/fallback."""
    try:
        print(f"Downloading {LOOKBACK_PERIOD} of historical prices from Yahoo Finance...")
        raw_data = yf.download(tickers, period=LOOKBACK_PERIOD, auto_adjust=True, progress=False)['Close']
        
        # Keep tickers with at least 90% valid trading days to avoid newly listed gaps
        valid_data = raw_data.dropna(axis=1, thresh=int(len(raw_data) * 0.90)).ffill().dropna()
        
        if valid_data.shape[1] >= 10 and len(valid_data) > 100:
            valid_data.to_csv(CACHE_FILE)
            print(f"Cached clean dataset ({valid_data.shape[1]} stocks, {len(valid_data)} days) to {CACHE_FILE}.")
            return valid_data
    except Exception as e:
        print(f"[Warning] yfinance download failed ({e}).")

    # Fallback to local CSV if APIs are down or rate-limited
    if os.path.exists(CACHE_FILE):
        print(f"Loading historical data from local fallback cache: {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    
    raise RuntimeError("Failed to fetch live data and no fallback CSV cache was found.")

# =========================================================
# 3. MONTE CARLO ENGINE (Cholesky Correlated GBM)
# =========================================================
def run_monte_carlo(price_df):
    tickers = price_df.columns.tolist()
    num_assets = len(tickers)
    weights = np.ones(num_assets) / num_assets  # Equal-weight (1/N) across filtered basket

    # Calculate daily log returns, mean drift, and covariance matrix
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values

    # Regularize covariance matrix slightly to guarantee positive-definiteness for 50 assets
    cov_matrix += np.eye(num_assets) * 1e-8
    L = np.linalg.cholesky(cov_matrix)

    drift = mean_returns - 0.5 * np.diag(cov_matrix)
    portfolio_paths = np.zeros((HORIZON_DAYS, NUM_SIMULATIONS))

    np.random.seed(42)  # Reproducible results for GitHub README
    for sim in range(NUM_SIMULATIONS):
        Z = np.random.normal(size=(HORIZON_DAYS, num_assets))
        correlated_shocks = Z @ L.T
        daily_returns = np.exp(drift + correlated_shocks)
        asset_paths = np.cumprod(daily_returns, axis=0)
        portfolio_paths[:, sim] = INITIAL_CAPITAL * (asset_paths @ weights)

    return portfolio_paths, tickers

# =========================================================
# 4. RISK METRICS & VISUALIZATION
# =========================================================
def generate_report(portfolio_paths, active_tickers):
    ending_values = portfolio_paths[-1, :]
    returns_pct = (ending_values - INITIAL_CAPITAL) / INITIAL_CAPITAL

    median_val = np.median(ending_values)
    var_95_dollar = INITIAL_CAPITAL - np.percentile(ending_values, 5)
    var_95_pct = np.percentile(returns_pct, 5) * 100
    
    worst_5_pct = ending_values[ending_values <= np.percentile(ending_values, 5)]
    cvar_95_dollar = INITIAL_CAPITAL - worst_5_pct.mean()
    prob_loss = np.mean(ending_values < INITIAL_CAPITAL) * 100

    print("\n" + "=" * 48)
    print(f" MONTE CARLO RISK REPORT ({len(active_tickers)} US Mega-Caps > EMA50)")
    print("=" * 48)
    print(f"Initial Capital:       ${INITIAL_CAPITAL:,.2f}")
    print(f"Median 1-Yr Value:     ${median_val:,.2f} ({((median_val/INITIAL_CAPITAL)-1)*100:+.2f}%)")
    print(f"Probability of Loss:   {prob_loss:.2f}%")
    print(f"95% VaR (Max Loss):    ${var_95_dollar:,.2f} ({var_95_pct:.2f}%)")
    print(f"95% CVaR (Tail Avg):   ${cvar_95_dollar:,.2f}")
    print("=" * 48)

    # Plot simulation fan chart
    plt.figure(figsize=(11, 5.5))
    plt.plot(portfolio_paths[:, :150], color='#4A90E2', alpha=0.12)
    plt.plot(np.median(portfolio_paths, axis=1), color='#FF8C00', linewidth=2.2, label='Median Outcome (50th Pct)')
    plt.plot(np.percentile(portfolio_paths, 95, axis=1), color='#2ECC71', linestyle='--', linewidth=1.8, label='95th Percentile (Bull Case)')
    plt.plot(np.percentile(portfolio_paths, 5, axis=1), color='#E74C3C', linestyle='--', linewidth=2.0, label='5th Percentile (95% VaR)')
    plt.axhline(INITIAL_CAPITAL, color='#2C3E50', linestyle=':', linewidth=1.5, label=f'Initial Capital (${INITIAL_CAPITAL:,.0f})')

    plt.title(f'1-Year Monte Carlo Simulation: Top {len(active_tickers)} US Mega-Caps Above EMA50 ({NUM_SIMULATIONS:,} Runs)')
    plt.xlabel('Trading Days (1 Year = 252 Days)')
    plt.ylabel('Portfolio Value ($ USD)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('monte_carlo_chart.png', dpi=300)
    print("Saved visualization to monte_carlo_chart.png")

if __name__ == '__main__':
    tickers = fetch_top50_ema50_tickers()
    prices = load_market_data(tickers)
    paths, final_tickers = run_monte_carlo(prices)
    generate_report(paths, final_tickers)
