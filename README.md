# US Mega-Cap Momentum Monte Carlo Risk Simulator

A quantitative portfolio risk engine in Python that dynamically screens the **Top 50 US Mega-Cap stocks trading above their 50-day Exponential Moving Average (`Price > EMA50`)** via TradingView, pulls historical pricing via `yfinance`, and simulates **thousands of correlated portfolio paths** (10,000 over 1 year by default) using Geometric Brownian Motion (GBM) and Cholesky Decomposition. It compares **Equal-Weighted (1/N)** and **Market-Cap Weighted** portfolios side by side.

![Monte Carlo Simulation Chart](monte_carlo_chart.png)

The chart is regenerated on every run and shows, for each weighting strategy:

* **Simulated paths (top):** a sample of individual paths, the median path, and the 25th-75th and 5th-95th percentile bands against the initial capital line.
* **Ending value distribution (middle):** a histogram of final portfolio values with the initial capital, median, and VaR cutoff marked.
* **Risk panel (bottom):** a separate box, outside the plot axes, showing the **Value at Risk (VaR)** and **Expected Shortfall (CVaR)** in dollars and percent at your chosen confidence level.

---

## Key Features
* **Live Momentum & Mega-Cap Screening:** Queries TradingView's screener API at runtime to filter NYSE/NASDAQ primary stocks where `Close > EMA50`, ranked by market capitalization.
* **Two Weighting Strategies:** Run `equal`, `market_cap`, or `both` and compare them in one report and one chart.
* **Correlation-Preserving Simulation:** Applies **Cholesky Decomposition** ($L L^T = \Sigma$) to the asset covariance matrix so simulated daily shocks reflect real-world sector correlations rather than independent random walks.
* **Institutional Risk Metrics:** Median ending value and return, probability of loss, **Value at Risk (VaR)**, **Conditional VaR / Expected Shortfall (CVaR)**, median maximum drawdown, and annualized **Sharpe** and **Sortino** ratios.
* **Configurable Inputs:** Capital, number of simulations, horizon, confidence level, weighting, and risk-free rate are all command-line arguments.
* **GitHub Actions Workflow:** Run the simulation from the Actions tab with custom inputs; the results table appears in the run summary and the chart and cache are committed back to the repo.
* **Self-Healing Data Cache:** Automatically saves a local `fallback_prices.csv` snapshot so the simulation runs reliably even offline or under API rate limits.

---

## Quantitative Methodology

1. **Log Returns & Drift:**
   Daily log returns are calculated from adjusted closing prices ($S_t$) over a 3-year lookback:
   $$r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)$$
   The drift $\mu$ is the sample mean of these daily log returns.

2. **Cholesky Decomposition:**
   To simulate correlated shocks across the assets, the regularized covariance matrix $\Sigma$ of log returns is decomposed into a lower triangular matrix $L$:
   $$\Sigma = L L^T$$
   Given uncorrelated standard normal random draws $Z \sim \mathcal{N}(0, 1)$, correlated shocks are generated via $\epsilon = Z L^T$.

3. **Geometric Brownian Motion (GBM):**
   Each asset path evolves daily over $T$ trading days (252 by default) using:
   $$S_{t} = S_{t-1} \exp\left(\mu + \epsilon_t\right)$$
   Because $\mu$ is estimated from *log* returns, it already includes the $-\frac{1}{2}\sigma^2$ convexity adjustment, so it is not subtracted a second time.

4. **Portfolio Construction:**
   Portfolio value is the weighted sum of each asset's simulated growth, starting from the initial capital. Weights are set once at the start (buy-and-hold, no rebalancing). Equal weighting uses $1/N$; market-cap weighting uses each stock's share of total market capitalization from the screener.

5. **Risk Metrics:**
   * **VaR:** the loss at the $(100 - c)$th percentile of ending values, for confidence level $c$.
   * **CVaR (Expected Shortfall):** the average loss across all simulations at or beyond the VaR cutoff.
   * **Sharpe / Sortino:** annualized from each simulated path's daily returns against the risk-free rate (Sortino uses downside deviation only); the median across paths is reported.
   * **Max Drawdown:** the largest peak-to-trough decline on each path; the median across paths is reported.

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the simulation
```bash
python simulation.py
```
Running the script prints the risk report to your terminal and refreshes `monte_carlo_chart.png` and `fallback_prices.csv`.

### Command-line options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--capital` | `100000` | Initial portfolio capital ($ USD) |
| `--sims` | `10000` | Number of Monte Carlo simulations |
| `--days` | `252` | Horizon in trading days (252 = 1 year) |
| `--confidence` | `95` | VaR / CVaR confidence level (e.g. 90, 95, 99) |
| `--weighting` | `both` | `equal`, `market_cap`, or `both` |
| `--rf` | `0.0425` | Annual risk-free rate as a decimal |

Example:
```bash
python simulation.py --capital 250000 --days 126 --confidence 99 --weighting both
```

---

## Run on GitHub Actions

The workflow in `.github/workflows/simulation.yml` can be started manually:

1. Open the **Actions** tab and select **Run Monte Carlo Simulation**.
2. Click **Run workflow** and fill in the inputs (capital, simulations, horizon, confidence, risk-free rate, weighting).
3. When it finishes, the comparison table is shown in the run summary, and the updated `monte_carlo_chart.png` and `fallback_prices.csv` are committed to the repo.

---

## Notes and Limitations
* If the TradingView query fails, the script falls back to a built-in list of mega-caps with no market-cap data, so market-cap weights become equal to 1/N weights. A warning is printed when this happens.
* GBM assumes normally distributed returns with constant drift and volatility estimated from the past 3 years. Real markets have fat tails and regime changes, so treat tail-risk figures as estimates rather than guarantees.
* Screening on `Price > EMA50` today and simulating forward from historical statistics introduces selection bias; results describe the screened basket, not a tested trading strategy.

---

## Disclaimer
*This project is for educational and quantitative research purposes only and does not constitute financial or investment advice.*
