Defensive data access is essential when scraping financial APIs — yfinance returns None, NaN, missing keys, and the occasional string where you expect a float. The _safe() helper that tries multiple key names and validates the result before returning is a pattern used in production data pipelines
The Sharpe ratio with zero risk-free rate is a useful first-pass quality filter: it rewards high returns without punishing low-volatility names the way raw return rankings do. NVDA at 2.18× Sharpe tells you more than "it went up 178%" alone
RSI on daily closes is a 1-liner with pandas ewm(), but the tricky part is that a perfectly monotone series (all gains, no losses) makes the denominator zero — the .replace(0, np.nan) guard is what keeps it from crashing
ANSI colour codes in terminal output make a surprisingly big difference to usability — green/red/yellow colouring on the return column lets you scan 30 rows in 2 seconds instead of reading each number individually
Market cap tiers are more useful than raw cap values for filtering — "give me Mega and Large caps only" is a natural way to think about a universe, even if the exact boundary (≥$200B = Mega) is a convention


Tech stack

yfinance — live fundamentals and price history
pandas + numpy — RSI, returns, volatility, Sharpe
matplotlib — bar chart PNG export
pytest — unit tests for RSI, filters, market cap tiers
