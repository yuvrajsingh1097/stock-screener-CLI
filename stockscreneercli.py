"""
screener.py
──────────────────────────────────────────────────────────────────────────────
Stock Screener CLI

A command-line stock screener that filters tickers across multiple criteria:
  - P/E ratio range
  - Market capitalisation tier
  - 52-week return threshold
  - Momentum (price vs 50-day / 200-day MA)
  - Volume filter (avg daily volume)
  - Dividend yield range
  - Sector filter
  - RSI overbought / oversold exclusion

Outputs a coloured, ranked table to the terminal and optionally exports
results to CSV or saves a bar-chart PNG of the top performers.

Usage:
  python screener.py                               # scan default watchlist
  python screener.py --tickers AAPL MSFT NVDA TSLA AMZN
  python screener.py --pe-max 25 --return-min 10  # filter by P/E and return
  python screener.py --sector Technology --top 10
  python screener.py --sort sharpe --export results.csv
  python screener.py --help
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# DEFAULT WATCHLIST
# ─────────────────────────────────────────────────────────────

DEFAULT_TICKERS = [
    # Large-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    # Finance
    "JPM", "BAC", "GS", "BRK-B", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    # Consumer
    "PG", "KO", "PEP", "WMT", "COST", "MCD",
    # Energy
    "XOM", "CVX", "COP",
    # Industrial / Other
    "CAT", "BA", "HON", "GE", "LMT",
    # ETFs
    "SPY", "QQQ", "IWM",
]

# ─────────────────────────────────────────────────────────────
# TERMINAL COLOURS (ANSI)
# ─────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    # Foreground
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"
    # Background
    BG_DARK= "\033[40m"

def green(s):  return f"{C.GREEN}{s}{C.RESET}"
def red(s):    return f"{C.RED}{s}{C.RESET}"
def yellow(s): return f"{C.YELLOW}{s}{C.RESET}"
def cyan(s):   return f"{C.CYAN}{s}{C.RESET}"
def bold(s):   return f"{C.BOLD}{s}{C.RESET}"
def grey(s):   return f"{C.GREY}{s}{C.RESET}"
def dim(s):    return f"{C.DIM}{s}{C.RESET}"

def colour_pct(val: float) -> str:
    """Colour a percentage value: green if positive, red if negative."""
    if pd.isna(val):
        return grey("  —  ")
    s = f"{val:+.1f}%"
    if val >= 15:  return f"{C.BOLD}{C.GREEN}{s}{C.RESET}"
    if val >= 5:   return f"{C.GREEN}{s}{C.RESET}"
    if val >= 0:   return f"{C.CYAN}{s}{C.RESET}"
    if val >= -10: return f"{C.YELLOW}{s}{C.RESET}"
    return f"{C.RED}{s}{C.RESET}"

def colour_pe(val: float) -> str:
    if pd.isna(val) or val <= 0:
        return grey("  —  ")
    s = f"{val:.1f}x"
    if val <= 15:  return f"{C.GREEN}{s}{C.RESET}"
    if val <= 25:  return f"{C.CYAN}{s}{C.RESET}"
    if val <= 40:  return f"{C.YELLOW}{s}{C.RESET}"
    return f"{C.RED}{s}{C.RESET}"

def colour_rsi(val: float) -> str:
    if pd.isna(val):
        return grey("  —  ")
    s = f"{val:.0f}"
    if val >= 70:  return f"{C.RED}{s}{C.RESET}"
    if val >= 60:  return f"{C.YELLOW}{s}{C.RESET}"
    if val <= 30:  return f"{C.GREEN}{s}{C.RESET}"
    if val <= 40:  return f"{C.CYAN}{s}{C.RESET}"
    return s

def colour_mktcap(val: float) -> str:
    if pd.isna(val):
        return grey("  —  ")
    if val >= 1e12:
        return f"{C.MAGENTA}{val/1e12:.1f}T{C.RESET}"
    if val >= 1e9:
        return f"{C.CYAN}{val/1e9:.1f}B{C.RESET}"
    return f"{val/1e6:.0f}M"


# ─────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_ticker_data(ticker: str) -> dict | None:
    """
    Fetch fundamentals + price history for a single ticker.
    Returns a flat dict of computed metrics, or None on failure.
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.info

        # Price history — 1 year for return and MA calculations
        hist = t.history(period="1y", auto_adjust=True)
        if hist.empty or len(hist) < 50:
            return None

        closes = hist["Close"].dropna()
        volumes= hist["Volume"].dropna()

        # ── Returns ──
        ret_52w  = (closes.iloc[-1] / closes.iloc[0]  - 1) * 100
        ret_1m   = (closes.iloc[-1] / closes.iloc[-22] - 1) * 100 if len(closes) >= 22 else np.nan
        ret_3m   = (closes.iloc[-1] / closes.iloc[-63] - 1) * 100 if len(closes) >= 63 else np.nan
        ret_ytd  = _ytd_return(closes)

        # ── Moving averages ──
        ma50  = closes.tail(50).mean()
        ma200 = closes.tail(200).mean() if len(closes) >= 200 else np.nan
        price = float(closes.iloc[-1])
        above_50  = price > ma50
        above_200 = price > ma200 if not np.isnan(ma200) else None

        # ── RSI(14) ──
        rsi = _compute_rsi(closes, 14)

        # ── Volatility & Sharpe ──
        daily_rets = closes.pct_change().dropna()
        vol_annual = daily_rets.std() * np.sqrt(252) * 100
        # Simple Sharpe: annualised return / annualised vol (risk-free = 0)
        ann_ret  = ret_52w
        sharpe   = (ann_ret / vol_annual) if vol_annual > 0 else np.nan

        # ── Average daily volume ──
        avg_vol = float(volumes.tail(20).mean())

        # ── Fundamentals from info dict ──
        pe        = _safe(info, ["trailingPE", "forwardPE"])
        mkt_cap   = _safe(info, ["marketCap"])
        div_yield = (_safe(info, ["dividendYield"]) or 0) * 100
        sector    = info.get("sector", "Unknown")
        industry  = info.get("industry", "")
        name      = info.get("shortName", ticker)
        beta      = _safe(info, ["beta"])

        return {
            "ticker":    ticker,
            "name":      name[:22],
            "sector":    sector,
            "industry":  industry,
            "price":     price,
            "mkt_cap":   mkt_cap,
            "pe":        pe,
            "div_yield": div_yield,
            "ret_52w":   ret_52w,
            "ret_3m":    ret_3m,
            "ret_1m":    ret_1m,
            "ret_ytd":   ret_ytd,
            "ma50":      ma50,
            "ma200":     ma200,
            "above_50":  above_50,
            "above_200": above_200,
            "rsi":       rsi,
            "vol_ann":   vol_annual,
            "sharpe":    sharpe,
            "avg_vol":   avg_vol,
            "beta":      beta,
        }

    except Exception:
        return None


def _safe(info: dict, keys: list) -> float | None:
    for k in keys:
        v = info.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            try:
                return float(v)
            except Exception:
                pass
    return None


def _ytd_return(closes: pd.Series) -> float:
    """Return from first trading day of current year to today."""
    year_start = closes[closes.index.year == datetime.now().year]
    if len(year_start) < 2:
        return np.nan
    return (year_start.iloc[-1] / year_start.iloc[0] - 1) * 100


def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff().dropna()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs   = avg_gain / avg_loss.replace(0, np.nan)
    rsi  = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else np.nan


def _mktcap_tier(cap: float | None) -> str:
    if cap is None or np.isnan(cap):
        return "Unknown"
    if cap >= 200e9:  return "Mega"
    if cap >= 10e9:   return "Large"
    if cap >= 2e9:    return "Mid"
    if cap >= 300e6:  return "Small"
    return "Micro"


# ─────────────────────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)

    if args.pe_min is not None:
        mask &= df["pe"].fillna(np.inf) >= args.pe_min
    if args.pe_max is not None:
        mask &= df["pe"].fillna(np.inf) <= args.pe_max

    if args.return_min is not None:
        mask &= df["ret_52w"].fillna(-np.inf) >= args.return_min
    if args.return_max is not None:
        mask &= df["ret_52w"].fillna(np.inf) <= args.return_max

    if args.mktcap_tier:
        tiers = [t.strip().capitalize() for t in args.mktcap_tier.split(",")]
        mask &= df["mkt_cap_tier"].isin(tiers)

    if args.vol_min is not None:
        mask &= df["avg_vol"].fillna(0) >= args.vol_min * 1e6

    if args.div_min is not None:
        mask &= df["div_yield"].fillna(0) >= args.div_min

    if args.sector:
        sectors = [s.strip() for s in args.sector.split(",")]
        mask &= df["sector"].str.lower().isin([s.lower() for s in sectors])

    if args.above_50:
        mask &= df["above_50"] == True
    if args.above_200:
        mask &= df["above_200"] == True

    if args.rsi_max is not None:
        mask &= df["rsi"].fillna(100) <= args.rsi_max
    if args.rsi_min is not None:
        mask &= df["rsi"].fillna(0) >= args.rsi_min

    if args.beta_max is not None:
        mask &= df["beta"].fillna(np.inf) <= args.beta_max

    return df[mask].copy()


# ─────────────────────────────────────────────────────────────
# SORTING
# ─────────────────────────────────────────────────────────────

SORT_MAP = {
    "return":   ("ret_52w",  True),
    "return1m": ("ret_1m",   True),
    "return3m": ("ret_3m",   True),
    "pe":       ("pe",       False),
    "mktcap":   ("mkt_cap",  True),
    "sharpe":   ("sharpe",   True),
    "rsi":      ("rsi",      False),
    "vol":      ("vol_ann",  False),
    "dividend": ("div_yield",True),
    "volume":   ("avg_vol",  True),
    "ytd":      ("ret_ytd",  True),
}


# ─────────────────────────────────────────────────────────────
# DISPLAY TABLE
# ─────────────────────────────────────────────────────────────

RANK_ICONS = {1: "🥇", 2: "🥈", 3: "🥉"}

def print_table(df: pd.DataFrame, args, total_scanned: int):
    """Render the results as a rich ANSI terminal table."""

    n  = len(df)
    sort_col, sort_desc = SORT_MAP.get(args.sort, ("ret_52w", True))
    df = df.sort_values(sort_col, ascending=not sort_desc, na_position="last")
    if args.top:
        df = df.head(args.top)

    # ── Header ──────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
    width = 120

    print()
    print(f"{C.BOLD}{C.BG_DARK}{'':─<{width}}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  ██ STOCK SCREENER CLI  ██{C.RESET}  "
          f"{grey(ts)}  "
          f"  scanned {bold(str(total_scanned))} tickers  "
          f"  passed filters: {bold(green(str(n)))}")
    print(f"{C.BOLD}{C.BG_DARK}{'':─<{width}}{C.RESET}")

    if n == 0:
        print(f"\n  {yellow('No tickers matched the current filters.')}\n")
        return

    # ── Column header ──────────────────────────────────────
    header = (
        f"  {'#':>3}  "
        f"{'TICKER':<7} "
        f"{'NAME':<23} "
        f"{'SECTOR':<14} "
        f"{'PRICE':>8} "
        f"{'MKTCAP':>8} "
        f"{'P/E':>7} "
        f"{'52W RTN':>9} "
        f"{'3M RTN':>8} "
        f"{'YTD':>8} "
        f"{'RSI':>5} "
        f"{'VOL%':>7} "
        f"{'SHARPE':>7} "
        f"{'DIV%':>6} "
        f"{'MA':>6}"
    )
    print(f"\n{C.BOLD}{C.DIM}{header}{C.RESET}")
    print(f"  {'─'*115}")

    # ── Rows ────────────────────────────────────────────────
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        rank_str = RANK_ICONS.get(rank, f"{rank:>3}")

        # Ticker & name
        ticker = f"{C.BOLD}{C.WHITE}{row['ticker']:<7}{C.RESET}"
        name   = f"{grey(row['name'][:22]):<23}"

        # Sector (short)
        sector_short = row["sector"][:13] if pd.notna(row["sector"]) else ""
        sector_str   = f"{dim(sector_short):<14}"

        # Price
        price_str = f"{C.WHITE}{row['price']:>8,.2f}{C.RESET}"

        # Market cap
        mktcap_str = f"{colour_mktcap(row['mkt_cap']):>8}"

        # P/E
        pe_str = f"{colour_pe(row['pe']):>7}"

        # Returns
        r52_str = f"{colour_pct(row['ret_52w']):>9}"
        r3m_str = f"{colour_pct(row['ret_3m']):>8}"
        ytd_str = f"{colour_pct(row['ret_ytd']):>8}"

        # RSI
        rsi_str = f"{colour_rsi(row['rsi']):>5}"

        # Annualised volatility
        vol_s  = f"{row['vol_ann']:.0f}%" if pd.notna(row["vol_ann"]) else "  —"
        vol_str = f"{grey(vol_s):>7}"

        # Sharpe
        if pd.notna(row["sharpe"]):
            sh_s   = f"{row['sharpe']:.2f}"
            sh_col = C.GREEN if row["sharpe"] > 0.5 else (C.YELLOW if row["sharpe"] > 0 else C.RED)
            sharpe_str = f"{sh_col}{sh_s:>7}{C.RESET}"
        else:
            sharpe_str = f"{grey('  —'):>7}"

        # Dividend
        div_s  = f"{row['div_yield']:.1f}%" if row["div_yield"] > 0 else grey("  — ")
        div_str = f"{div_s:>6}"

        # MA signal
        a50  = row.get("above_50",  False)
        a200 = row.get("above_200", None)
        if a200 is True and a50:
            ma_str = green(" ↑↑ ")
        elif a200 is False and not a50:
            ma_str = red(" ↓↓ ")
        elif a50:
            ma_str = cyan(" ↑─ ")
        else:
            ma_str = yellow(" ─↓ ")

        print(
            f"  {rank_str}  "
            f"{ticker}"
            f"{name}"
            f"{sector_str}"
            f"{price_str}"
            f"{mktcap_str}"
            f"{pe_str}"
            f"{r52_str}"
            f"{r3m_str}"
            f"{ytd_str}"
            f"{rsi_str}"
            f"{vol_str}"
            f"{sharpe_str}"
            f"{div_str}"
            f"{ma_str}"
        )

    # ── Footer ──────────────────────────────────────────────
    print(f"\n  {'─'*115}")
    print(
        f"  {grey('Sorted by:')} {cyan(args.sort)}  "
        f"  {grey('MA:')} {green('↑↑')} above 50&200  "
        f"{cyan('↑─')} above 50 only  "
        f"{yellow('─↓')} below 50  "
        f"{red('↓↓')} below both"
    )
    print(
        f"  {grey('RSI colour:')} "
        f"{red('≥70')} overbought  "
        f"{yellow('60–70')}  "
        f"{cyan('40–50')}  "
        f"{green('≤30')} oversold"
    )
    print()


# ─────────────────────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """Print a one-line summary row of key aggregate stats."""
    if df.empty:
        return
    above_200_count = f"{int(df['above_200'].sum())}/{len(df)}"
    sharpe_median = df['sharpe'].median()
    sharpe_str = cyan(f"{sharpe_median:.2f}") if pd.notna(sharpe_median) else grey('—')
    print(f"  {bold('SUMMARY')}  "
          f"median 52w return: {colour_pct(df['ret_52w'].median())}  "
          f"median P/E: {colour_pe(df['pe'].median())}  "
          f"median Sharpe: {sharpe_str}  "
          f"avg RSI: {colour_rsi(df['rsi'].mean())}  "
          f"above 200MA: {green(above_200_count)}")
    print()


# ─────────────────────────────────────────────────────────────
# CHART EXPORT
# ─────────────────────────────────────────────────────────────

def save_chart(df: pd.DataFrame, sort_col: str,
               output_path: str = "screener_chart.png",
               top_n: int = 20):
    """Save a horizontal bar chart of top-N results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not installed — skipping chart export.")
        return

    sort_field, desc = SORT_MAP.get(sort_col, ("ret_52w", True))
    plot_df = (df.sort_values(sort_field, ascending=not desc, na_position="last")
                 .head(top_n)
                 .dropna(subset=[sort_field])
                 .iloc[::-1])   # flip so top rank is at the top of chart

    if plot_df.empty:
        return

    vals   = plot_df[sort_field].values
    labels = [f"{row['ticker']}  {row['name'][:18]}"
              for _, row in plot_df.iterrows()]
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in vals]

    BG    = "#0d1117"; PANEL = "#161b22"
    TC    = "#e6edf3"; MC    = "#8b949e"; GC   = "#21262d"

    fig_h = max(5, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(12, fig_h), facecolor=BG)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GC)
    ax.grid(axis="x", color=GC, lw=0.5, ls="--", alpha=0.6)
    ax.tick_params(colors=MC, labelsize=8)
    ax.xaxis.label.set_color(MC)

    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, vals, color=colors, height=0.65,
                    edgecolor=BG, linewidth=0.5, alpha=0.88)

    # Value labels
    for bar, val in zip(bars, vals):
        x_offset = 0.3 if val >= 0 else -0.3
        ha = "left" if val >= 0 else "right"
        suffix = "%" if "ret" in sort_field or "yield" in sort_field else ""
        fmt    = f"{val:+.1f}{suffix}" if "ret" in sort_field else f"{val:.2f}"
        ax.text(val + x_offset, bar.get_y() + bar.get_height() / 2,
                fmt, va="center", ha=ha, fontsize=7.5, color=TC)

    ax.axvline(0, color=MC, lw=0.7, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, color=TC)
    ax.set_title(
        f"Top {len(labels)} by {sort_col.upper()}  ·  {datetime.now().strftime('%Y-%m-%d')}",
        color=TC, fontsize=11, pad=10, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  Chart saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, path: str):
    cols = ["ticker", "name", "sector", "industry", "price", "mkt_cap",
            "mkt_cap_tier", "pe", "div_yield", "ret_52w", "ret_3m",
            "ret_1m", "ret_ytd", "rsi", "vol_ann", "sharpe",
            "avg_vol", "beta", "above_50", "above_200"]
    out = df[[c for c in cols if c in df.columns]]
    out.to_csv(path, index=False)
    print(f"  CSV  saved → {path}  ({len(out)} rows)")


# ─────────────────────────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────────────────────────

def progress(current: int, total: int, ticker: str, width: int = 30):
    pct  = current / total
    done = int(pct * width)
    bar  = "█" * done + "░" * (width - done)
    print(f"\r  [{bar}] {current}/{total}  {ticker:<12}", end="", flush=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stock Screener CLI — filter, rank and export tickers",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ── Ticker selection ──
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        metavar="TICKER",
        help="Space-separated list of tickers to scan.\n"
             "Default: built-in watchlist of ~35 large-caps."
    )

    # ── Filter arguments ──
    parser.add_argument("--pe-min",      type=float, default=None,
                        help="Minimum trailing P/E ratio")
    parser.add_argument("--pe-max",      type=float, default=None,
                        help="Maximum trailing P/E ratio")
    parser.add_argument("--return-min",  type=float, default=None,
                        help="Minimum 52-week return (%)")
    parser.add_argument("--return-max",  type=float, default=None,
                        help="Maximum 52-week return (%)")
    parser.add_argument("--mktcap-tier", type=str,   default=None,
                        metavar="TIER",
                        help="Market cap tier: Mega, Large, Mid, Small, Micro\n"
                             "Comma-separate multiple: 'Mega,Large'")
    parser.add_argument("--vol-min",     type=float, default=None,
                        metavar="MILLIONS",
                        help="Minimum average daily volume in millions")
    parser.add_argument("--div-min",     type=float, default=None,
                        help="Minimum dividend yield (%)")
    parser.add_argument("--sector",      type=str,   default=None,
                        help="Filter by sector name (partial, comma-separated).\n"
                             "e.g. 'Technology' or 'Technology,Healthcare'")
    parser.add_argument("--above-50",    action="store_true",
                        help="Only show tickers above their 50-day MA")
    parser.add_argument("--above-200",   action="store_true",
                        help="Only show tickers above their 200-day MA")
    parser.add_argument("--rsi-min",     type=float, default=None,
                        help="Minimum RSI(14)")
    parser.add_argument("--rsi-max",     type=float, default=None,
                        help="Maximum RSI(14)")
    parser.add_argument("--beta-max",    type=float, default=None,
                        help="Maximum beta (e.g. 1.5 to exclude high-beta names)")

    # ── Output arguments ──
    parser.add_argument("--sort",   type=str, default="return",
                        choices=list(SORT_MAP.keys()),
                        help="Sort column (default: return)\n"
                             "Options: " + ", ".join(SORT_MAP.keys()))
    parser.add_argument("--top",    type=int, default=None,
                        help="Show only the top N results")
    parser.add_argument("--export", type=str, default=None,
                        metavar="FILE.csv",
                        help="Export filtered results to CSV")
    parser.add_argument("--chart",  type=str, default=None,
                        metavar="FILE.png",
                        help="Save bar chart of top results to PNG")
    parser.add_argument("--no-colour", action="store_true",
                        help="Disable ANSI colour output")

    args = parser.parse_args()

    if args.no_colour:
        # Disable all colour codes
        for attr in vars(C):
            if not attr.startswith("_"):
                setattr(C, attr, "")

    tickers = args.tickers or DEFAULT_TICKERS
    total   = len(tickers)

    # ── Banner ──────────────────────────────────────────────
    print()
    print(f"{C.BOLD}{C.CYAN}  ╔══════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  ║         STOCK SCREENER CLI               ║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  ╚══════════════════════════════════════════╝{C.RESET}")
    print(f"  {grey('Scanning')} {bold(str(total))} {grey('tickers via yfinance ...')}")
    print()

    # ── Fetch ────────────────────────────────────────────────
    rows = []
    failed = []
    for i, ticker in enumerate(tickers):
        progress(i + 1, total, ticker)
        data = fetch_ticker_data(ticker)
        if data:
            rows.append(data)
        else:
            failed.append(ticker)

    print()  # newline after progress bar

    if not rows:
        print(f"\n  {red('No data fetched. Check your internet connection or ticker symbols.')}\n")
        sys.exit(1)

    # ── Build DataFrame ──────────────────────────────────────
    df = pd.DataFrame(rows)
    df["mkt_cap_tier"] = df["mkt_cap"].apply(_mktcap_tier)

    # ── Filter ───────────────────────────────────────────────
    filtered = apply_filters(df, args)

    # ── Display ──────────────────────────────────────────────
    print_table(filtered, args, total_scanned=len(df))
    print_summary(filtered)

    # ── Failed tickers ────────────────────────────────────────
    if failed:
        print(f"  {grey('Skipped (no data):')} {grey(', '.join(failed[:10]))}"
              + (f" … +{len(failed)-10} more" if len(failed) > 10 else ""))
        print()

    # ── Exports ───────────────────────────────────────────────
    if args.export:
        export_csv(filtered, args.export)
    if args.chart:
        save_chart(filtered, args.sort, output_path=args.chart,
                   top_n=args.top or 20)

    print()


if __name__ == "__main__":
    main()