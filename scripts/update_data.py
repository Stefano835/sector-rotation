"""
Sector Rotation · Data Update Script
=====================================
Scarica prezzi via yfinance, calcola tutte le metriche (RRG, Stage Weinstein,
performance, cross-region) e salva un JSON pronto per essere letto dall'HTML.

Pensato per girare su GitHub Actions una volta al giorno.

Output: data/sector_data.json
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TICKERS
# ============================================================
US_BENCHMARK = 'SPY'
US_SECTORS = {
    'XLK':  'Tecnologia',
    'XLF':  'Finanziari',
    'XLV':  'Sanità',
    'XLY':  'Consumi voluttuari',
    'XLP':  'Consumi essenziali',
    'XLE':  'Energia',
    'XLI':  'Industriali',
    'XLB':  'Materiali base',
    'XLU':  'Utility',
    'XLRE': 'Immobiliare',
    'XLC':  'Comunicazioni',
    'SOXX': 'Semiconduttori',
    'IBB':  'Biotech',
    'KRE':  'Banche regionali',
    'XHB':  'Costruttori case',
    'XRT':  'Retail',
    'ITA':  'Aerospazio & Difesa',
}

EU_BENCHMARK = 'EXSA.DE'
EU_SECTORS = {
    'EXV3.DE': 'Tecnologia',
    'EXV1.DE': 'Banche',
    'EXV5.DE': 'Automobili',
    'EXH1.DE': 'Energia',
    'EXV4.DE': 'Sanità',
    'EXH5.DE': 'Assicurazioni',
    'EXH3.DE': 'Servizi finanziari',
    'EXH2.DE': 'Retail',
    'EXH8.DE': 'Media',
    'EXH4.DE': 'Industriali',
    'EXV8.DE': 'Costruzioni',
    'EXV6.DE': 'Risorse base',
    'EXH6.DE': 'Beni personali',
    'EXV2.DE': 'Telecomunicazioni',
    'EXH7.DE': 'Alimentari & Bevande',
    'EXV7.DE': 'Chimica',
    'EXH9.DE': 'Utility',
    'EXV9.DE': 'Viaggi & Tempo libero',
}

CROSS_PAIRS = [
    ('Tecnologia',          'XLK',  'EXV3.DE'),
    ('Banche / Finanziari', 'XLF',  'EXV1.DE'),
    ('Energia',             'XLE',  'EXH1.DE'),
    ('Sanità',              'XLV',  'EXV4.DE'),
    ('Industriali',         'XLI',  'EXH4.DE'),
    ('Auto / Consumi vol.', 'XLY',  'EXV5.DE'),
    ('Utility',             'XLU',  'EXH9.DE'),
    ('Materiali base',      'XLB',  'EXV6.DE'),
    ('Consumi essenziali',  'XLP',  'EXH7.DE'),
    ('Comunicazioni',       'XLC',  'EXV2.DE'),
    ('Retail',              'XRT',  'EXH2.DE'),
]


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_prices(tickers, period='2y'):
    """Bulk download + fallback per-ticker. Robusto a fallimenti parziali."""
    if isinstance(tickers, str):
        tickers = [tickers]
    
    result = pd.DataFrame()
    
    # Strategy 1: bulk
    try:
        data = yf.download(
            tickers, period=period, auto_adjust=True,
            progress=False, threads=True, group_by='ticker'
        )
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                cols = {}
                for t in tickers:
                    try:
                        if t in data.columns.get_level_values(0):
                            s = data[t]['Close'].dropna()
                            if len(s) > 0:
                                cols[t] = s
                    except Exception:
                        pass
                if cols:
                    result = pd.DataFrame(cols)
            elif 'Close' in data.columns:
                close = data['Close']
                if isinstance(close, pd.Series):
                    result = close.to_frame(name=tickers[0])
                else:
                    result = close.dropna(axis=1, how='all')
    except Exception as e:
        print(f"  Bulk failed: {e}", file=sys.stderr)
    
    # Strategy 2: per-ticker fallback
    missing = [t for t in tickers if t not in result.columns or result[t].dropna().empty]
    for t in missing:
        try:
            hist = yf.Ticker(t).history(period=period, auto_adjust=True)
            if hist is not None and not hist.empty and 'Close' in hist.columns:
                s = hist['Close'].dropna()
                if len(s) > 0:
                    if s.index.tz is not None:
                        s.index = s.index.tz_localize(None)
                    result[t] = s
        except Exception as e:
            print(f"  Per-ticker {t} failed: {e}", file=sys.stderr)
    
    if not result.empty:
        result = result.dropna(axis=1, how='all')
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)
    
    return result


# ============================================================
# RRG · RS-Ratio + RS-Momentum (metodo JdK approssimato)
# ============================================================
def calculate_rrg(symbol_prices, benchmark_prices, window=14):
    sym_w = symbol_prices.resample('W-FRI').last()
    bench_w = benchmark_prices.resample('W-FRI').last()
    df = pd.concat([sym_w, bench_w], axis=1).dropna()
    df.columns = ['sym', 'bench']
    
    if len(df) < window * 5:
        return None
    
    rs = df['sym'] / df['bench']
    long_window = window * 4
    rs_mean = rs.rolling(window=long_window, min_periods=window*2).mean()
    rs_std = rs.rolling(window=long_window, min_periods=window*2).std()
    rs_z = (rs - rs_mean) / rs_std
    rs_ratio = 100 + rs_z.clip(-3.5, 3.5) * 3
    
    rs_roc = rs_ratio.diff(periods=window // 3)
    roc_std = rs_roc.rolling(window=long_window, min_periods=window*2).std()
    rs_mom_z = rs_roc / roc_std
    rs_momentum = 100 + rs_mom_z.clip(-3.5, 3.5) * 3
    
    out = pd.DataFrame({
        'rs_ratio': rs_ratio,
        'rs_momentum': rs_momentum
    }).dropna()
    return out if len(out) > 5 else None


def classify_quadrant(rs, mom):
    if pd.isna(rs) or pd.isna(mom):
        return 'N/A'
    if rs >= 100 and mom >= 100:
        return 'In testa'
    if rs < 100 and mom >= 100:
        return 'In ripresa'
    if rs < 100 and mom < 100:
        return 'In ritardo'
    return 'In calo'


def classify_stage(prices, ma_weeks=30):
    weekly = prices.resample('W-FRI').last().dropna()
    if len(weekly) < ma_weeks + 8:
        return 'N/A'
    ma = weekly.rolling(window=ma_weeks).mean()
    ma_slope = (ma.diff(periods=4) / ma.shift(4)) * 100
    if pd.isna(ma.iloc[-1]) or pd.isna(ma_slope.iloc[-1]):
        return 'N/A'
    price = float(weekly.iloc[-1])
    ma_curr = float(ma.iloc[-1])
    slope = float(ma_slope.iloc[-1])
    distance_pct = ((price - ma_curr) / ma_curr) * 100
    
    if price > ma_curr:
        if slope > 0.3: return '2'
        if slope < -0.3: return '3'
        return '3' if distance_pct < 3 else '2'
    else:
        if slope < -0.3: return '4'
        return '1'


# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_sector_metrics(prices_df, bench_ticker, sector_dict):
    if bench_ticker not in prices_df.columns:
        return []
    bench_series = prices_df[bench_ticker].dropna()
    
    rows = []
    for ticker, name in sector_dict.items():
        if ticker not in prices_df.columns:
            continue
        sym_prices = prices_df[ticker].dropna()
        if len(sym_prices) < 100:
            continue
        
        rrg = calculate_rrg(sym_prices, bench_series)
        if rrg is None or len(rrg) < 6:
            continue
        
        last_rs = float(rrg['rs_ratio'].iloc[-1])
        last_mom = float(rrg['rs_momentum'].iloc[-1])
        prev_rs = float(rrg['rs_ratio'].iloc[-5]) if len(rrg) > 5 else last_rs
        quadrant = classify_quadrant(last_rs, last_mom)
        stage = classify_stage(sym_prices)
        
        # Performance
        roc_13w = ((sym_prices.iloc[-1] / sym_prices.iloc[-min(65, len(sym_prices)-1)]) - 1) * 100
        roc_52w = ((sym_prices.iloc[-1] / sym_prices.iloc[-min(252, len(sym_prices)-1)]) - 1) * 100
        
        bench_aligned = bench_series.reindex(sym_prices.index).dropna()
        if len(bench_aligned) > 65:
            sym_ret_13w = ((sym_prices.iloc[-1] / sym_prices.iloc[-65]) - 1) * 100
            bench_ret_13w = ((bench_aligned.iloc[-1] / bench_aligned.iloc[-65]) - 1) * 100
            rel_13w = sym_ret_13w - bench_ret_13w
        else:
            rel_13w = 0
        
        # Tail ultime 5 settimane
        tail_rs = [round(float(v), 2) for v in rrg['rs_ratio'].iloc[-5:].tolist()]
        tail_mom = [round(float(v), 2) for v in rrg['rs_momentum'].iloc[-5:].tolist()]
        
        # Serie 26 settimane per chart
        weeks_26 = rrg['rs_ratio'].iloc[-26:] if len(rrg) >= 26 else rrg['rs_ratio']
        rs_series = [
            {'date': d.strftime('%Y-%m-%d'), 'value': round(float(v), 2)}
            for d, v in weeks_26.items()
        ]
        
        display_ticker = ticker.replace('.DE', '').replace('.US', '')
        
        rows.append({
            'ticker': display_ticker,
            'ticker_raw': ticker,
            'name': name,
            'stage': stage,
            'state': quadrant,
            'rsRatio': round(last_rs, 2),
            'rsMom': round(last_mom, 2),
            'delta5w': round(last_rs - prev_rs, 2),
            'roc13w': round(float(roc_13w), 1),
            'roc52w': round(float(roc_52w), 1),
            'rel13w': round(float(rel_13w), 1),
            'tailRS': tail_rs,
            'tailMom': tail_mom,
            'rsRatioSeries': rs_series,
        })
    
    return rows


def compute_cross_region(prices_df):
    """Cross-region analysis: USA vs EU per settore omologo."""
    if US_BENCHMARK not in prices_df.columns or EU_BENCHMARK not in prices_df.columns:
        return []
    
    us_bench = prices_df[US_BENCHMARK].dropna()
    eu_bench = prices_df[EU_BENCHMARK].dropna()
    
    rows = []
    for label, us_tick, eu_tick in CROSS_PAIRS:
        if us_tick not in prices_df.columns or eu_tick not in prices_df.columns:
            continue
        
        us = prices_df[us_tick].dropna()
        eu = prices_df[eu_tick].dropna()
        
        if len(us) < 65 or len(eu) < 65:
            continue
        
        us_rel = ((us.iloc[-1]/us.iloc[-65]) / (us_bench.iloc[-1]/us_bench.iloc[-65]) - 1) * 100
        eu_rel = ((eu.iloc[-1]/eu.iloc[-65]) / (eu_bench.iloc[-1]/eu_bench.iloc[-65]) - 1) * 100
        
        cross_now = us.iloc[-1] / eu.iloc[-1]
        cross_13w = us.iloc[-65] / eu.iloc[-65]
        cross_change = ((cross_now / cross_13w) - 1) * 100
        
        leader = 'USA' if cross_change > 0 else 'EU'
        
        rows.append({
            'label': label,
            'usT': us_tick.replace('.US', ''),
            'euT': eu_tick.replace('.DE', ''),
            'usRel': round(float(us_rel), 1),
            'euRel': round(float(eu_rel), 1),
            'cross': round(float(cross_change), 1),
            'leader': leader,
        })
    
    return rows


# ============================================================
# MAIN
# ============================================================
def main():
    print("Sector Rotation · Data Update")
    print("=" * 60)
    
    all_tickers = (
        [US_BENCHMARK] + list(US_SECTORS.keys()) +
        [EU_BENCHMARK] + list(EU_SECTORS.keys())
    )
    all_tickers = list(set(all_tickers))
    
    print(f"Scarico {len(all_tickers)} ticker da yfinance...")
    prices = fetch_prices(all_tickers, period='2y')
    
    obtained = list(prices.columns)
    missing = [t for t in all_tickers if t not in obtained]
    print(f"  Ottenuti: {len(obtained)}/{len(all_tickers)}")
    if missing:
        print(f"  Mancanti: {', '.join(missing)}")
    
    if prices.empty:
        print("ERRORE: nessun dato scaricato.", file=sys.stderr)
        sys.exit(1)
    
    print("\nCalcolo metriche USA...")
    us_metrics = compute_sector_metrics(prices, US_BENCHMARK, US_SECTORS)
    print(f"  {len(us_metrics)} settori USA elaborati")
    
    print("Calcolo metriche EU...")
    eu_metrics = compute_sector_metrics(prices, EU_BENCHMARK, EU_SECTORS)
    print(f"  {len(eu_metrics)} settori EU elaborati")
    
    print("Calcolo cross-region...")
    cross_rows = compute_cross_region(prices)
    print(f"  {len(cross_rows)} coppie cross-region")
    
    # Date più recente nei dati
    last_data_date = prices.index[-1].strftime('%Y-%m-%d')
    
    output = {
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'last_data_date': last_data_date,
        'tickers_obtained': len(obtained),
        'tickers_total': len(all_tickers),
        'tickers_missing': missing,
        'us': {
            'benchmark': US_BENCHMARK,
            'benchmark_label': 'S&P 500',
            'sectors': us_metrics,
        },
        'eu': {
            'benchmark': EU_BENCHMARK,
            'benchmark_label': 'STOXX 600',
            'sectors': eu_metrics,
        },
        'cross': cross_rows,
    }
    
    out_path = Path(__file__).parent.parent / 'data' / 'sector_data.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Salvato in {out_path}")
    print(f"  Dimensione: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"  Data ultimi dati: {last_data_date}")


if __name__ == '__main__':
    main()
