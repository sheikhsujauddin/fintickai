import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="FintickAI Elite Dashboard", layout="wide")

# --- Function to fetch Nifty 50 symbols from Wikipedia ---
@st.cache_data
def fetch_nifty50_symbols():
    url = "https://en.wikipedia.org/wiki/NIFTY_50"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    nifty_symbols = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        if table:
            for row in table.find_all('tr')[1:]:
                columns = row.find_all('td')
                if columns and len(columns) > 1:
                    symbol = columns[1].text.strip()
                    if re.match(r'^[A-Z0-9]+$', symbol):
                        nifty_symbols.append(symbol + '.NS')
        return nifty_symbols
    except Exception as e:
        st.error(f"Could not fetch Nifty 50 symbols: {str(e)}")
        return []

nifty50_symbols = fetch_nifty50_symbols()

# --- Data Prep and Features Functions ---
@st.cache_data
def get_processed_data(symbol):
    df = yf.Ticker(symbol).history(period='10y')
    if df.empty:
        return None
    df.dropna(inplace=True)
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['stoch'] = stoch.stoch()
    for p in [5, 20, 50, 100, 200]:
        df[f'sma_{p}'] = df['Close'].rolling(p).mean()
        df[f'ema_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    atr_period, multiplier = 10, 3
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, min_periods=atr_period, adjust=False).mean()
    hl2 = (high + low) / 2
    up_lev, dn_lev = hl2 + (multiplier * atr), hl2 - (multiplier * atr)
    st_arr, direction = [0.0] * len(df), [True] * len(df)
    for i in range(1, len(df)):
        if close.iloc[i] > up_lev.iloc[i-1]:
            direction[i] = True
        elif close.iloc[i] < dn_lev.iloc[i-1]:
            direction[i] = False
        else:
            direction[i] = direction[i-1]
            if direction[i] and dn_lev.iloc[i] < dn_lev.iloc[i-1]: dn_lev.values[i] = dn_lev.values[i-1]
            if not direction[i] and up_lev.iloc[i] > up_lev.iloc[i-1]: up_lev.values[i] = up_lev.values[i-1]
        st_arr[i] = dn_lev.iloc[i] if direction[i] else up_lev.iloc[i]
    df['supertrend'], df['st_direction'] = st_arr, direction
    return df

def get_summary(df):
    latest = df.iloc[-1]
    price = latest['Close']
    rsi_sent = 'Bullish' if latest['rsi'] < 40 else 'Bearish' if latest['rsi'] > 60 else 'Neutral'
    macd_sent = 'Bullish' if latest['macd'] > latest['macd_signal'] else 'Bearish'
    stoch_sent = 'Bullish' if latest['stoch'] < 20 else 'Bearish' if latest['stoch'] > 80 else 'Neutral'
    st_sent = 'Bullish' if latest['st_direction'] else 'Bearish'
    summary = {
        'Price': round(price, 2),
        'RSI': f"{round(latest['rsi'],2)} ({rsi_sent})",
        'MACD': f"{round(latest['macd'],2)} ({macd_sent})",
        'Stoch': f"{round(latest['stoch'],2)} ({stoch_sent})",
        'SuperTrend': f"{round(latest['supertrend'],2)} ({st_sent})",
        'MA_Data': {}
    }
    for p in [5, 20, 50, 100, 200]:
        s_val, e_val = latest[f'sma_{p}'], latest[f'ema_{p}']
        summary['MA_Data'][f'sma_{p}'] = f"{round(s_val,2)} ({'Bullish' if price > s_val else 'Bearish'})"
        summary['MA_Data'][f'ema_{p}'] = f"{round(e_val,2)} ({'Bullish' if price > e_val else 'Bearish'})"
    return summary

@st.cache_data
def get_gainers_losers(symbols):
    today = datetime.now()
    start_date = today - pd.DateOffset(days=10)
    end_date = today + pd.DateOffset(days=1)
    data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=True)
    stock_performance = []
    for symbol in symbols:
        try:
            symbol_data = data[symbol].dropna()
            if len(symbol_data) >= 2:
                latest_close = symbol_data['Close'].iloc[-1]
                previous_close = symbol_data['Close'].iloc[-2]
                if previous_close != 0:  
                    daily_change = ((latest_close - previous_close) / previous_close) * 100
                    stock_performance.append({'symbol': symbol, 'change': daily_change})
        except Exception:
            continue
    performance_df = pd.DataFrame(stock_performance)
    if performance_df.empty:
        return [], []
    performance_df = performance_df.sort_values(by='change', ascending=False)
    top_gainers = performance_df.head(5).to_dict('records')
    top_losers = performance_df.tail(5).to_dict('records')
    top_losers.reverse()
    return top_gainers, top_losers

def plot_market(df, title):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [4, 1, 1]})
    ax1.plot(df.index, df['Close'], color='black', label='Price', linewidth=1.5)
    up, down = df['supertrend'].copy(), df['supertrend'].copy()
    up[df['st_direction'] == False] = np.nan; down[df['st_direction'] == True] = np.nan
    ax1.plot(df.index, up, color='green', linewidth=2); ax1.plot(df.index, down, color='red', linewidth=2)
    for p, c in zip([5, 20, 50, 100, 200], ['cyan', 'orange', 'red', 'blue', 'green']):
        ax1.plot(df.index, df[f'sma_{p}'], color=c, alpha=0.15)
    ax1.set_ylim(df['Close'].min()*0.98, df['Close'].max()*1.02)
    ax1.set_title(title)
    ax2.plot(df.index, df['rsi'], color='purple')
    ax3.plot(df.index, df['macd'], label='MACD')
    ax3.plot(df.index, df['macd_signal'], label='Signal')
    ax3.legend()
    plt.tight_layout()
    return fig

# --- Streamlit App Layout ---
st.title('FINTICKAI ELITE DASHBOARD')
st.write(f"**Date:** {datetime.now().strftime('%d %B %Y')}")

st.sidebar.header("Index Selection")
index_options = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK"}
selected_index = st.sidebar.selectbox("Select Index for detail view", list(index_options.keys()))

nifty_df = get_processed_data(index_options["NIFTY 50"])
bank_df = get_processed_data(index_options["BANK NIFTY"])
n_sum = get_summary(nifty_df) if nifty_df is not None else None
b_sum = get_summary(bank_df) if bank_df is not None else None

st.header("Index Metrics")
cols = st.columns(2)
if n_sum is not None:
    with cols[0]:
        st.subheader("NIFTY 50")
        st.metric("Price", n_sum['Price'])
        st.write(f"RSI: {n_sum['RSI']} | MACD: {n_sum['MACD']} | STOCH: {n_sum['Stoch']}")
        st.write(f"SuperTrend: {n_sum['SuperTrend']}")
        st.table(n_sum['MA_Data'])
else:
    st.warning("NIFTY 50 data not available.")

if b_sum is not None:
    with cols[1]:
        st.subheader("BANK NIFTY")
        st.metric("Price", b_sum['Price'])
        st.write(f"RSI: {b_sum['RSI']} | MACD: {b_sum['MACD']} | STOCH: {b_sum['Stoch']}")
        st.write(f"SuperTrend: {b_sum['SuperTrend']}")
        st.table(b_sum['MA_Data'])
else:
    st.warning("BANK NIFTY data not available.")

st.header("NIFTY 50 - Top 5 Gainers / Losers")
if nifty50_symbols:
    gainers, losers = get_gainers_losers(nifty50_symbols)
    gainers_df = pd.DataFrame(gainers)
    losers_df = pd.DataFrame(losers)
    cols_g = st.columns(2)
    with cols_g[0]:
        st.subheader("Top 5 Gainers")
        st.dataframe(gainers_df if not gainers_df.empty else pd.DataFrame([{"symbol":"No data", "change":0}]))
    with cols_g[1]:
        st.subheader("Top 5 Losers")
        st.dataframe(losers_df if not losers_df.empty else pd.DataFrame([{"symbol":"No data", "change":0}]))
else:
    st.warning("No Nifty 50 symbols available.")

st.header(f"{selected_index} Price Trend & Indicators")
data_sel = nifty_df if selected_index == "NIFTY 50" else bank_df
if data_sel is not None:
    fig = plot_market(data_sel, selected_index)
    st.pyplot(fig)
else:
    st.error("No data to plot.")

st.caption("Powered by FintickAI | Data source: Yahoo Finance & Wikipedia")
