import os
import yfinance as yf
import pandas as pd
import requests

def get_prices(ticker, start_date, end_date):
    """Fetch price data using yfinance with improved error handling."""
    try:
        ticker_data = yf.Ticker(ticker)
        
        # Try different intervals if data is missing
        intervals = ['1d', '1wk']
        df = None
        
        for interval in intervals:
            df = ticker_data.history(start=start_date, end=end_date, interval=interval)
            if not df.empty:
                break
        
        if df is None or df.empty:
            raise ValueError(f"No price data available for {ticker} between {start_date} and {end_date}")
        
        # Forward fill any missing values
        df = df.ffill()
        
        # Ensure we have all required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert weekly data to daily if needed
        if interval == '1wk':
            df = df.resample('D').ffill()
        
        return df.reset_index().to_dict('records')
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")

def prices_to_df(prices):
    """Convert prices to a DataFrame."""
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.columns = [col.lower() for col in df.columns]
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

def get_price_data(ticker, start_date, end_date):
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)

def get_financial_metrics(ticker, report_period, period='ttm', limit=1):
    """Fetch financial metrics using yfinance."""
    try:
        ticker_data = yf.Ticker(ticker)
        info = ticker_data.info
        metrics = {
            'marketCap': info.get('marketCap'),
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'priceToBook': info.get('priceToBook'),
            'profitMargins': info.get('profitMargins'),
            'returnOnEquity': info.get('returnOnEquity'),
            'returnOnAssets': info.get('returnOnAssets'),
            'debtToEquity': info.get('debtToEquity'),
            'currentRatio': info.get('currentRatio'),
            'beta': info.get('beta')
        }
        return [metrics]  # Return as list to maintain compatibility with existing code
    except Exception as e:
        raise Exception(f"Error fetching financial metrics: {str(e)}")

def calculate_confidence_level(signals):
    """Calculate confidence level based on the difference between SMAs."""
    sma_diff_prev = abs(signals['sma_5_prev'] - signals['sma_20_prev'])
    sma_diff_curr = abs(signals['sma_5_curr'] - signals['sma_20_curr'])
    diff_change = sma_diff_curr - sma_diff_prev
    # Normalize confidence between 0 and 1
    confidence = min(max(diff_change / signals['current_price'], 0), 1)
    return confidence

def calculate_macd(prices_df):
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_rsi(prices_df, period=14):
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices_df, window=20):
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

def calculate_obv(prices_df):
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] - prices_df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df['OBV'] = obv
    return prices_df['OBV']