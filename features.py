"""
Feature engineering - computes technical indicators, market regime
features, and cross-sectional rankings for each stock.
"""

import pandas as pd
import numpy as np
import ta


def compute_momentum_features(df):
    """Compute momentum and trend features for a single stock."""
    close = df['Close']
    features = pd.DataFrame(index=df.index)

    # Returns over various lookback windows
    for w in [5, 10, 20, 40, 60, 120]:
        features[f'return_{w}d'] = close.pct_change(w)

    # Rate of Change
    for w in [5, 10, 20]:
        features[f'roc_{w}d'] = (close / close.shift(w) - 1) * 100

    # Moving Average Ratios
    for short, long in [(5, 20), (10, 50), (20, 100), (50, 200)]:
        sma_s = close.rolling(short).mean()
        sma_l = close.rolling(long).mean()
        features[f'sma_{short}_{long}_ratio'] = sma_s / sma_l

    # Price vs Moving Averages
    for w in [20, 50, 200]:
        sma = close.rolling(w).mean()
        features[f'close_vs_sma{w}'] = close / sma - 1

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # ADX
    try:
        adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], close, window=14)
        features['adx'] = adx_indicator.adx()
    except:
        features['adx'] = np.nan

    # Consecutive up/down days
    daily_ret = close.pct_change()
    features['consec_up'] = daily_ret.gt(0).groupby((~daily_ret.gt(0)).cumsum()).cumsum()
    features['consec_down'] = daily_ret.lt(0).groupby((~daily_ret.lt(0)).cumsum()).cumsum()

    # Closed-form rolling least-squares slope avoids a separate polyfit for
    # every stock-day while preserving the original normalized slope.
    window = 20
    centered_time = np.arange(window, dtype=float) - (window - 1) / 2
    slope_denominator = float(np.dot(centered_time, centered_time))
    close_values = pd.to_numeric(close, errors='coerce').to_numpy(dtype=float)
    weighted_sums = np.convolve(
        np.nan_to_num(close_values, nan=0.0), centered_time[::-1], mode='valid'
    )
    rolling_mean = close.rolling(window, min_periods=window).mean().to_numpy(dtype=float)
    slopes = weighted_sums / slope_denominator / rolling_mean[window - 1:]
    slope_values = np.full(len(close), np.nan, dtype=float)
    slope_values[window - 1:] = slopes
    slope_values[close.rolling(window, min_periods=window).count().to_numpy() < window] = np.nan
    features['lr_slope_20'] = slope_values

    return features


def compute_volatility_features(df):
    """Compute volatility features."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    open_ = df['Open']
    features = pd.DataFrame(index=df.index)
    daily_ret = close.pct_change()

    # Rolling StdDev
    for w in [5, 10, 20, 60]:
        features[f'vol_{w}d'] = daily_ret.rolling(w).std()

    # Volatility ratio
    features['vol_ratio_5_20'] = features['vol_5d'] / features['vol_20d']

    # ATR
    for w in [7, 14]:
        atr = ta.volatility.AverageTrueRange(high, low, close, window=w)
        features[f'atr_{w}'] = atr.average_true_range() / close

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    features['bb_pct'] = bb.bollinger_pband()
    features['bb_width'] = bb.bollinger_wband()

    # Garman-Klass volatility
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk_variance = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean()
    features['gk_vol'] = np.sqrt(gk_variance.clip(lower=0))

    # Intraday range
    features['intraday_range'] = (high - low) / close

    # Overnight gap
    features['overnight_gap'] = open_ / close.shift(1) - 1

    # Skewness and Kurtosis
    features['return_skew_20'] = daily_ret.rolling(20).skew()
    features['return_kurt_20'] = daily_ret.rolling(20).kurt()

    return features


def compute_volume_features(df):
    """Compute volume-based features."""
    close = df['Close']
    volume = df['Volume']
    high = df['High']
    low = df['Low']
    features = pd.DataFrame(index=df.index)

    # Relative volume
    for w in [20, 50]:
        features[f'rel_volume_{w}d'] = volume / volume.rolling(w).mean()

    # OBV rate of change
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
    obv_values = obv.on_balance_volume()
    features['obv_roc_5'] = obv_values.pct_change(5)
    features['obv_roc_10'] = obv_values.pct_change(10)

    # Money Flow Index
    mfi = ta.volume.MFIIndicator(high, low, close, volume, window=14)
    features['mfi'] = mfi.money_flow_index()

    # Chaikin Money Flow
    cmf = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, window=20)
    features['cmf'] = cmf.chaikin_money_flow()

    # Volume-price trend confirmation
    price_dir = np.sign(close.pct_change())
    vol_dir = np.sign(volume.pct_change())
    features['vol_price_confirm'] = (price_dir * vol_dir).rolling(5).mean()

    # Force Index
    features['force_index'] = (close.pct_change() * volume).ewm(span=13).mean()

    # Volume spike
    volume_mean_20 = volume.rolling(20, min_periods=20).mean()
    features['volume_spike'] = (volume > 2 * volume_mean_20).where(volume_mean_20.notna())

    return features


def compute_oscillator_features(df):
    """Compute mean-reversion / oscillator features."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    features = pd.DataFrame(index=df.index)

    # RSI (multiple timeframes)
    for w in [7, 14, 21]:
        rsi = ta.momentum.RSIIndicator(close, window=w)
        features[f'rsi_{w}'] = rsi.rsi()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    features['stoch_k'] = stoch.stoch()
    features['stoch_d'] = stoch.stoch_signal()

    # Williams %R
    wr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
    features['williams_r'] = wr.williams_r()

    # CCI
    cci = ta.trend.CCIIndicator(high, low, close, window=20)
    features['cci'] = cci.cci()

    # Distance from 52-week high/low
    features['dist_52w_high'] = close / close.rolling(252).max() - 1
    features['dist_52w_low'] = close / close.rolling(252).min() - 1

    # Mean reversion z-score
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features['zscore_20'] = (close - sma20) / std20

    return features


def compute_candle_features(df):
    """Compute candlestick / price pattern features."""
    close = df['Close']
    open_ = df['Open']
    high = df['High']
    low = df['Low']
    features = pd.DataFrame(index=df.index)

    body = (close - open_).abs()
    full_range = high - low + 1e-10

    features['candle_body_ratio'] = body / full_range
    features['upper_shadow'] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / full_range
    features['lower_shadow'] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / full_range

    # Autocorrelation
    daily_ret = close.pct_change()
    features['autocorr_1'] = daily_ret.rolling(20, min_periods=20).corr(daily_ret.shift(1))

    return features


def compute_calendar_features(df):
    """Compute calendar / seasonal features."""
    features = pd.DataFrame(index=df.index)
    features['day_of_week'] = df.index.dayofweek
    features['week_of_month'] = (df.index.day - 1) // 7 + 1
    features['month'] = df.index.month
    features['quarter_end'] = df.index.month.isin([3, 6, 9, 12]).astype(int)

    # Cyclical encoding
    features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    return features


def compute_market_regime_features(stock_df, spy_data, vix_data=None):
    """Compute market regime features using SPY and VIX."""
    features = pd.DataFrame(index=stock_df.index)

    # Align indices
    spy_close = spy_data['Close'].reindex(stock_df.index, method='ffill')
    vix_close = None
    if vix_data is not None and not vix_data.empty and 'Close' in vix_data:
        vix_close = vix_data['Close'].reindex(stock_df.index, method='ffill')

    # SPY returns
    for w in [5, 10, 20]:
        features[f'spy_return_{w}d'] = spy_close.pct_change(w)

    # SPY trend
    spy_sma50 = spy_close.rolling(50).mean()
    spy_sma200 = spy_close.rolling(200).mean()
    features['spy_trend'] = spy_sma50 / spy_sma200

    # VIX is optional for providers that do not carry the volatility index.
    if vix_close is not None:
        features['vix_level'] = vix_close
        features['vix_change_5d'] = vix_close.pct_change(5)
        features['vix_pct_rank'] = vix_close.rolling(60).rank(pct=True)

    # Stock beta (rolling 60-day)
    stock_ret = stock_df['Close'].pct_change()
    spy_ret = spy_close.pct_change()

    def rolling_beta(stock_r, market_r, window=60):
        cov = stock_r.rolling(window).cov(market_r)
        var = market_r.rolling(window).var()
        return cov / var

    features['beta_60d'] = rolling_beta(stock_ret, spy_ret, 60)

    # Correlation with market
    features['corr_spy_20d'] = stock_ret.rolling(20).corr(spy_ret)

    # Market breadth (filled in later at cross-sectional level)

    return features


def compute_all_features(stock_data, spy_data, vix_data, tickers, *, include_cross_sectional=True):
    """
    Compute all features for all stocks. Returns a dict of DataFrames.
    stock_data: dict of {ticker: DataFrame with OHLCV}
    """
    all_features = {}

    for ticker in tickers:
        df = stock_data[ticker].copy()

        # Compute feature groups
        momentum = compute_momentum_features(df)
        volatility = compute_volatility_features(df)
        volume = compute_volume_features(df)
        oscillator = compute_oscillator_features(df)
        candle = compute_candle_features(df)
        calendar = compute_calendar_features(df)
        regime = compute_market_regime_features(df, spy_data, vix_data)

        # Combine all features
        features = pd.concat([momentum, volatility, volume, oscillator,
                              candle, calendar, regime], axis=1)

        # Add ticker column
        features['ticker'] = ticker

        all_features[ticker] = features

    # Compute cross-sectional features
    if include_cross_sectional:
        all_features = compute_cross_sectional_features(all_features, tickers)

    return all_features


def compute_cross_sectional_features(all_features, tickers):
    """Compute cross-sectional (relative) features across stocks."""
    # Collect return and indicator data across stocks for each date
    dates = all_features[tickers[0]].index

    ret_5d = pd.DataFrame({t: all_features[t]['return_5d'] for t in tickers})
    ret_20d = pd.DataFrame({t: all_features[t]['return_20d'] for t in tickers})
    rsi_14 = pd.DataFrame({t: all_features[t]['rsi_14'] for t in tickers})
    vol_20d = pd.DataFrame({t: all_features[t]['vol_20d'] for t in tickers})
    rel_vol = pd.DataFrame({t: all_features[t]['rel_volume_20d'] for t in tickers})

    for ticker in tickers:
        # Rank of return among peers (percentile)
        all_features[ticker]['cs_ret5d_rank'] = ret_5d.rank(axis=1, pct=True)[ticker]
        all_features[ticker]['cs_ret20d_rank'] = ret_20d.rank(axis=1, pct=True)[ticker]
        all_features[ticker]['cs_rsi_rank'] = rsi_14.rank(axis=1, pct=True)[ticker]
        all_features[ticker]['cs_vol_rank'] = vol_20d.rank(axis=1, pct=True)[ticker]
        all_features[ticker]['cs_relvol_rank'] = rel_vol.rank(axis=1, pct=True)[ticker]

        # Z-score vs universe
        all_features[ticker]['cs_ret5d_zscore'] = (
            (ret_5d[ticker] - ret_5d.mean(axis=1)) / ret_5d.std(axis=1)
        )

        # Relative strength: stock return / universe mean return
        all_features[ticker]['cs_rel_strength'] = (
            ret_5d[ticker] / ret_5d.mean(axis=1).replace(0, np.nan)
        )

        # Market dispersion (same for all stocks)
        all_features[ticker]['cs_dispersion'] = ret_5d.std(axis=1)

        # Market breadth
        available = ret_5d.notna().sum(axis=1).replace(0, np.nan)
        all_features[ticker]['cs_breadth'] = (ret_5d > 0).sum(axis=1) / available

    return all_features


def compute_lagged_target_features(weekly_df, tickers):
    """
    Compute lagged target / return-based features at weekly level.
    These use PAST targets only — no look-ahead bias.
    """
    weekly_df = weekly_df.copy()
    ticker_groups = weekly_df.groupby('ticker', sort=False)

    # Grouped shifts replace repeated full-frame ticker masks and keep each
    # security era isolated, including when a ticker symbol is reused.
    for lag in [1, 2, 3, 4]:
        weekly_df[f'lag_return_{lag}w'] = ticker_groups['forward_return'].shift(lag + 1)

    # Open-to-open labels need two calendar rows to mature before the next
    # signal; every rolling statistic below uses only those matured outcomes.
    realized_return = ticker_groups['forward_return'].shift(2)
    past_positive = realized_return.gt(0).astype(float).where(realized_return.notna())
    positive_groups = past_positive.groupby(weekly_df['ticker'], sort=False)
    realized_groups = realized_return.groupby(weekly_df['ticker'], sort=False)
    for window in [4, 8, 13, 26]:
        weekly_df[f'win_rate_{window}w'] = positive_groups.transform(
            lambda values: values.rolling(window, min_periods=2).mean()
        )
    for window in [4, 8, 13]:
        weekly_df[f'avg_past_ret_{window}w'] = realized_groups.transform(
            lambda values: values.rolling(window, min_periods=2).mean()
        )
    weekly_df['reversal_1w'] = -realized_return
    for window in [8, 13]:
        weekly_df[f'ret_vol_{window}w'] = realized_groups.transform(
            lambda values: values.rolling(window, min_periods=4).std()
        )

    def trailing_drawdown(values):
        wealth = np.cumprod(1.0 + values)
        peaks = np.maximum.accumulate(wealth)
        return float(np.min(wealth / peaks - 1.0))

    weekly_df['past_maxdd_13w'] = realized_groups.transform(
        lambda values: (1.0 + values).rolling(13, min_periods=4).apply(
            trailing_drawdown, raw=True
        )
    )
    return weekly_df


def prepare_weekly_dataset(all_features, stock_data, tickers, dividends=None):
    """
    Resample to weekly, create target variable, and combine into single DataFrame.
    """
    weekly_dfs = []

    for ticker in tickers:
        feat = all_features[ticker].copy()
        price = stock_data[ticker]['Close'].copy()

        # Resample features to the last available close in each Friday-ending
        # week. Keep the actual signal date for holiday weeks.
        weekly_feat = feat.resample('W-FRI').last()

        # A close-derived signal cannot realistically be filled at that same
        # close. Simulate execution at the next trading week's open and hold to
        # the following week's open.
        weekly_price = price.resample('W-FRI').last()
        open_price = stock_data[ticker]['Open'].astype(float)
        weekly_open = open_price.resample('W-FRI').first()
        weekly_first_date = pd.Series(open_price.index, index=open_price.index).resample('W-FRI').first()
        weekly_signal_date = pd.Series(feat.index, index=feat.index).resample('W-FRI').last()

        entry_price = weekly_open.shift(-1).reindex(weekly_feat.index)
        exit_price = weekly_open.shift(-2).reindex(weekly_feat.index)
        entry_date = weekly_first_date.shift(-1).reindex(weekly_feat.index)
        label_end_date = weekly_first_date.shift(-2).reindex(weekly_feat.index)
        signal_date = weekly_signal_date.reindex(weekly_feat.index)
        cash_dividends = pd.Series(0.0, index=weekly_feat.index)
        dividend_events = (dividends or {}).get(ticker)
        if dividend_events is not None and not dividend_events.empty:
            event_dates = pd.DatetimeIndex(dividend_events.index).tz_localize(None).normalize().to_numpy(dtype='datetime64[ns]')
            event_amounts = pd.to_numeric(dividend_events, errors='coerce').fillna(0.0).to_numpy(dtype=float)
            cumulative_amounts = np.concatenate(([0.0], np.cumsum(event_amounts)))
            valid_dates = entry_date.notna() & label_end_date.notna()
            if valid_dates.any():
                entry_values = pd.DatetimeIndex(entry_date.loc[valid_dates]).tz_localize(None).normalize().to_numpy(dtype='datetime64[ns]')
                exit_values = pd.DatetimeIndex(label_end_date.loc[valid_dates]).tz_localize(None).normalize().to_numpy(dtype='datetime64[ns]')
                first_event = np.searchsorted(event_dates, entry_values, side='right')
                last_event = np.searchsorted(event_dates, exit_values, side='right')
                cash_dividends.loc[valid_dates] = cumulative_amounts[last_event] - cumulative_amounts[first_event]
        forward_return = (exit_price + cash_dividends) / entry_price - 1
        weekly_feat['forward_return'] = forward_return
        weekly_feat['target'] = forward_return.gt(0).where(forward_return.notna()).astype(float)
        weekly_feat['cash_dividends'] = cash_dividends

        weekly_feat['ticker'] = ticker
        weekly_feat['close'] = weekly_price
        weekly_feat['entry_price'] = entry_price
        weekly_feat['exit_price'] = exit_price
        weekly_feat['signal_date'] = signal_date
        weekly_feat['entry_date'] = entry_date
        weekly_feat['label_end_date'] = label_end_date

        weekly_dfs.append(weekly_feat)

    combined = pd.concat(weekly_dfs, axis=0).sort_index()

    # Keep weeks whose future return is still unknown. The walk-forward engine
    # excludes these labels until their scheduled exit date; retaining the rows
    # preserves true calendar-week offsets for lagged target features.

    return combined


def compute_weekly_cross_sectional_features(weekly_df):
    """Add relative features using each signal week's actual eligible universe."""
    if weekly_df.empty:
        return weekly_df
    weekly_df = weekly_df.copy()
    grouped = weekly_df.groupby(level=0, sort=False)
    for column, output in (
        ('return_5d', 'cs_ret5d_rank'),
        ('return_20d', 'cs_ret20d_rank'),
        ('rsi_14', 'cs_rsi_rank'),
        ('vol_20d', 'cs_vol_rank'),
        ('rel_volume_20d', 'cs_relvol_rank'),
    ):
        weekly_df[output] = grouped[column].rank(pct=True) if column in weekly_df else np.nan
    if 'return_5d' in weekly_df:
        means = grouped['return_5d'].transform('mean')
        deviations = grouped['return_5d'].transform('std')
        weekly_df['cs_ret5d_zscore'] = (weekly_df['return_5d'] - means) / deviations.replace(0, np.nan)
        weekly_df['cs_rel_strength'] = weekly_df['return_5d'] / means.replace(0, np.nan)
        weekly_df['cs_dispersion'] = deviations
        positive = weekly_df['return_5d'].gt(0).groupby(level=0).transform('sum')
        available = weekly_df['return_5d'].notna().groupby(level=0).transform('sum').replace(0, np.nan)
        weekly_df['cs_breadth'] = positive / available
    return weekly_df
