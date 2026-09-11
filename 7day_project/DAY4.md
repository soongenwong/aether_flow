Day 4: Resampling & Pandas Analytical Layer (4 Hours)
Goal: Provide a clean API for strategies or risk desks to query live and
historical slices.
Tasks:
Write an API method engine.get_bars(symbol, timeframe="1s") that
extracts a view of the circular buffer, converts it into a pd.DataFrame,
and resamples it to OHLCV bars.
Add multi-asset capabilities: Track 10 symbols simultaneously and
compute a rolling dynamic correlation matrix with pd.DataFrame.corr().