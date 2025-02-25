from alpaca.data import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
import numpy as np

# ======================
# CONFIGURATION
# ======================
ALPACA_API_KEY = 'PK8DPGKPJLXDYXHZYR7S'
ALPACA_SECRET_KEY = 'o5UibXlKY3VTn5XP6JTQNye2HNbonThpzd6tuUdp'
SYMBOL = 'MU'
RSI_PERIOD = 2
BUFFER_SIZE = RSI_PERIOD + 1  # 3 data points for 2-period RSI
TIMEFRAME = TimeFrame.Day


# ======================
# RSI CALCULATION CLASS (Wilder's Smoothing)
# ======================
class RSIBuffer:
    def __init__(self):
        self.prices = []
        self.avg_gain = None
        self.avg_loss = None
        self.last_rsi = 0

    def add_historical_bars(self, bars):
        self.prices = [bar.close for bar in bars]
        if len(self.prices) >= BUFFER_SIZE:
            self._calculate_initial_avg()
            self.update_rsi()

    def _calculate_initial_avg(self):
        deltas = np.diff(self.prices[:RSI_PERIOD + 1])
        gains = [x if x > 0 else 0 for x in deltas]
        losses = [-x if x < 0 else 0 for x in deltas]

        self.avg_gain = sum(gains) / RSI_PERIOD
        self.avg_loss = sum(losses) / RSI_PERIOD

    def update_rsi(self):
        # Process remaining prices with Wilder's smoothing
        for price in self.prices[RSI_PERIOD + 1:]:
            delta = price - self.prices[-1]
            gain = max(delta, 0)
            loss = max(-delta, 0)

            self.avg_gain = (self.avg_gain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD
            self.avg_loss = (self.avg_loss * (RSI_PERIOD - 1) + loss) / RSI_PERIOD

        if self.avg_loss == 0:
            self.last_rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.last_rsi = 100.0 - (100.0 / (1 + rs))


# ======================
# MAIN FUNCTION
# ======================
def get_current_rsi():
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    buffer = RSIBuffer()

    try:
        request = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TIMEFRAME,
            limit=14,  # Fetch extra data for proper Wilder's initialization
            adjustment='raw'
        )
        bars = client.get_stock_bars(request).data[SYMBOL]

        # Verify data matches Yahoo's closes
        print("Fetched closing prices:")
        for bar in bars[-BUFFER_SIZE:]:
            print(f"{bar.timestamp.date()}: {bar.close}")

        buffer.add_historical_bars(bars)
        buffer.update_rsi()
        return buffer.last_rsi
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    current_rsi = get_current_rsi()
    if current_rsi is not None:
        print(f"\nCurrent RSI({RSI_PERIOD}) for {SYMBOL}: {current_rsi:.2f}")
    else:
        print("Failed to retrieve RSI value")