import pytz
from datetime import datetime, timedelta
from alpaca.data import TimeFrame
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import pandas as pd
import numpy as np
import time
import asyncio
import logging
from typing import Optional
import argparse

# ======================
# CONFIGURATION
# ======================
ALPACA_API_KEY = 'PKYRJ7EDWUB6SLBS9XV5'
ALPACA_SECRET_KEY = 'uBuv3cZRY3jsEsNkiK6dbmJHqbN83brmLbtMlip2'
SYMBOL = 'SPY'
TRADE_QTY = 100
RSI_PERIOD = 5
MA_PERIOD = 10                      # Moving average period
UP_TREND_RSI_ENTRY = 31             # Uptrend entry threshold
DOWN_TREND_RSI_EXIT = 89            # Downtrend exit threshold
RSI_EXIT_LEVEL = 40                 # Centerline exit level
BUFFER_SIZE = max(RSI_PERIOD + 1, MA_PERIOD)  # Buffer for MA calculations
TIMEFRAME = TimeFrame.Minute  # Ensures we're using 1-minute bars
POLL_INTERVAL = 60  # Adjust to 60 seconds for 1-minute bars
COOLDOWN = 60

# ======================
# LOGGING SETUP
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# ======================
# DATA BUFFER CLASS
# ======================
class RSIBuffer:
    def __init__(self):
        self.prices = []
        self.avg_gain = None
        self.avg_loss = None
        self.last_rsi = 50.0
        self.previous_rsi = 50.0  # Track previous RSI value
        self.latest_price = None

    def add_historical_bars(self, bars):
        self.prices = [bar.close for bar in bars[-BUFFER_SIZE:]]
        if len(self.prices) >= RSI_PERIOD + 1:
            self._calculate_rsi()

    def update(self, price):
        self.latest_price = price
        if len(self.prices) < BUFFER_SIZE:
            self.prices.append(price)
        else:
            self.prices = self.prices[1:] + [price]
        if len(self.prices) >= RSI_PERIOD + 1:
            self._calculate_rsi()

    def _calculate_rsi(self):
        self.previous_rsi = self.last_rsi
        if len(self.prices) < BUFFER_SIZE:
            return

        deltas = np.diff(self.prices)
        gains = [x if x > 0 else 0 for x in deltas]
        losses = [-x if x < 0 else 0 for x in deltas]

        if self.avg_gain is None or self.avg_loss is None:
            self.avg_gain = sum(gains) / RSI_PERIOD
            self.avg_loss = sum(losses) / RSI_PERIOD
        else:
            self.avg_gain = (self.avg_gain * (RSI_PERIOD - 1) + gains[-1]) / RSI_PERIOD
            self.avg_loss = (self.avg_loss * (RSI_PERIOD - 1) + losses[-1]) / RSI_PERIOD

        if self.avg_loss == 0:
            self.last_rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.last_rsi = 100.0 - (100.0 / (1 + rs))

    def get_ma(self):
        """Get moving average for trend analysis"""
        if len(self.prices) >= MA_PERIOD:
            return sum(self.prices[-MA_PERIOD:]) / MA_PERIOD
        return None

# ======================
# TRADING BOT CORE
# ======================
class RSITradingBot:
    def __init__(self):
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        self.data_stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.buffer = RSIBuffer()
        self.running = True
        self.last_trade_time = 0
        self.trading_allowed = False

    def _check_market_hours(self, timestamp: Optional[datetime] = None):
        ny_time = timestamp.astimezone(pytz.timezone('America/New_York')) if timestamp else pd.Timestamp.now(tz='America/New_York')
        if ny_time.weekday() >= 5:
            return False
        market_open = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= ny_time <= market_close

    async def handle_trade(self, trade):
        try:
            if trade.price and trade.timestamp:
                price = float(trade.price)
                self.buffer.update(price)
                self._display_status()
        except Exception as e:
            logging.error(f"Trade error: {e}")

    def _display_status(self):
        status = f"Price: {self.buffer.latest_price:.2f} | RSI: {self.buffer.last_rsi:.1f}"
        print(status.ljust(80), end='\r')

    async def execute_strategy(self):
        while self.running:
            try:
                if time.time() - self.last_trade_time > COOLDOWN:
                    self.trading_allowed = self._check_market_hours()

                if not self.trading_allowed:
                    await asyncio.sleep(60)
                    continue

                position = self._get_position()
                current_price = self.buffer.latest_price
                ma = self.buffer.get_ma()

                if ma is None:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                uptrend = current_price > ma
                current_rsi = self.buffer.last_rsi
                previous_rsi = self.buffer.previous_rsi

                if uptrend:
                    if previous_rsi < UP_TREND_RSI_ENTRY and current_rsi >= UP_TREND_RSI_ENTRY:
                        if position <= 0:
                            self._submit_order(OrderSide.BUY, TRADE_QTY)
                            self.last_trade_time = time.time()
                    elif position > 0 and previous_rsi >= RSI_EXIT_LEVEL and current_rsi < RSI_EXIT_LEVEL:
                        self._submit_order(OrderSide.SELL, position)
                        self.last_trade_time = time.time()
                else:
                    if previous_rsi > DOWN_TREND_RSI_EXIT and current_rsi <= DOWN_TREND_RSI_EXIT:
                        if position >= 0:
                            self._submit_order(OrderSide.SELL, TRADE_QTY)
                            self.last_trade_time = time.time()
                    elif position < 0 and previous_rsi <= RSI_EXIT_LEVEL and current_rsi > RSI_EXIT_LEVEL:
                        self._submit_order(OrderSide.BUY, abs(position))
                        self.last_trade_time = time.time()

                await asyncio.sleep(POLL_INTERVAL)

            except Exception as e:
                logging.error(f"Strategy error: {e}")
                await asyncio.sleep(POLL_INTERVAL)

    def _get_position(self):
        try:
            position = self.trading_client.get_open_position(SYMBOL)
            return int(position.qty)
        except Exception as e:
            if 'position does not exist' in str(e):
                return 0
            else:
                logging.error(f"Position error: {e}")
                return 0

    def _submit_order(self, side: OrderSide, qty: int):
        if time.time() - self.last_trade_time < COOLDOWN:
            return

        try:
            order = MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            self.trading_client.submit_order(order)
            logging.info(f"{side} {qty} {SYMBOL}")
        except Exception as e:
            logging.error(f"Order failed: {e}")

    async def run(self):
        logging.info("Starting Live Trading Bot")
        self.running = True

        self.data_stream.subscribe_trades(self.handle_trade, SYMBOL)
        await asyncio.gather(
            self.data_stream._run_forever(),
            self.execute_strategy()
        )

    async def stop(self):
        logging.info("Shutting down...")
        self.running = False
        await self.data_stream.close()

# ======================
# MAIN EXECUTION
# ======================
def main():
    parser = argparse.ArgumentParser(description='RSI Trading Bot')
    parser.add_argument('--dummy', action='store_true', help='Dummy arg to mimic original script')
    args = parser.parse_args()

    bot = RSITradingBot()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        loop.run_until_complete(bot.stop())

if __name__ == "__main__":
    main()
