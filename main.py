import pytz
from alpaca.data import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.requests import StockBarsRequest
import pandas as pd
import numpy as np
import time
import threading
import asyncio
import logging

# ======================
# CONFIGURATION
# ======================
ALPACA_API_KEY = 'PK8DPGKPJLXDYXHZYR7S'
ALPACA_SECRET_KEY = 'o5UibXlKY3VTn5XP6JTQNye2HNbonThpzd6tuUdp'
SYMBOL = 'MU'
TRADE_QTY = 20000
RSI_PERIOD = 2
BUFFER_SIZE = RSI_PERIOD + 1  # 3 data points needed for 2-period RSI
TIMEFRAME = TimeFrame.Minute
OVERBOUGHT = 80
OVERSOLD = 20
POLL_INTERVAL = 1
COOLDOWN = 1

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
        self.lock = threading.Lock()
        self.avg_gain = None
        self.avg_loss = None
        self.last_rsi = 50.0  # Initialize with neutral value
        self.latest_price = None

    def add_historical_bars(self, bars):
        with self.lock:
            # Load last 3 closing prices
            self.prices = [bar.close for bar in bars[-BUFFER_SIZE:]]
            if len(self.prices) >= BUFFER_SIZE:
                self._calculate_rsi()

    def update(self, price):
        with self.lock:
            self.latest_price = price
            if len(self.prices) < BUFFER_SIZE:
                self.prices.append(price)
            else:
                # Maintain rolling window of 3 prices
                self.prices = self.prices[1:] + [price]
            self._calculate_rsi()

    def _calculate_rsi(self):
        if len(self.prices) < BUFFER_SIZE:
            return

        deltas = np.diff(self.prices)
        gains = [x if x > 0 else 0 for x in deltas]
        losses = [-x if x < 0 else 0 for x in deltas]

        # Wilder's smoothing (RSI Standard)
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


# ======================
# TRADING BOT CORE
# ======================
class RSITradingBot:
    def __init__(self):
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        self.data_stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.buffer = RSIBuffer()
        self.running = threading.Event()
        self.position_lock = threading.Lock()
        self.last_trade_time = 0
        self.trading_allowed = False

    def _check_market_hours(self):
        try:
            clock = self.trading_client.get_clock()
            ny_time = pd.Timestamp.now(tz='America/New_York')
            market_open = clock.is_open
            pre_market = ny_time.time() >= pd.Timestamp("04:00:00").time()
            post_market = ny_time.time() <= pd.Timestamp("20:00:00").time()
            self.trading_allowed = market_open or (pre_market and post_market)
        except Exception as e:
            logging.error(f"Market check failed: {e}")

    def _preload_historical(self):
        try:
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            request = StockBarsRequest(
                symbol_or_symbols=SYMBOL,
                timeframe=TIMEFRAME,
                limit=BUFFER_SIZE * 2,
                adjustment='raw'
            )
            bars = client.get_stock_bars(request).data[SYMBOL]
            self.buffer.add_historical_bars(bars)
        except Exception as e:
            logging.error(f"Historical data error: {e}")

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

    def execute_strategy(self):
        while self.running.is_set():
            try:
                # Check market hours every 5 minutes
                if time.time() - self.last_trade_time > 300:
                    self._check_market_hours()

                if not self.trading_allowed:
                    time.sleep(60)
                    continue

                position = self._get_position()
                current_rsi = self.buffer.last_rsi

                if current_rsi < OVERSOLD and position <= 0:
                    self._submit_order(OrderSide.BUY, TRADE_QTY)
                    self.last_trade_time = time.time()
                elif current_rsi > OVERBOUGHT and position > 0:
                    self._submit_order(OrderSide.SELL, position)
                    self.last_trade_time = time.time()

                time.sleep(POLL_INTERVAL)

            except Exception as e:
                logging.error(f"Strategy error: {e}")

    def _get_position(self):
        try:
            position = self.trading_client.get_open_position(SYMBOL)
            return int(position.qty)
        except:
            return 0

    def _submit_order(self, side, qty):
        try:
            if time.time() - self.last_trade_time < COOLDOWN:
                return

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

    def start(self):
        logging.info("Starting RSI Trading Bot")
        self._preload_historical()
        self.running.set()

        # Start WebSocket
        self.data_stream.subscribe_trades(self.handle_trade, SYMBOL)
        threading.Thread(target=asyncio.run, args=(self.data_stream.run(),), daemon=True).start()

        # Start strategy thread
        threading.Thread(target=self.execute_strategy, daemon=True).start()

        try:
            while self.running.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logging.info("Shutting down...")
        self.running.clear()
        asyncio.run(self.data_stream.close())


if __name__ == "__main__":
    bot = RSITradingBot()
    bot.start()