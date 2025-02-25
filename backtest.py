import pytz
from datetime import datetime, timedelta
from alpaca.data import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import StockBarsRequest
import pandas as pd
import numpy as np
import time
import threading
import asyncio
import logging
from typing import Optional, Tuple
import argparse

# ======================
# CONFIGURATION
# ======================
ALPACA_API_KEY = 'PK8DPGKPJLXDYXHZYR7S'
ALPACA_SECRET_KEY = 'o5UibXlKY3VTn5XP6JTQNye2HNbonThpzd6tuUdp'
SYMBOL = 'MU'
TRADE_QTY = 20000
RSI_PERIOD = 14
BUFFER_SIZE = RSI_PERIOD + 1
TIMEFRAME = TimeFrame.Minute
OVERBOUGHT = 90
OVERSOLD = 10
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
            self.prices = [bar.close for bar in bars[-BUFFER_SIZE:]]
            if len(self.prices) >= BUFFER_SIZE:
                self._calculate_rsi()

    def update(self, price):
        with self.lock:
            self.latest_price = price
            if len(self.prices) < BUFFER_SIZE:
                self.prices.append(price)
            else:
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
# BACKTEST RESULTS CLASS
# ======================
class BacktestResult:
    def __init__(self):
        self.trades = []
        self.portfolio_values = []
        self.current_position = 0
        self.cash = 1000000  # Starting cash
        self.commission = 0.01  # Per share commission
        self.trade_count = 0

    def add_trade(self, timestamp: datetime, side: str, price: float, quantity: int):
        self.trade_count += 1
        cost = quantity * price * (1 + self.commission)
        if side == OrderSide.BUY:
            self.cash -= cost
            self.current_position += quantity
        else:
            self.cash += cost
            self.current_position -= quantity

        self.trades.append({
            'trade_id': self.trade_count,
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'quantity': quantity,
            'portfolio_value': self.portfolio_value(price),
            'cash': self.cash,
            'position': self.current_position
        })

    def portfolio_value(self, current_price: float) -> float:
        return self.cash + (self.current_position * current_price)

    def summary(self):
        df = pd.DataFrame(self.trades)
        df['returns'] = df['portfolio_value'].pct_change()
        return {
            'total_trades': len(self.trades),
            'final_value': self.portfolio_values[-1] if self.portfolio_values else 0,
            'max_drawdown': (df['portfolio_value'].max() - df['portfolio_value'].min()) / df['portfolio_value'].max(),
            'profit_factor': (self.portfolio_values[-1] if self.portfolio_values else 0) / 1000000,
        }


# ======================
# TRADING BOT CORE
# ======================
class RSITradingBot:
    def __init__(self, backtest_mode: bool = False,
                 backtest_start: Optional[datetime] = None,
                 backtest_end: Optional[datetime] = None):
        self.backtest_mode = backtest_mode
        self.backtest_result = BacktestResult() if backtest_mode else None
        self.backtest_data = None
        self.backtest_index = 0

        if not backtest_mode:
            self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            self.data_stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        self.buffer = RSIBuffer()
        self.running = threading.Event()
        self.position_lock = threading.Lock()
        self.last_trade_time = 0
        self.trading_allowed = False

        if backtest_mode:
            self._load_backtest_data(backtest_start, backtest_end)

    def _load_backtest_data(self, start: datetime, end: datetime):
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        request = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TIMEFRAME,
            start=start,
            end=end,
            adjustment='raw'
        )
        bars = client.get_stock_bars(request).data[SYMBOL]
        self.backtest_data = [(bar.timestamp, bar.close) for bar in bars]
        logging.info(f"Loaded {len(self.backtest_data)} bars for backtesting")

    def _check_market_hours(self, timestamp: Optional[datetime] = None):
        if self.backtest_mode and timestamp:
            ny_time = timestamp.astimezone(pytz.timezone('America/New_York'))
        else:
            ny_time = pd.Timestamp.now(tz='America/New_York')

        market_open_time = pd.Timestamp("09:30:00").time()
        market_close_time = pd.Timestamp("16:00:00").time()

        if self.backtest_mode:
            return market_open_time <= ny_time.time() <= market_close_time
        else:
            try:
                clock = self.trading_client.get_clock()
                return clock.is_open
            except Exception as e:
                logging.error(f"Market check failed: {e}")
                return False

    async def handle_trade(self, trade):
        if self.backtest_mode:
            return

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
        if self.backtest_mode:
            self._run_backtest()
            return

        while self.running.is_set():
            try:
                if time.time() - self.last_trade_time > 300:
                    self.trading_allowed = self._check_market_hours()

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

    def _run_backtest(self):
        logging.info("Starting backtest...")
        print("\nBacktest Progress:")
        for idx, (timestamp, price) in enumerate(self.backtest_data, 1):
            if not self._check_market_hours(timestamp):
                continue

            # Update buffer and calculate RSI
            self.buffer.update(price)
            current_rsi = self.buffer.last_rsi

            # Get current position
            position = self.backtest_result.current_position

            # Print market data for each bar
            print(f"{timestamp.strftime('%Y-%m-%d %H:%M')} | Price: {price:.2f} | RSI: {current_rsi:.2f}", end='')

            # Check trading signals
            if current_rsi < OVERSOLD and position <= 0:
                self._submit_order(OrderSide.BUY, TRADE_QTY, timestamp, price)
                print(f" | BUY SIGNAL TRIGGERED", end='')
            elif current_rsi > OVERBOUGHT and position > 0:
                self._submit_order(OrderSide.SELL, position, timestamp, price)
                print(f" | SELL SIGNAL TRIGGERED", end='')

            # Update portfolio value
            current_value = self.backtest_result.portfolio_value(price)
            self.backtest_result.portfolio_values.append(current_value)
            print(f" | Portfolio Value: ${current_value:,.2f}")

            # Progress indicator
            # if idx % 10 == 0:
                # logging.info(f"Processed {idx}/{len(self.backtest_data)} bars")

        logging.info("Backtest complete")

        # Print detailed trade log
        print("\nTrade History:")
        trade_df = pd.DataFrame(self.backtest_result.trades)
        if not trade_df.empty:
            print(trade_df[['timestamp', 'side', 'price', 'quantity', 'portfolio_value']])

        # Print summary statistics
        print("\nBacktest Summary:")
        summary_df = pd.DataFrame(self.backtest_result.summary(), index=[0])
        print(summary_df)

    def _get_position(self):
        if self.backtest_mode:
            return self.backtest_result.current_position

        try:
            position = self.trading_client.get_open_position(SYMBOL)
            return int(position.qty)
        except:
            return 0

    def _submit_order(self, side: OrderSide, qty: int,
                      timestamp: Optional[datetime] = None,
                      price: Optional[float] = None):
        if self.backtest_mode:
            if timestamp and price:
                self.backtest_result.add_trade(timestamp, side, price, qty)
            return

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

    def start(self, backtest_days: int = 7):
        if self.backtest_mode:
            self.running.set()
            self.execute_strategy()
            return

        logging.info("Starting Live Trading Bot")
        self._preload_historical()
        self.running.set()

        self.data_stream.subscribe_trades(self.handle_trade, SYMBOL)
        threading.Thread(target=asyncio.run, args=(self.data_stream.run(),), daemon=True).start()
        threading.Thread(target=self.execute_strategy, daemon=True).start()

        try:
            while self.running.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

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

    def stop(self):
        if not self.backtest_mode:
            logging.info("Shutting down...")
            self.running.clear()
            asyncio.run(self.data_stream.close())


# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RSI Trading Bot')
    parser.add_argument('--backtest', action='store_true', help='Enable backtesting mode')
    parser.add_argument('--days', type=int, default=7, help='Number of days to backtest')
    args = parser.parse_args()

    if args.backtest:
        end_date = datetime.now(pytz.utc)
        start_date = end_date - timedelta(days=args.days)
        bot = RSITradingBot(
            backtest_mode=True,
            backtest_start=start_date,
            backtest_end=end_date
        )
        bot.start()
    else:
        bot = RSITradingBot()
        bot.start()