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

# Enable full column display
pd.set_option('display.max_columns', None)

# ======================
# CONFIGURATION
# ======================
ALPACA_API_KEY = 'PK8DPGKPJLXDYXHZYR7S'
ALPACA_SECRET_KEY = 'o5UibXlKY3VTn5XP6JTQNye2HNbonThpzd6tuUdp'
SYMBOL = 'MU'
TRADE_QTY = 100
RSI_PERIOD = 5
MA_PERIOD = 10
UP_TREND_RSI_ENTRY = 34
DOWN_TREND_RSI_EXIT = 58
RSI_EXIT_LEVEL = 35
BUFFER_SIZE = max(RSI_PERIOD + 1, MA_PERIOD)  # MODIFIED: Buffer for MA calculations
TIMEFRAME = TimeFrame.Minute
POLL_INTERVAL = 1
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
        self.lock = threading.Lock()
        self.avg_gain = None
        self.avg_loss = None
        self.last_rsi = 50.0
        self.previous_rsi = 50.0     # ADDED: Track previous RSI value
        self.latest_price = None

    def add_historical_bars(self, bars):
        with self.lock:
            # MODIFIED: Store enough prices for MA calculation
            self.prices = [bar.close for bar in bars[-BUFFER_SIZE:]]
            if len(self.prices) >= RSI_PERIOD + 1:
                self._calculate_rsi()

    def update(self, price):
        with self.lock:
            self.latest_price = price
            if len(self.prices) < BUFFER_SIZE:
                self.prices.append(price)
            else:
                self.prices = self.prices[1:] + [price]
            if len(self.prices) >= RSI_PERIOD + 1:
                self._calculate_rsi()

    def _calculate_rsi(self):
        # Store previous RSI before calculation
        self.previous_rsi = self.last_rsi  # ADDED: Track previous value
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

    def get_ma(self):
        """Get moving average for trend analysis"""
        if len(self.prices) >= MA_PERIOD:
            return sum(self.prices[-MA_PERIOD:]) / MA_PERIOD
        return None

# ======================
# BACKTEST RESULTS CLASS
# ======================
class BacktestResult:
    def __init__(self):
        self.trades = []            # Individual trade logs
        self.trade_pairs = []       # Paired buy-sell trades for review
        self.open_trade = None      # Holds active buy trade
        self.portfolio_values = []
        self.current_position = 0
        self.cash = 10000           # Starting cash
        self.commission = 0.01      # Per share commission
        self.trade_count = 0

    def add_trade(self, timestamp: datetime, side: str, price: float, quantity: int, rsi: Optional[float]=None):
        self.trade_count += 1
        cost = quantity * price * (1 + self.commission)
        if side == OrderSide.BUY:
            self.cash -= cost
            self.current_position += quantity
            # Store the buy trade details for pairing later
            self.open_trade = {
                'buy_trade_id': self.trade_count,
                'buy_timestamp': timestamp,
                'buy_price': price,
                'quantity': quantity
            }
        elif side == OrderSide.SELL:
            self.cash += cost
            self.current_position -= quantity
            if self.open_trade:
                # Pair the sell with the open buy
                trade_pair = {
                    'buy_trade_id': self.open_trade['buy_trade_id'],
                    'buy_timestamp': self.open_trade['buy_timestamp'],
                    'buy_price': self.open_trade['buy_price'],
                    'sell_trade_id': self.trade_count,
                    'sell_timestamp': timestamp,
                    'sell_price': price,
                    'quantity': quantity,
                    'sell_rsi': rsi
                }
                self.trade_pairs.append(trade_pair)
                self.open_trade = None
            else:
                # In case there's no open trade, record the sell standalone
                trade_pair = {
                    'buy_trade_id': None,
                    'buy_timestamp': None,
                    'buy_price': None,
                    'sell_trade_id': self.trade_count,
                    'sell_timestamp': timestamp,
                    'sell_price': price,
                    'quantity': quantity,
                    'sell_rsi': rsi
                }
                self.trade_pairs.append(trade_pair)

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
            'profit_factor': (self.portfolio_values[-1] if self.portfolio_values else 0) / 10000,
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
            adjustment='raw',
            feed='iex'  # NEW: Explicitly request regular trading hours data
        )
        bars = client.get_stock_bars(request).data[SYMBOL]
        self.backtest_data = [(bar.timestamp, bar.close) for bar in bars]
        logging.info(f"Loaded {len(self.backtest_data)} bars for backtesting")

    def _check_market_hours(self, timestamp: Optional[datetime] = None):
        if self.backtest_mode and timestamp:
            ny_time = timestamp.astimezone(pytz.timezone('America/New_York'))
        else:
            ny_time = pd.Timestamp.now(tz='America/New_York')

        # New: Strictly exclude weekends
        if ny_time.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False

        # New: Precise time boundary checks
        market_open = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= ny_time <= market_close

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
                # MODIFIED: Use proper cooldown check
                if time.time() - self.last_trade_time > COOLDOWN:
                    self.trading_allowed = self._check_market_hours()

                if not self.trading_allowed:
                    time.sleep(60)
                    continue

                position = self._get_position()
                current_price = self.buffer.latest_price
                ma = self.buffer.get_ma()

                if ma is None:
                    continue  # Not enough data for MA calculation

                uptrend = current_price > ma
                current_rsi = self.buffer.last_rsi
                previous_rsi = self.buffer.previous_rsi

                # MODIFIED: Full trend-based strategy logic
                if uptrend:
                    # Buy signal: RSI crosses above entry level from below
                    if previous_rsi < UP_TREND_RSI_ENTRY and current_rsi >= UP_TREND_RSI_ENTRY:
                        if position <= 0:
                            self._submit_order(OrderSide.BUY, TRADE_QTY)
                            self.last_trade_time = time.time()
                    # Exit signal: RSI crosses below centerline
                    elif position > 0 and previous_rsi >= RSI_EXIT_LEVEL and current_rsi < RSI_EXIT_LEVEL:
                        self._submit_order(OrderSide.SELL, position)
                        self.last_trade_time = time.time()
                else:
                    # Sell signal: RSI crosses below exit level from above
                    if previous_rsi > DOWN_TREND_RSI_EXIT and current_rsi <= DOWN_TREND_RSI_EXIT:
                        if position >= 0:  # FIXED: Only sell if flat or long
                            self._submit_order(OrderSide.SELL, TRADE_QTY)
                            self.last_trade_time = time.time()
                    # Cover signal: RSI crosses above centerline
                    elif position < 0 and previous_rsi <= RSI_EXIT_LEVEL and current_rsi > RSI_EXIT_LEVEL:
                        self._submit_order(OrderSide.BUY, abs(position))
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

            # Update buffer with historical price
            self.buffer.update(price)

            # Get indicators
            current_rsi = self.buffer.last_rsi
            previous_rsi = self.buffer.previous_rsi
            ma = self.buffer.get_ma()
            position = self.backtest_result.current_position

            # Print market data
            ma_display = f"{ma:.2f}" if ma else "N/A"
            print(f"{timestamp.astimezone(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M')} | "
                  f"Price: {price:.2f} | "
                  f"RSI: {current_rsi:.2f} | "
                  f"MA: {ma_display}", end='')

            # Only trade if we have valid MA
            if ma is not None:
                uptrend = price > ma

                if uptrend:
                    # Uptrend Buy Signal
                    if previous_rsi < UP_TREND_RSI_ENTRY and current_rsi >= UP_TREND_RSI_ENTRY:
                        if position <= 0:
                            self._submit_order(OrderSide.BUY, TRADE_QTY, timestamp, price)
                            print(" | BUY SIGNAL", end='')

                    # Uptrend Exit Signal
                    elif position > 0 and previous_rsi >= RSI_EXIT_LEVEL and current_rsi < RSI_EXIT_LEVEL:
                        # Pass current RSI when selling
                        self._submit_order(OrderSide.SELL, position, timestamp, price)
                        print(" | SELL EXIT", end='')

                else:
                    # Downtrend Sell Signal
                    if previous_rsi > DOWN_TREND_RSI_EXIT and current_rsi <= DOWN_TREND_RSI_EXIT:
                        if position >= 0:
                            self._submit_order(OrderSide.SELL, TRADE_QTY, timestamp, price)
                            print(" | SELL SIGNAL", end='')

                    # Downtrend Cover Signal
                    elif position < 0 and previous_rsi <= RSI_EXIT_LEVEL and current_rsi > RSI_EXIT_LEVEL:
                        self._submit_order(OrderSide.BUY, abs(position), timestamp, price)
                        print(" | BUY COVER", end='')

            # Update portfolio value
            current_value = self.backtest_result.portfolio_value(price)
            self.backtest_result.portfolio_values.append(current_value)
            print(f" | Portfolio: ${current_value:,.2f}")

        logging.info("Backtest complete")

        # Print detailed trade log
        print("\nTrade History:")
        trade_df = pd.DataFrame(self.backtest_result.trades)
        if not trade_df.empty:
            print(trade_df[['timestamp', 'side', 'price', 'quantity', 'portfolio_value']])

        # Print summary statistics
        print("\nBacktest Summary:")
        summary = self.backtest_result.summary()
        summary_df = pd.DataFrame(summary, index=[0])
        print(summary_df)

        # Additional performance metrics - MODIFIED FIX
        if len(self.backtest_result.trades) > 0:
            trade_df = pd.DataFrame(self.backtest_result.trades)
            trade_df['returns'] = trade_df['portfolio_value'].pct_change()

            winning_trades = len(trade_df[trade_df['returns'] > 0])
            total_trades = len(trade_df)

            print(f"\nAdditional Metrics:")
            print(f"Total Return: {(summary['final_value'] / 10000 - 1) * 100:.2f}%")

            if total_trades > 0:
                print(f"Win Rate: {winning_trades / total_trades * 100:.1f}%")
            else:
                print("Win Rate: No trades executed")

            # Rank the top 10 best and 10 worst trades by return
            trade_df = trade_df.dropna(subset=['returns'])
            top_trades = trade_df.nlargest(10, 'returns')
            bottom_trades = trade_df.nsmallest(10, 'returns')

            print("\nTop 10 Trades:")
            print(top_trades[['timestamp', 'side', 'price', 'quantity', 'portfolio_value', 'returns']])
            print("\nWorst 10 Trades:")
            print(bottom_trades[['timestamp', 'side', 'price', 'quantity', 'portfolio_value', 'returns']])
        else:
            print("\nNo trades were executed during the backtest period")

        # Print the paired trade summary showing buy price, sell price, and RSI at selling
        print("\nTrade Pairs (Buy/Sell) Summary:")
        if self.backtest_result.trade_pairs:
            pairs_df = pd.DataFrame(self.backtest_result.trade_pairs)
            print(pairs_df[['buy_timestamp', 'buy_price', 'sell_timestamp', 'sell_price', 'sell_rsi']])
            # Now rank the trade pairs by return
            valid_pairs = pairs_df.dropna(subset=['buy_price', 'sell_price']).copy()
            if not valid_pairs.empty:
                valid_pairs['return'] = (valid_pairs['sell_price'] - valid_pairs['buy_price']) / valid_pairs['buy_price']
                top_pairs = valid_pairs.nlargest(10, 'return')
                worst_pairs = valid_pairs.nsmallest(10, 'return')
                print("\nTop 10 Trade Pairs (Best):")
                print(top_pairs[['buy_timestamp', 'buy_price', 'sell_timestamp', 'sell_price', 'sell_rsi', 'return']])
                print("\nTop 10 Trade Pairs (Worst):")
                print(worst_pairs[['buy_timestamp', 'buy_price', 'sell_timestamp', 'sell_price', 'sell_rsi', 'return']])
        else:
            print("No trade pairs executed")

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
                # When selling in backtest mode, pass the current RSI value
                rsi = self.buffer.last_rsi if side == OrderSide.SELL else None
                self.backtest_result.add_trade(timestamp, side, price, qty, rsi)
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
                limit=MA_PERIOD * 2,  # MODIFIED: Get enough data for MA
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
        end_date = datetime.now()
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
