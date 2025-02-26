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
import argparse
from typing import Optional, Tuple, List, Dict
import concurrent.futures

# ======================
# CONFIGURATION
# ======================
ALPACA_API_KEY = 'PK8DPGKPJLXDYXHZYR7S'
ALPACA_SECRET_KEY = 'o5UibXlKY3VTn5XP6JTQNye2HNbonThpzd6tuUdp'
SYMBOL = 'MU'
TRADE_QTY = 100
RSI_PERIOD = 5
MA_PERIOD = 10
BUFFER_SIZE = max(RSI_PERIOD + 1, MA_PERIOD)
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
        self.previous_rsi = 50.0
        self.latest_price = None

    def add_historical_bars(self, bars):
        with self.lock:
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
        if len(self.prices) >= MA_PERIOD:
            return sum(self.prices[-MA_PERIOD:]) / MA_PERIOD
        return None

# ======================
# BACKTEST RESULTS CLASS
# ======================
class BacktestResult:
    def __init__(self):
        self.trades = []
        self.portfolio_values = []
        self.current_position = 0
        self.cash = 10000
        self.commission = 0.01
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
        if not df.empty:
            df['returns'] = df['portfolio_value'].pct_change()
        final_value = self.portfolio_values[-1] if self.portfolio_values else 10000
        max_value = df['portfolio_value'].max() if not df.empty else 10000
        min_value = df['portfolio_value'].min() if not df.empty else 10000
        return {
            'total_trades': len(self.trades),
            'final_value': final_value,
            'max_drawdown': (max_value - min_value) / max_value if max_value != 0 else 0,
            'profit_factor': final_value / 10000,
        }

# ======================
# TRADING BOT CORE
# ======================
class RSITradingBot:
    def __init__(self, backtest_mode: bool = False,
                 backtest_start: Optional[datetime] = None,
                 backtest_end: Optional[datetime] = None,
                 up_trend_rsi_entry: int = 40,
                 down_trend_rsi_exit: int = 60,
                 rsi_exit_level: int = 50,
                 verbose: bool = True,
                 color: str = ""):
        self.backtest_mode = backtest_mode
        self.backtest_result = BacktestResult() if backtest_mode else None
        self.backtest_data = None
        self.backtest_index = 0
        self.up_trend_rsi_entry = up_trend_rsi_entry
        self.down_trend_rsi_exit = down_trend_rsi_exit
        self.rsi_exit_level = rsi_exit_level
        self.verbose = verbose
        self.color = color  # ANSI color code (e.g., "\033[91m" for red)

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

    def cprint(self, text, end='\n'):
        """Helper method to print colored output if color is set."""
        if self.color:
            print(f"{self.color}{text}\033[0m", end=end)
        else:
            print(text, end=end)

    def _load_backtest_data(self, start: datetime, end: datetime):
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        request = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TIMEFRAME,
            start=start,
            end=end,
            adjustment='raw',
            feed='iex'
        )
        bars = client.get_stock_bars(request).data[SYMBOL]
        self.backtest_data = [(bar.timestamp, bar.close) for bar in bars]
        # logging.info(f"Loaded {len(self.backtest_data)} bars for backtesting")

    def _check_market_hours(self, timestamp: Optional[datetime] = None):
        if self.backtest_mode and timestamp:
            ny_time = timestamp.astimezone(pytz.timezone('America/New_York'))
        else:
            ny_time = pd.Timestamp.now(tz='America/New_York')

        if ny_time.weekday() >= 5:
            return False

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
        # In live mode, just print normally.
        if self.backtest_mode:
            self.cprint(status, end='\r')
        else:
            print(status.ljust(80), end='\r')

    def execute_strategy(self):
        if self.backtest_mode:
            self._run_backtest()
            return

        while self.running.is_set():
            try:
                if time.time() - self.last_trade_time > COOLDOWN:
                    self.trading_allowed = self._check_market_hours()

                if not self.trading_allowed:
                    time.sleep(60)
                    continue

                position = self._get_position()
                current_price = self.buffer.latest_price
                ma = self.buffer.get_ma()

                if ma is None:
                    continue

                uptrend = current_price > ma
                current_rsi = self.buffer.last_rsi
                previous_rsi = self.buffer.previous_rsi

                if uptrend:
                    if previous_rsi < self.up_trend_rsi_entry and current_rsi >= self.up_trend_rsi_entry:
                        if position <= 0:
                            self._submit_order(OrderSide.BUY, TRADE_QTY)
                            self.last_trade_time = time.time()
                    elif position > 0 and previous_rsi >= self.rsi_exit_level and current_rsi < self.rsi_exit_level:
                        self._submit_order(OrderSide.SELL, position)
                        self.last_trade_time = time.time()
                else:
                    if previous_rsi > self.down_trend_rsi_exit and current_rsi <= self.down_trend_rsi_exit:
                        if position >= 0:
                            self._submit_order(OrderSide.SELL, TRADE_QTY)
                            self.last_trade_time = time.time()
                    elif position < 0 and previous_rsi <= self.rsi_exit_level and current_rsi > self.rsi_exit_level:
                        self._submit_order(OrderSide.BUY, abs(position))
                        self.last_trade_time = time.time()

                time.sleep(POLL_INTERVAL)

            except Exception as e:
                logging.error(f"Strategy error: {e}")

    def _run_backtest(self):
        if self.verbose:
            self.cprint("\nBacktest Progress:")
        for idx, (timestamp, price) in enumerate(self.backtest_data, 1):
            if not self._check_market_hours(timestamp):
                continue

            self.buffer.update(price)
            current_rsi = self.buffer.last_rsi
            previous_rsi = self.buffer.previous_rsi
            ma = self.buffer.get_ma()
            position = self.backtest_result.current_position

            if self.verbose:
                ma_display = f"{ma:.2f}" if ma else "N/A"
                # Print progress with color
                # self.cprint(f"{timestamp.astimezone(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M')} | "
                            # f"Price: {price:.2f} | RSI: {current_rsi:.2f} | MA: {ma_display}", end='')

            if ma is not None:
                uptrend = price > ma

                if uptrend:
                    if previous_rsi < self.up_trend_rsi_entry and current_rsi >= self.up_trend_rsi_entry:
                        if position <= 0:
                            self._submit_order(OrderSide.BUY, TRADE_QTY, timestamp, price)
                            if self.verbose:
                                self.cprint(" | BUY SIGNAL", end='')
                    elif position > 0 and previous_rsi >= self.rsi_exit_level and current_rsi < self.rsi_exit_level:
                        self._submit_order(OrderSide.SELL, position, timestamp, price)
                        if self.verbose:
                            self.cprint(" | SELL EXIT", end='')
                else:
                    if previous_rsi > self.down_trend_rsi_exit and current_rsi <= self.down_trend_rsi_exit:
                        if position >= 0:
                            self._submit_order(OrderSide.SELL, TRADE_QTY, timestamp, price)
                            if self.verbose:
                                self.cprint(" | SELL SIGNAL", end='')
                    elif position < 0 and previous_rsi <= self.rsi_exit_level and current_rsi > self.rsi_exit_level:
                        self._submit_order(OrderSide.BUY, abs(position), timestamp, price)
                        if self.verbose:
                            self.cprint(" | BUY COVER", end='')

            current_value = self.backtest_result.portfolio_value(price)
            self.backtest_result.portfolio_values.append(current_value)
            if self.verbose:
                self.cprint(f" | Portfolio: ${current_value:,.2f}")

        # logging.info("Backtest complete")
        if self.verbose:
            self.cprint("\nTrade History:")
            trade_df = pd.DataFrame(self.backtest_result.trades)
            if not trade_df.empty:
                self.cprint(trade_df[['timestamp', 'side', 'price', 'quantity', 'portfolio_value']].to_string())
            self.cprint("\nBacktest Summary:")
            summary = self.backtest_result.summary()
            summary_df = pd.DataFrame(summary, index=[0])
            self.cprint(summary_df.to_string())

            if len(self.backtest_result.trades) > 0:
                trade_df = pd.DataFrame(self.backtest_result.trades)
                trade_df['returns'] = trade_df['portfolio_value'].pct_change()
                winning_trades = len(trade_df[trade_df['returns'] > 0])
                total_trades = len(trade_df)
                self.cprint(f"\nAdditional Metrics:")
                self.cprint(f"Total Return: {(summary['final_value'] / 10000 - 1) * 100:.2f}%")
                if total_trades > 0:
                    self.cprint(f"Win Rate: {winning_trades / total_trades * 100:.1f}%")
                else:
                    self.cprint("Win Rate: No trades executed")
            else:
                self.cprint("\nNo trades were executed during the backtest period")

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
                limit=MA_PERIOD * 2,
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
# PARAMETER OPTIMIZER CLASS
# ======================
class ParameterOptimizer:
    """
    This class optimizes trading strategy parameters by running a backtest
    over a specified date range for each combination in the provided grid.
    The performance (using final portfolio value as a metric) is recorded
    and the best parameter combination is returned.
    """
    def __init__(self, backtest_start: datetime, backtest_end: datetime, parameter_grid: Dict[str, List[int]]):
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.parameter_grid = parameter_grid
        self.results = []

    def run_backtest(self, up_trend_rsi_entry: int, down_trend_rsi_exit: int, rsi_exit_level: int, color: str) -> Dict:
        bot = RSITradingBot(
            backtest_mode=True,
            backtest_start=self.backtest_start,
            backtest_end=self.backtest_end,
            up_trend_rsi_entry=up_trend_rsi_entry,
            down_trend_rsi_exit=down_trend_rsi_exit,
            rsi_exit_level=rsi_exit_level,
            verbose=False,
            color=color
        )
        bot.start()
        return bot.backtest_result.summary()

    def optimize(self) -> Tuple[Dict[str, int], List[Dict]]:
        best_params = None
        best_profit = -float('inf')
        tasks = []
        colors = [
            "\033[91m",  # red
            "\033[92m",  # green
            "\033[93m",  # yellow
            "\033[94m",  # blue
            "\033[95m",  # magenta
            "\033[96m"   # cyan
        ]
        comb_index = 0
        # Use a thread pool executor to run backtests concurrently.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for entry in self.parameter_grid.get("up_trend_rsi_entry", [40]):
                for exit_down in self.parameter_grid.get("down_trend_rsi_exit", [60]):
                    for exit_rsi in self.parameter_grid.get("rsi_exit_level", [50]):
                        color = colors[comb_index % len(colors)]
                        comb_index += 1
                        future = executor.submit(self.run_backtest, entry, exit_down, exit_rsi, color)
                        tasks.append((future, {
                            "up_trend_rsi_entry": entry,
                            "down_trend_rsi_exit": exit_down,
                            "rsi_exit_level": exit_rsi
                        }))
            for future, params in tasks:
                summary = future.result()  # waits for task to complete
                profit = summary.get('final_value', 0) - 10000  # profit relative to initial capital
                self.results.append({
                    "up_trend_rsi_entry": params["up_trend_rsi_entry"],
                    "down_trend_rsi_exit": params["down_trend_rsi_exit"],
                    "rsi_exit_level": params["rsi_exit_level"],
                    "summary": summary,
                    "profit": profit
                })
                if profit > best_profit:
                    best_profit = profit
                    best_params = {
                        "up_trend_rsi_entry": params["up_trend_rsi_entry"],
                        "down_trend_rsi_exit": params["down_trend_rsi_exit"],
                        "rsi_exit_level": params["rsi_exit_level"]
                    }
        return best_params, self.results

# ======================
# MAIN EXECUTION
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description="RSI Trading Bot")
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--optimize', action='store_true', help='Optimize parameters')
    parser.add_argument('--start', type=str, help='Backtest start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, help='Backtest end date in YYYY-MM-DD format')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.backtest:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
        bot = RSITradingBot(backtest_mode=True, backtest_start=start_date, backtest_end=end_date, verbose=True)
        bot.start()
    elif args.optimize:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
        # A robust parameter grid:
        parameter_grid = {
            "up_trend_rsi_entry": list(range(1, 51)),    # 1 to 50 inclusive
            "down_trend_rsi_exit": list(range(55, 99)),    # 55 to 98 inclusive
            "rsi_exit_level": list(range(40, 61))          # 40 to 60 inclusive
        }
        optimizer = ParameterOptimizer(start_date, end_date, parameter_grid)
        best_params, results = optimizer.optimize()
        print("Optimization Results:")
        print(f"Best Parameters: {best_params}")
        for result in results:
            print(result)
    else:
        # Default: Run live trading bot.
        bot = RSITradingBot(backtest_mode=False, verbose=True)
        bot.start()
