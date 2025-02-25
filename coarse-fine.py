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
DEFAULT_SYMBOL = 'MU'
TRADE_QTY = 100
RSI_PERIOD = 5
MA_PERIOD = 10
BUFFER_SIZE = max(RSI_PERIOD + 1, MA_PERIOD)
TIMEFRAME = TimeFrame.Minute
POLL_INTERVAL = 1
COOLDOWN = 60
MAX_WORKERS = 10

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
# GLOBAL API CLIENTS
# ======================
HISTORICAL_DATA_CLIENT = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ======================
# THREAD-SAFE HISTORICAL DATA CACHE AND PER-KEY LOCKS
# ======================
# Cache keyed by (symbol, start_str, end_str)
HISTORICAL_DATA_CACHE: Dict[Tuple[str, str, str], List[Tuple[datetime, float]]] = {}
# Global lock protecting the overall cache dictionary and a per-key-locks dictionary.
GLOBAL_CACHE_LOCK = threading.Lock()
CACHE_KEY_LOCKS: Dict[Tuple[str, str, str], threading.Lock] = {}

def normalize_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def get_cache_key(symbol: str, start: datetime, end: datetime) -> Tuple[str, str, str]:
    return (symbol, normalize_date(start), normalize_date(end))

def get_lock_for_key(key: Tuple[str, str, str]) -> threading.Lock:
    with GLOBAL_CACHE_LOCK:
        if key not in CACHE_KEY_LOCKS:
            CACHE_KEY_LOCKS[key] = threading.Lock()
        return CACHE_KEY_LOCKS[key]

def get_cached_historical_data(symbol: str, start: datetime, end: datetime) -> Optional[List[Tuple[datetime, float]]]:
    key = get_cache_key(symbol, start, end)
    # First, check the cache quickly.
    with GLOBAL_CACHE_LOCK:
        if key in HISTORICAL_DATA_CACHE:
            return HISTORICAL_DATA_CACHE[key]
    # Now acquire the per-key lock.
    key_lock = get_lock_for_key(key)
    with key_lock:
        # Check again—another thread might have fetched the data while we waited.
        with GLOBAL_CACHE_LOCK:
            if key in HISTORICAL_DATA_CACHE:
                return HISTORICAL_DATA_CACHE[key]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TIMEFRAME,
                start=start,
                end=end,
                adjustment='raw',
                feed='iex'
            )
            data_response = HISTORICAL_DATA_CLIENT.get_stock_bars(request).data
            if symbol not in data_response:
                logging.error(f"Historical data for symbol {symbol} not available.")
                return None
            bars = data_response[symbol]
            data = [(bar.timestamp, bar.close) for bar in bars]
            with GLOBAL_CACHE_LOCK:
                HISTORICAL_DATA_CACHE[key] = data
            logging.info(f"Cached historical data for {symbol} from {key[1]} to {key[2]}")
            return data
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return None

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
        prices_array = np.array(self.prices)
        deltas = np.diff(prices_array)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        if self.avg_gain is None or self.avg_loss is None:
            self.avg_gain = np.mean(gains[-RSI_PERIOD:])
            self.avg_loss = np.mean(losses[-RSI_PERIOD:])
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
# BACKTEST RESULTS CLASS (Optimized Storage)
# ======================
class BacktestResult:
    def __init__(self):
        self.trade_count = 0
        self.cash = 10000.0
        self.current_position = 0
        self.commission = 0.000023
        self.initial_value = 10000.0
        self.last_portfolio_value = 10000.0
        self.max_portfolio_value = 10000.0
        self.min_portfolio_value = 10000.0

    def add_trade(self, timestamp: datetime, side: str, price: float, quantity: int):
        self.trade_count += 1
        cost = quantity * price * (1 + self.commission)
        # Update position and cash
        if side == OrderSide.BUY:
            self.cash -= cost
            self.current_position += quantity
        else:
            self.cash += cost
            self.current_position -= quantity

        # Update portfolio value and track extremes without storing each trade.
        current_value = self.portfolio_value(price)
        self.last_portfolio_value = current_value
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        self.min_portfolio_value = min(self.min_portfolio_value, current_value)

    def portfolio_value(self, current_price: float) -> float:
        return self.cash + (self.current_position * current_price)

    def summary(self):
        final_value = self.last_portfolio_value
        total_profit = final_value - self.initial_value
        max_drawdown = (
            (self.max_portfolio_value - self.min_portfolio_value) / self.max_portfolio_value
            if self.max_portfolio_value != 0 else 0.0
        )
        profit_factor = final_value / self.initial_value
        return {
            "total_trades": self.trade_count,
            "final_value": final_value,
            "total_profit": total_profit,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
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
                 color: str = "",
                 symbol: str = DEFAULT_SYMBOL,
                 preloaded_data: Optional[List[Tuple[datetime, float]]] = None):
        self.backtest_mode = backtest_mode
        self.backtest_result = BacktestResult() if backtest_mode else None
        self.backtest_data = None
        self.up_trend_rsi_entry = up_trend_rsi_entry
        self.down_trend_rsi_exit = down_trend_rsi_exit
        self.rsi_exit_level = rsi_exit_level
        self.verbose = verbose
        self.color = color
        self.symbol = symbol

        if not backtest_mode:
            self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            self.data_stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        self.buffer = RSIBuffer()
        self.running = threading.Event()
        self.position_lock = threading.Lock()
        self.last_trade_time = 0
        self.trading_allowed = False

        if backtest_mode:
            if preloaded_data is not None:
                self.backtest_data = preloaded_data
            else:
                self._load_backtest_data(backtest_start, backtest_end)

    def cprint(self, text, end='\n'):
        if self.color:
            print(f"{self.color}{text}\033[0m", end=end)
        else:
            print(text, end=end)

    def _load_backtest_data(self, start: datetime, end: datetime):
        data = get_cached_historical_data(self.symbol, start, end)
        if data is None:
            msg = f"Historical data for symbol {self.symbol} not available."
            logging.error(msg)
            raise KeyError(msg)
        self.backtest_data = data

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
        status = f"Symbol: {self.symbol} | Price: {self.buffer.latest_price:.2f} | RSI: {self.buffer.last_rsi:.1f}"
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
            if ma is not None:
                uptrend = price > ma
                if uptrend:
                    if previous_rsi < self.up_trend_rsi_entry <= current_rsi:
                        if position <= 0:
                            self._submit_order(OrderSide.BUY, TRADE_QTY, timestamp, price)
                            if self.verbose:
                                self.cprint(" | BUY SIGNAL", end='')
                    elif position > 0 and previous_rsi >= self.rsi_exit_level > current_rsi:
                        self._submit_order(OrderSide.SELL, position, timestamp, price)
                        if self.verbose:
                            self.cprint(" | SELL EXIT", end='')
                else:
                    if previous_rsi > self.down_trend_rsi_exit >= current_rsi:
                        if position >= 0:
                            self._submit_order(OrderSide.SELL, TRADE_QTY, timestamp, price)
                            if self.verbose:
                                self.cprint(" | SELL SIGNAL", end='')
                    elif position < 0 and previous_rsi <= self.rsi_exit_level < current_rsi:
                        self._submit_order(OrderSide.BUY, abs(position), timestamp, price)
                        if self.verbose:
                            self.cprint(" | BUY COVER", end='')
            current_value = self.backtest_result.portfolio_value(price)
            if self.verbose:
                self.cprint(f" | Portfolio: ${current_value:,.2f}")
        if self.verbose:
            self.cprint("\nBacktest Summary:")
            summary = self.backtest_result.summary()
            summary_df = pd.DataFrame(summary, index=[0])
            self.cprint(summary_df.to_string())

    def _get_position(self):
        if self.backtest_mode:
            return self.backtest_result.current_position
        try:
            position = self.trading_client.get_open_position(self.symbol)
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
                symbol=self.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            self.trading_client.submit_order(order)
            logging.info(f"{side} {qty} {self.symbol}")
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
        self.data_stream.subscribe_trades(self.handle_trade, self.symbol)
        threading.Thread(target=asyncio.run, args=(self.data_stream.run(),), daemon=True).start()
        threading.Thread(target=self.execute_strategy, daemon=True).start()

    def _preload_historical(self):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TIMEFRAME,
                limit=MA_PERIOD * 2,
                adjustment='raw'
            )
            data_response = HISTORICAL_DATA_CLIENT.get_stock_bars(request).data
            if self.symbol not in data_response:
                msg = (f"Historical preload data for symbol {self.symbol} not available. "
                       f"Returned keys: {list(data_response.keys())}.")
                logging.error(msg)
                raise KeyError(msg)
            bars = data_response[self.symbol]
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
    over a specified date range and across multiple symbols for each combination in the provided grid.
    The performance (using a weighted profit metric that accounts for drawdown) is recorded and
    the best parameter combination (averaged across symbols) is returned.
    """
    def __init__(self, backtest_start: datetime, backtest_end: datetime,
                 parameter_grid: Dict[str, List[int]], symbols: List[str]):
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.parameter_grid = parameter_grid
        self.symbols = symbols
        self.results = []

    # Process a single symbol.
    def run_backtest_for_symbol(self, symbol: str, up_trend_rsi_entry: int,
                                down_trend_rsi_exit: int, rsi_exit_level: int,
                                color: str) -> Dict:
        historical_data = get_cached_historical_data(symbol, self.backtest_start, self.backtest_end)
        if historical_data is None:
            return {}
        bot = RSITradingBot(
            backtest_mode=True,
            backtest_start=self.backtest_start,
            backtest_end=self.backtest_end,
            up_trend_rsi_entry=up_trend_rsi_entry,
            down_trend_rsi_exit=down_trend_rsi_exit,
            rsi_exit_level=rsi_exit_level,
            verbose=False,
            color=color,
            symbol=symbol,
            preloaded_data=historical_data
        )
        bot.start()
        return bot.backtest_result.summary()

    # Run backtests for all symbols.
    def run_backtest_multi(self, up_trend_rsi_entry: int, down_trend_rsi_exit: int,
                           rsi_exit_level: int, color: str) -> Dict:
        summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(self.run_backtest_for_symbol, symbol,
                                up_trend_rsi_entry, down_trend_rsi_exit, rsi_exit_level, color)
                for symbol in self.symbols
            ]
            for future in concurrent.futures.as_completed(futures):
                summary = future.result()
                if summary:
                    summaries.append(summary)
        aggregated_summary = {
            'total_trades': sum(s['total_trades'] for s in summaries),
            'final_value': np.mean([s['final_value'] for s in summaries]),
            'max_drawdown': np.mean([s['max_drawdown'] for s in summaries]),
            'profit_factor': np.mean([s['profit_factor'] for s in summaries])
        }
        return aggregated_summary

    def _compute_weighted_profit(self, summary: Dict, drawdown_weight: float = 1.0) -> float:
        profit = summary.get('final_value', 0) - 10000
        max_drawdown = summary.get('max_drawdown', 0)
        return profit * (1 - drawdown_weight * max_drawdown)

    def optimize(self, drawdown_weight: float = 1.0) -> Tuple[Dict[str, int], List[Dict]]:
        best_params = None
        best_weighted_profit = -float('inf')
        tasks = []
        colors = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
        comb_index = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for entry in self.parameter_grid.get("up_trend_rsi_entry", [40]):
                for exit_down in self.parameter_grid.get("down_trend_rsi_exit", [60]):
                    for exit_rsi in self.parameter_grid.get("rsi_exit_level", [50]):
                        color = colors[comb_index % len(colors)]
                        comb_index += 1
                        future = executor.submit(self.run_backtest_multi, entry, exit_down, exit_rsi, color)
                        tasks.append((future, {
                            "up_trend_rsi_entry": entry,
                            "down_trend_rsi_exit": exit_down,
                            "rsi_exit_level": exit_rsi
                        }))
            for future, params in tasks:
                summary = future.result()
                profit = summary.get('final_value', 0) - 10000
                weighted_profit = self._compute_weighted_profit(summary, drawdown_weight)
                self.results.append({
                    "up_trend_rsi_entry": params["up_trend_rsi_entry"],
                    "down_trend_rsi_exit": params["down_trend_rsi_exit"],
                    "rsi_exit_level": params["rsi_exit_level"],
                    "summary": summary,
                    "profit": profit,
                    "weighted_profit": weighted_profit
                })
                if weighted_profit > best_weighted_profit:
                    best_weighted_profit = weighted_profit
                    best_params = {
                        "up_trend_rsi_entry": params["up_trend_rsi_entry"],
                        "down_trend_rsi_exit": params["down_trend_rsi_exit"],
                        "rsi_exit_level": params["rsi_exit_level"]
                    }
        return best_params, self.results

    def optimize_coarse_fine(self, coarse_steps: Optional[Dict[str, int]] = None,
                               fine_delta: Optional[Dict[str, int]] = None,
                               drawdown_weight: float = 1.0) -> Tuple[Dict[str, int], Dict[str, List[Dict]]]:
        if coarse_steps is None:
            coarse_steps = {"up_trend_rsi_entry": 5, "down_trend_rsi_exit": 5, "rsi_exit_level": 3}
        if fine_delta is None:
            fine_delta = {"up_trend_rsi_entry": 4, "down_trend_rsi_exit": 4, "rsi_exit_level": 2}
        coarse_grid = {}
        for param, values in self.parameter_grid.items():
            lower = min(values)
            upper = max(values)
            step = coarse_steps.get(param, 1)
            coarse_grid[param] = list(range(lower, upper + 1, step))
        coarse_results = []
        tasks = []
        comb_index = 0
        colors = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for entry in coarse_grid["up_trend_rsi_entry"]:
                for exit_down in coarse_grid["down_trend_rsi_exit"]:
                    for exit_rsi in coarse_grid["rsi_exit_level"]:
                        color = colors[comb_index % len(colors)]
                        comb_index += 1
                        future = executor.submit(self.run_backtest_multi, entry, exit_down, exit_rsi, color)
                        tasks.append((future, {
                            "up_trend_rsi_entry": entry,
                            "down_trend_rsi_exit": exit_down,
                            "rsi_exit_level": exit_rsi
                        }))
            for future, params in tasks:
                summary = future.result()
                profit = summary.get("final_value", 0) - 10000
                weighted_profit = self._compute_weighted_profit(summary, drawdown_weight)
                coarse_results.append({
                    "up_trend_rsi_entry": params["up_trend_rsi_entry"],
                    "down_trend_rsi_exit": params["down_trend_rsi_exit"],
                    "rsi_exit_level": params["rsi_exit_level"],
                    "summary": summary,
                    "profit": profit,
                    "weighted_profit": weighted_profit
                })
        best_coarse = max(coarse_results, key=lambda x: x["weighted_profit"])
        fine_grid = {}
        for param in self.parameter_grid:
            lower_bound = min(self.parameter_grid[param])
            upper_bound = max(self.parameter_grid[param])
            best_val = best_coarse[param]
            delta = fine_delta.get(param, 1)
            fine_lower = max(lower_bound, best_val - delta)
            fine_upper = min(upper_bound, best_val + delta)
            fine_grid[param] = list(range(fine_lower, fine_upper + 1))
        fine_results = []
        tasks = []
        comb_index = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for entry in fine_grid["up_trend_rsi_entry"]:
                for exit_down in fine_grid["down_trend_rsi_exit"]:
                    for exit_rsi in fine_grid["rsi_exit_level"]:
                        color = colors[comb_index % len(colors)]
                        comb_index += 1
                        future = executor.submit(self.run_backtest_multi, entry, exit_down, exit_rsi, color)
                        tasks.append((future, {
                            "up_trend_rsi_entry": entry,
                            "down_trend_rsi_exit": exit_down,
                            "rsi_exit_level": exit_rsi
                        }))
            for future, params in tasks:
                summary = future.result()
                profit = summary.get("final_value", 0) - 10000
                weighted_profit = self._compute_weighted_profit(summary, drawdown_weight)
                fine_results.append({
                    "up_trend_rsi_entry": params["up_trend_rsi_entry"],
                    "down_trend_rsi_exit": params["down_trend_rsi_exit"],
                    "rsi_exit_level": params["rsi_exit_level"],
                    "summary": summary,
                    "profit": profit,
                    "weighted_profit": weighted_profit
                })
        best_fine = max(fine_results, key=lambda x: x["weighted_profit"])
        all_results = {"coarse": coarse_results, "fine": fine_results}
        return best_fine, all_results

# ======================
# MAIN EXECUTION
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description="RSI Trading Bot")
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--optimize', action='store_true', help='Optimize parameters using coarse-fine search')
    parser.add_argument('--start', type=str, help='Backtest start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, help='Backtest end date in YYYY-MM-DD format')
    parser.add_argument('--drawdown_weight', type=float, default=1.0, help='Weighting factor for drawdown in the optimization score')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, help='Stock symbol for backtest or live trading')
    parser.add_argument('--symbols', type=str, default="MU,AAPL,GOOGL", help='Comma-separated list of symbols for optimization')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.backtest:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
        bot = RSITradingBot(backtest_mode=True, backtest_start=start_date, backtest_end=end_date, verbose=True, symbol=args.symbol)
        bot.start()
    elif args.optimize:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
        parameter_grid = {
            "up_trend_rsi_entry": list(range(1, 40)),
            "down_trend_rsi_exit": list(range(55, 90)),
            "rsi_exit_level": list(range(35, 66))
        }
        symbols = [sym.strip() for sym in args.symbols.split(",")]
        optimizer = ParameterOptimizer(start_date, end_date, parameter_grid, symbols)
        best_params, results = optimizer.optimize_coarse_fine(drawdown_weight=args.drawdown_weight)
        print("Optimization Results (Coarse-Fine):")
        print(f"Best Parameters: {best_params}")
        print("\nDetailed Results:")
        print("Coarse Search Results:")
        for result in results["coarse"]:
            print(result)
        print("\nFine Search Results:")
        for result in results["fine"]:
            print(result)
    else:
        bot = RSITradingBot(backtest_mode=False, verbose=True, symbol=args.symbol)
        bot.start()
