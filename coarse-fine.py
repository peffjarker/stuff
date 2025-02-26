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
ALPACA_API_KEY = 'PKYRJ7EDWUB6SLBS9XV5'
ALPACA_SECRET_KEY = 'uBuv3cZRY3jsEsNkiK6dbmJHqbN83brmLbtMlip2'
DEFAULT_SYMBOL = 'SPY'
TRADE_QTY = 100
DEFAULT_RSI_PERIOD = 7
DEFAULT_MA_PERIOD = 10
TIMEFRAME = TimeFrame.Minute
POLL_INTERVAL = 1
COOLDOWN = 10
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
HISTORICAL_DATA_CACHE: Dict[Tuple[str, str, str], List[Tuple[datetime, float]]] = {}
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
    with GLOBAL_CACHE_LOCK:
        if key in HISTORICAL_DATA_CACHE:
            return HISTORICAL_DATA_CACHE[key]
    key_lock = get_lock_for_key(key)
    with key_lock:
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
                feed='sip'
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
    def __init__(self, rsi_period: int = DEFAULT_RSI_PERIOD, ma_period: int = DEFAULT_MA_PERIOD):
        self.rsi_period = rsi_period
        self.ma_period = ma_period
        self.buffer_size = max(rsi_period + 1, ma_period)
        self.prices = []
        self.lock = threading.Lock()
        self.avg_gain = None
        self.avg_loss = None
        self.last_rsi = 50.0
        self.previous_rsi = 50.0
        self.latest_price = None

    def add_historical_bars(self, bars):
        with self.lock:
            self.prices = [bar.close for bar in bars[-self.buffer_size:]]
            if len(self.prices) >= self.rsi_period + 1:
                self._calculate_rsi()

    def update(self, price):
        with self.lock:
            self.latest_price = price
            if len(self.prices) < self.buffer_size:
                self.prices.append(price)
            else:
                self.prices = self.prices[1:] + [price]
            if len(self.prices) >= self.rsi_period + 1:
                self._calculate_rsi()

    def _calculate_rsi(self):
        self.previous_rsi = self.last_rsi
        if len(self.prices) < self.buffer_size:
            return
        prices_array = np.array(self.prices)
        deltas = np.diff(prices_array)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        if self.avg_gain is None or self.avg_loss is None:
            self.avg_gain = np.mean(gains[-self.rsi_period:])
            self.avg_loss = np.mean(losses[-self.rsi_period:])
        else:
            self.avg_gain = (self.avg_gain * (self.rsi_period - 1) + gains[-1]) / self.rsi_period
            self.avg_loss = (self.avg_loss * (self.rsi_period - 1) + losses[-1]) / self.rsi_period
        if self.avg_loss == 0:
            self.last_rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.last_rsi = 100.0 - (100.0 / (1 + rs))

    def get_ma(self):
        if len(self.prices) >= self.ma_period:
            return sum(self.prices[-self.ma_period:]) / self.ma_period
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
        if side == OrderSide.BUY:
            self.cash -= cost
            self.current_position += quantity
        else:
            self.cash += cost
            self.current_position -= quantity
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
                 up_trend_rsi_entry: int = 31,
                 down_trend_rsi_exit: int = 89,
                 rsi_exit_level: int = 40,
                 verbose: bool = True,
                 color: str = "",
                 symbol: str = DEFAULT_SYMBOL,
                 rsi_period: int = DEFAULT_RSI_PERIOD,
                 ma_period: int = DEFAULT_MA_PERIOD,
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
        self.rsi_period = rsi_period
        self.ma_period = ma_period

        if not backtest_mode:
            self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            self.data_stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        self.buffer = RSIBuffer(rsi_period=self.rsi_period, ma_period=self.ma_period)
        self.running = threading.Event()
        self.position_lock = threading.Lock()
        self.last_trade_time = 0
        self.trading_allowed = True

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
                limit=self.ma_period * 2,
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
    over a specified date range. In this modified version, we process one stock
    at a time. For each stock, all candidate parameter combinations are evaluated
    concurrently before moving on to the next stock.
    """
    def __init__(self, backtest_start: datetime, backtest_end: datetime,
                 parameter_grid: Dict[str, List[int]], symbols: List[str]):
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.parameter_grid = parameter_grid
        self.symbols = symbols
        self.results = {}

    def run_backtest_for_symbol(self, symbol: str, up_trend_rsi_entry: int,
                                  down_trend_rsi_exit: int, rsi_exit_level: int,
                                  rsi_period: int, ma_period: int, color: str) -> Dict:
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
            rsi_period=rsi_period,
            ma_period=ma_period,
            preloaded_data=historical_data
        )
        bot.start()
        return bot.backtest_result.summary()

    def optimize_by_stock(self, drawdown_weight: float = 1.0) -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[Dict]]]:
        """
        For each stock, concurrently evaluate every parameter combination.
        Return a mapping of each symbol to its best parameter combination and detailed results.
        """
        best_params_by_stock = {}
        results_by_stock = {}
        colors = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]

        for symbol in self.symbols:
            tasks = []
            comb_index = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for entry in self.parameter_grid.get("up_trend_rsi_entry", [40]):
                    for exit_down in self.parameter_grid.get("down_trend_rsi_exit", [60]):
                        for exit_rsi in self.parameter_grid.get("rsi_exit_level", [50]):
                            for rsi_period in self.parameter_grid.get("rsi_period", [DEFAULT_RSI_PERIOD]):
                                for ma_period in self.parameter_grid.get("ma_period", [DEFAULT_MA_PERIOD]):
                                    color = colors[comb_index % len(colors)]
                                    comb_index += 1
                                    future = executor.submit(
                                        self.run_backtest_for_symbol,
                                        symbol,
                                        entry, exit_down, exit_rsi,
                                        rsi_period, ma_period, color
                                    )
                                    # print(future.result().get('final_value', 0))
                                    tasks.append((future, {
                                        "up_trend_rsi_entry": entry,
                                        "down_trend_rsi_exit": exit_down,
                                        "rsi_exit_level": exit_rsi,
                                        "rsi_period": rsi_period,
                                        "ma_period": ma_period
                                    }))
            best_weighted_profit = -float('inf')
            best_params = None
            stock_results = []
            for future, params in tasks:
                summary = future.result()
                profit = summary.get('final_value', 0) - 10000
                weighted_profit = profit  # You can adjust this with additional weighting if needed.
                stock_results.append({
                    **params,
                    "summary": summary,
                    "profit": profit,
                    "weighted_profit": weighted_profit
                })
                if weighted_profit > best_weighted_profit:
                    best_weighted_profit = weighted_profit
                    best_params = params
            best_params_by_stock[symbol] = best_params
            results_by_stock[symbol] = stock_results
        return best_params_by_stock, results_by_stock

# ======================
# MAIN EXECUTION
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description="RSI Trading Bot")
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--optimize', action='store_true', help='Optimize parameters using stock-by-stock search')
    parser.add_argument('--start', type=str, help='Backtest start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, help='Backtest end date in YYYY-MM-DD format')
    parser.add_argument('--drawdown_weight', type=float, default=1.0, help='Weighting factor for drawdown in the optimization score')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, help='Stock symbol for backtest or live trading')
    parser.add_argument('--symbols', type=str, default="SPY", help='Comma-separated list of symbols for optimization')
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
        # Parameter grid includes candidate values for rsi_period and ma_period.
        parameter_grid = {
            "up_trend_rsi_entry": list(range(1, 40)),
            "down_trend_rsi_exit": list(range(55, 90)),
            "rsi_exit_level": list(range(35, 66)),
            "rsi_period": list(range(2, 14)),   # example range from 2 to 13
            "ma_period": list(range(5, 50))       # example range from 5 to 49
        }
        symbols = [sym.strip() for sym in args.symbols.split(",")]
        optimizer = ParameterOptimizer(start_date, end_date, parameter_grid, symbols)
        best_params_by_stock, results_by_stock = optimizer.optimize_by_stock(drawdown_weight=args.drawdown_weight)
        print("Optimization Results (Stock-by-Stock):")
        for symbol in symbols:
            print(f"\nSymbol: {symbol}")
            print(f"Best Parameters: {best_params_by_stock[symbol]}")
            print("Detailed Results:")
            for result in results_by_stock[symbol]:
                print(result)
    else:
        bot = RSITradingBot(backtest_mode=False, verbose=True, symbol=args.symbol)
        bot.start()
