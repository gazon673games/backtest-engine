"""Paper-trading engine using the backtest execution and accounting primitives."""

from backtest_engine.paper.engine import PaperTradingEngine, run_paper_feed
from backtest_engine.paper.feed import ParquetPaperFeed
from backtest_engine.paper.feed import __getattr__ as __getattr__
from backtest_engine.paper.models import PaperTradingConfig, PaperTradingResult

__all__ = [
    "BinanceClosedKlineFeed",
    "PaperTradingConfig",
    "PaperTradingEngine",
    "PaperTradingResult",
    "ParquetPaperFeed",
    "run_paper_feed",
]
