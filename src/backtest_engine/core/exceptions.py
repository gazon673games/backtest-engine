class BacktestError(Exception):
    """Base exception for expected backtest engine failures."""


class DatasetValidationError(BacktestError):
    """Raised when a dataset cannot be used by the MVP engine."""


class QualityGateError(BacktestError):
    """Raised when the quality gate does not approve a dataset."""


class StrategyPluginError(BacktestError):
    """Raised when a Python strategy plugin cannot be loaded or validated."""


class ReportingError(BacktestError):
    """Raised when an experiment report cannot be generated."""
