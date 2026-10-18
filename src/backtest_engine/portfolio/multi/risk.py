"""Public compatibility facade for split implementation modules."""

from backtest_engine.portfolio.multi._risk_math import (
    apply_position_notional_limit as apply_position_notional_limit,
)
from backtest_engine.portfolio.multi._risk_math import close_at as close_at
from backtest_engine.portfolio.multi._risk_math import contracts_scale as contracts_scale
from backtest_engine.portfolio.multi._risk_math import gross_notional as gross_notional
from backtest_engine.portfolio.multi._risk_math import (
    instrument_cash_in_base as instrument_cash_in_base,
)
from backtest_engine.portfolio.multi._risk_math import (
    maybe_clip_asset_position as maybe_clip_asset_position,
)
from backtest_engine.portfolio.multi._risk_math import (
    portfolio_margin_requirement as portfolio_margin_requirement,
)
from backtest_engine.portfolio.multi._risk_policy import (
    apply_portfolio_level_limits as apply_portfolio_level_limits,
)
from backtest_engine.portfolio.multi._risk_policy import (
    apply_portfolio_risk as apply_portfolio_risk,
)
from backtest_engine.portfolio.multi._risk_policy import (
    limit_scales_for_portfolio as limit_scales_for_portfolio,
)
from backtest_engine.portfolio.multi._risk_policy import target_contracts as target_contracts
