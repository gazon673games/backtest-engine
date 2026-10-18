"""Hard post-execution and solvency invariants shared by backtest runners."""

from __future__ import annotations

from backtest_engine.core.exceptions import DatasetValidationError


def assert_post_fill_limits(
    *,
    equity_cash: float,
    gross_notional: float,
    margin_requirement: float | None,
    max_position_notional: float | None = None,
    max_leverage: float | None = None,
    max_margin_usage: float | None = None,
    label: str = "portfolio",
) -> None:
    if gross_notional <= 0.0 and not margin_requirement:
        return
    if equity_cash <= 0.0:
        raise DatasetValidationError(
            f"Post-fill risk breach for {label}: equity_cash={equity_cash} is not positive"
        )
    breaches = []
    if max_position_notional is not None and gross_notional > max_position_notional:
        breaches.append(
            f"notional {gross_notional} exceeds max_position_notional {max_position_notional}"
        )
    if max_leverage is not None and gross_notional > equity_cash * max_leverage:
        breaches.append(
            f"leverage {gross_notional / equity_cash} exceeds max_leverage {max_leverage}"
        )
    if margin_requirement is not None:
        if margin_requirement > equity_cash:
            breaches.append(f"initial margin {margin_requirement} exceeds equity {equity_cash}")
        if max_margin_usage is not None and margin_requirement > equity_cash * max_margin_usage:
            breaches.append(
                f"margin usage {margin_requirement / equity_cash} exceeds "
                f"max_margin_usage {max_margin_usage}"
            )
    if breaches:
        raise DatasetValidationError(f"Post-fill risk breach for {label}: " + "; ".join(breaches))


def assert_solvent(
    equity_cash: float,
    *,
    margin_requirement: float | None = None,
    label: str = "portfolio",
) -> None:
    if equity_cash <= 0.0:
        raise DatasetValidationError(
            f"Backtest insolvent after settlement for {label}: equity_cash={equity_cash}"
        )
    if margin_requirement is not None and margin_requirement > equity_cash:
        raise DatasetValidationError(
            f"Margin call after settlement for {label}: required_margin={margin_requirement}, "
            f"equity_cash={equity_cash}"
        )


def assert_end_position(position: float, policy: str, *, label: str) -> None:
    if policy == "reject" and abs(position) > 1e-12:
        raise DatasetValidationError(
            f"Open position remains at end of data for {label}: {position}. "
            "Provide a final flattening decision or set end_of_data_position_policy=mark "
            "for an explicit mark-to-market diagnostic run."
        )
