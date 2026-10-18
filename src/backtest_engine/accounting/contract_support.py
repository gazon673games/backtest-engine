"""Reject payout models that the execution ledger cannot settle correctly."""

from typing import Protocol

from backtest_engine.core.exceptions import DatasetValidationError


class ContractDescription(Protocol):
    symbol: str
    instrument_type: str
    contract_type: str


SUPPORTED_CONTRACTS = {"spot": {"spot", "linear"}, "future": {"linear"}, "perpetual": {"linear"}}


def validate_accounting_contract(instrument: ContractDescription) -> None:
    if instrument.contract_type not in SUPPORTED_CONTRACTS[instrument.instrument_type]:
        raise DatasetValidationError(
            f"Instrument {instrument.symbol}: unsupported {instrument.contract_type} "
            f"{instrument.instrument_type} payout model. Use a supported linear/spot "
            "specification; inverse/quanto require native payout, funding and margin models."
        )
