"""Convert native float quantities without changing actual off-step inventory."""

import math
from decimal import Decimal


def native_quantity(value, step):
    quantity, increment = Decimal(str(value)), Decimal(str(step))
    aligned = (quantity / increment).to_integral_value() * increment
    # Only remove machine representation noise. Fee dust and partial liquidations
    # remain exact inputs; this must never invent an execution or repair a quantity.
    if abs(quantity - aligned) <= Decimal(str(8 * math.ulp(float(value)))):
        return aligned
    return quantity
