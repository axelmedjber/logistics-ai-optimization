"""Inventory control formulas.

Classic, well-documented supply-chain formulas:

- Economic Order Quantity (EOQ) -- the order size that minimises the sum of
  ordering cost and holding cost.
- Safety stock -- the buffer that protects against demand variability during
  the replenishment lead time, sized for a target service level.
- Reorder point -- the stock level that triggers a new order.

The service-level z-score uses the standard normal inverse CDF from the
standard library (``statistics.NormalDist``), so there is no SciPy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


def economic_order_quantity(
    annual_demand: float, order_cost: float, holding_cost: float
) -> float:
    """Economic Order Quantity (Wilson formula).

    ``EOQ = sqrt(2 * D * S / H)``

    Parameters
    ----------
    annual_demand : float
        Total demand over one year (D), in units.
    order_cost : float
        Fixed cost of placing one order (S), in currency.
    holding_cost : float
        Cost of holding one unit in stock for one year (H), in currency.

    Returns
    -------
    float
        The order quantity that minimises total annual cost.
    """
    if annual_demand < 0:
        raise ValueError("annual_demand must be >= 0")
    if order_cost < 0:
        raise ValueError("order_cost must be >= 0")
    if holding_cost <= 0:
        raise ValueError("holding_cost must be > 0")
    return math.sqrt(2 * annual_demand * order_cost / holding_cost)


def safety_stock(
    demand_std_per_period: float, lead_time_periods: float, service_level: float = 0.95
) -> float:
    """Safety stock for a target service level.

    ``SS = z * sigma * sqrt(L)``

    where ``z`` is the standard-normal quantile of the service level, ``sigma``
    is the per-period standard deviation of demand, and ``L`` is the lead time
    expressed in the same periods.

    Parameters
    ----------
    demand_std_per_period : float
        Standard deviation of demand over one period (e.g. one day).
    lead_time_periods : float
        Replenishment lead time, in the same periods as the std deviation.
    service_level : float
        Probability of not running out of stock during the lead time
        (e.g. 0.95 for 95%). Must be in the open interval (0, 1).

    Returns
    -------
    float
        Recommended safety stock, in units.
    """
    if demand_std_per_period < 0:
        raise ValueError("demand_std_per_period must be >= 0")
    if lead_time_periods < 0:
        raise ValueError("lead_time_periods must be >= 0")
    if not 0.0 < service_level < 1.0:
        raise ValueError("service_level must be strictly between 0 and 1")

    z = NormalDist().inv_cdf(service_level)
    return z * demand_std_per_period * math.sqrt(lead_time_periods)


def reorder_point(
    avg_demand_per_period: float, lead_time_periods: float, safety_stock_units: float = 0.0
) -> float:
    """Reorder point: the stock level that should trigger a new order.

    ``ROP = average demand during lead time + safety stock``

    Parameters
    ----------
    avg_demand_per_period : float
        Average demand over one period.
    lead_time_periods : float
        Replenishment lead time, in the same periods.
    safety_stock_units : float
        Safety stock buffer (see :func:`safety_stock`).

    Returns
    -------
    float
        The reorder point, in units.
    """
    if avg_demand_per_period < 0:
        raise ValueError("avg_demand_per_period must be >= 0")
    if lead_time_periods < 0:
        raise ValueError("lead_time_periods must be >= 0")
    if safety_stock_units < 0:
        raise ValueError("safety_stock_units must be >= 0")
    return avg_demand_per_period * lead_time_periods + safety_stock_units


@dataclass
class InventoryPolicy:
    """A complete (Q, R) inventory policy for one product.

    Attributes
    ----------
    order_quantity : float
        How much to order each time (EOQ).
    reorder_point : float
        Stock level that triggers an order.
    safety_stock : float
        Buffer included in the reorder point.
    orders_per_year : float
        Expected number of orders per year (annual_demand / EOQ).
    """

    order_quantity: float
    reorder_point: float
    safety_stock: float
    orders_per_year: float

    @classmethod
    def build(
        cls,
        annual_demand: float,
        order_cost: float,
        holding_cost: float,
        avg_demand_per_period: float,
        demand_std_per_period: float,
        lead_time_periods: float,
        service_level: float = 0.95,
    ) -> "InventoryPolicy":
        """Compute a full replenishment policy from demand and cost inputs."""
        eoq = economic_order_quantity(annual_demand, order_cost, holding_cost)
        ss = safety_stock(demand_std_per_period, lead_time_periods, service_level)
        rop = reorder_point(avg_demand_per_period, lead_time_periods, ss)
        orders_per_year = annual_demand / eoq if eoq > 0 else 0.0
        return cls(
            order_quantity=eoq,
            reorder_point=rop,
            safety_stock=ss,
            orders_per_year=orders_per_year,
        )
