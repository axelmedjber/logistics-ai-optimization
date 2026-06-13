import math

import pytest

from src.inventory import (
    economic_order_quantity,
    safety_stock,
    reorder_point,
    InventoryPolicy,
)


def test_eoq_known_value():
    # D=1000, S=10, H=2  ->  EOQ = sqrt(2*1000*10/2) = sqrt(10000) = 100
    assert math.isclose(economic_order_quantity(1000, 10, 2), 100.0)


def test_eoq_requires_positive_holding_cost():
    with pytest.raises(ValueError):
        economic_order_quantity(1000, 10, 0)


def test_safety_stock_at_95_percent():
    # z(0.95) ~= 1.6449, sigma=10, lead time=4 -> SS = 1.6449 * 10 * 2 = 32.90
    ss = safety_stock(demand_std_per_period=10, lead_time_periods=4, service_level=0.95)
    assert math.isclose(ss, 32.897, abs_tol=0.01)


def test_safety_stock_grows_with_service_level():
    low = safety_stock(10, 4, service_level=0.90)
    high = safety_stock(10, 4, service_level=0.99)
    assert high > low


def test_safety_stock_rejects_invalid_service_level():
    with pytest.raises(ValueError):
        safety_stock(10, 4, service_level=1.0)


def test_reorder_point_combines_lead_time_and_buffer():
    # 20 units/day * 5 days + 30 safety = 130
    assert reorder_point(20, 5, safety_stock_units=30) == 130.0


def test_inventory_policy_build_is_consistent():
    policy = InventoryPolicy.build(
        annual_demand=3650,
        order_cost=25,
        holding_cost=3,
        avg_demand_per_period=10,
        demand_std_per_period=4,
        lead_time_periods=5,
        service_level=0.95,
    )
    assert policy.order_quantity > 0
    assert policy.safety_stock > 0
    # reorder point must include the safety stock buffer
    assert policy.reorder_point > policy.safety_stock
    # orders per year must be consistent with EOQ
    assert math.isclose(policy.orders_per_year, 3650 / policy.order_quantity)
