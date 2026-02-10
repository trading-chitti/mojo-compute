"""P&L Calculator - Leveraged trade profit/loss (Mojo-accelerated)."""

from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort


# ── Internal computation (pure Mojo, zero Python overhead) ──

fn _pnl(is_call: Bool, price_a: Float64, price_b: Float64, qty: Float64) -> Float64:
    """CALL: qty*(b-a), PUT: qty*(a-b)."""
    if is_call:
        return qty * (price_b - price_a)
    return qty * (price_a - price_b)


fn _pnl_pct(is_call: Bool, entry: Float64, other: Float64) -> Float64:
    if entry <= 0.0:
        return 0.0
    if is_call:
        return (other - entry) / entry * 100.0
    return (entry - other) / entry * 100.0


# ── Python-exported functions (PythonObject in/out) ──

fn _unrealized_pnl(
    py_st: PythonObject, py_entry: PythonObject,
    py_current: PythonObject, py_qty: PythonObject,
) raises -> PythonObject:
    var is_call = String(py=py_st) == "CALL"
    var entry = Float64(py=py_entry)
    var current = Float64(py=py_current)
    var qty = Float64(py=py_qty)
    return PythonObject(_pnl(is_call, entry, current, qty))


fn _unrealized_pnl_pct(
    py_st: PythonObject, py_entry: PythonObject,
    py_current: PythonObject,
) raises -> PythonObject:
    var is_call = String(py=py_st) == "CALL"
    var entry = Float64(py=py_entry)
    var current = Float64(py=py_current)
    return PythonObject(_pnl_pct(is_call, entry, current))


fn _realized_pnl(
    py_st: PythonObject, py_entry: PythonObject,
    py_exit: PythonObject, py_qty: PythonObject,
) raises -> PythonObject:
    var is_call = String(py=py_st) == "CALL"
    var entry = Float64(py=py_entry)
    var exit_p = Float64(py=py_exit)
    var qty = Float64(py=py_qty)
    return PythonObject(_pnl(is_call, entry, exit_p, qty))


fn _realized_pnl_pct(
    py_st: PythonObject, py_entry: PythonObject,
    py_exit: PythonObject,
) raises -> PythonObject:
    var is_call = String(py=py_st) == "CALL"
    var entry = Float64(py=py_entry)
    var exit_p = Float64(py=py_exit)
    return PythonObject(_pnl_pct(is_call, entry, exit_p))


fn _check_target(
    py_st: PythonObject, py_current: PythonObject,
    py_target: PythonObject,
) raises -> PythonObject:
    var is_call = String(py=py_st) == "CALL"
    var current = Float64(py=py_current)
    var target = Float64(py=py_target)
    if is_call:
        return PythonObject(current >= target)
    return PythonObject(current <= target)


fn _check_stoploss(
    py_st: PythonObject, py_current: PythonObject,
    py_sl: PythonObject,
) raises -> PythonObject:
    var is_call = String(py=py_st) == "CALL"
    var current = Float64(py=py_current)
    var sl = Float64(py=py_sl)
    if is_call:
        return PythonObject(current <= sl)
    return PythonObject(current >= sl)


fn _holding_minutes(
    py_entry_ts: PythonObject, py_exit_ts: PythonObject,
) raises -> PythonObject:
    var entry_ts = Float64(py=py_entry_ts)
    var exit_ts = Float64(py=py_exit_ts)
    var minutes = Int((exit_ts - entry_ts) / 60.0)
    return PythonObject(minutes)


# ── Module initialization ──

@export
fn PyInit_pnl_calculator() -> PythonObject:
    try:
        var m = PythonModuleBuilder("pnl_calculator")
        m.def_function[_unrealized_pnl](
            "calculate_unrealized_pnl",
            docstring="Unrealized P&L for open position.",
        )
        m.def_function[_unrealized_pnl_pct](
            "calculate_unrealized_pnl_pct",
            docstring="Unrealized P&L percentage.",
        )
        m.def_function[_realized_pnl](
            "calculate_realized_pnl",
            docstring="Realized P&L on trade close.",
        )
        m.def_function[_realized_pnl_pct](
            "calculate_realized_pnl_pct",
            docstring="Realized P&L percentage.",
        )
        m.def_function[_check_target](
            "check_target_hit",
            docstring="Check if target price hit.",
        )
        m.def_function[_check_stoploss](
            "check_stoploss_hit",
            docstring="Check if stop loss hit.",
        )
        m.def_function[_holding_minutes](
            "calculate_holding_duration_minutes",
            docstring="Holding duration in minutes.",
        )
        return m.finalize()
    except e:
        abort(String("error creating pnl_calculator module:", e))
