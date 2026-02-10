"""Position Sizer - Half-Kelly with drawdown dampening (Mojo-accelerated)."""

from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort


# ── Internal computation (pure Mojo, zero Python overhead) ──

fn _kelly_fraction(win_prob: Float64, risk_reward: Float64) -> Float64:
    var p = win_prob
    var q = 1.0 - p
    var b = risk_reward
    var raw = (p * b - q) / b
    if raw < 0.0:
        return 0.0
    return raw


fn _dd_dampening(current_dd: Float64, max_dd: Float64) -> Float64:
    if current_dd <= 0.0:
        return 1.0
    if current_dd >= max_dd:
        return 0.2
    return 1.0 - (current_dd / max_dd * 0.8)


# ── Python-exported functions (PythonObject in/out) ──
# Uses dict param for 7-arg function (max 6 PythonObject args per def_function).

fn _calc_position_size(params: PythonObject) raises -> PythonObject:
    """Takes dict with: ml_score, confidence, rr, dd, max_dd, min_pct, max_pct."""
    var ml_score = Float64(py=params["ml_score"])
    var confidence = Float64(py=params["confidence"])
    var rr = Float64(py=params["rr"])
    var dd = Float64(py=params["dd"])
    var max_dd = Float64(py=params["max_dd"])
    var min_pct = Float64(py=params["min_pct"])
    var max_pct = Float64(py=params["max_pct"])

    var kelly = _kelly_fraction(ml_score, rr)
    var half_kelly = kelly * 0.5
    var dd_factor = _dd_dampening(dd, max_dd)
    var raw_pct = half_kelly * dd_factor * confidence * 100.0

    if raw_pct < min_pct:
        return PythonObject(min_pct)
    if raw_pct > max_pct:
        return PythonObject(max_pct)
    return PythonObject(raw_pct)


fn _calc_quantity(
    py_capital: PythonObject,
    py_leverage: PythonObject,
    py_price: PythonObject,
) raises -> PythonObject:
    var capital = Float64(py=py_capital)
    var leverage = Float64(py=py_leverage)
    var price = Float64(py=py_price)
    if price <= 0.0:
        return PythonObject(0)
    var qty = Int(capital * leverage / price)
    return PythonObject(qty)


fn _calc_position_value(
    py_capital: PythonObject,
    py_leverage: PythonObject,
) raises -> PythonObject:
    var capital = Float64(py=py_capital)
    var leverage = Float64(py=py_leverage)
    return PythonObject(capital * leverage)


# ── Module initialization ──

@export
fn PyInit_position_sizer() -> PythonObject:
    try:
        var m = PythonModuleBuilder("position_sizer")
        m.def_function[_calc_position_size](
            "calculate_position_size_pct",
            docstring="Half-Kelly position sizing with drawdown dampening.",
        )
        m.def_function[_calc_quantity](
            "calculate_quantity",
            docstring="Calculate number of shares to buy.",
        )
        m.def_function[_calc_position_value](
            "calculate_position_value",
            docstring="Calculate total position value with leverage.",
        )
        return m.finalize()
    except e:
        abort(String("error creating position_sizer module:", e))
