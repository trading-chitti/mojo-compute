"""Performance Metrics - Fast trading metrics (Mojo-accelerated).

Exports scalar functions for real-time dashboard updates.
List-based functions (max_drawdown, sharpe_ratio) stay in Python
as they're not in the hot path.
"""

from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort


# ── Internal computation (pure Mojo, zero Python overhead) ──

fn _win_rate(winning: Int, total: Int) -> Float64:
    if total == 0:
        return 0.0
    return Float64(winning) / Float64(total) * 100.0


fn _profit_factor(gross_profit: Float64, gross_loss: Float64) -> Float64:
    if gross_loss == 0.0:
        if gross_profit > 0.0:
            return 999.99
        return 0.0
    return gross_profit / abs(gross_loss)


fn _current_dd(current: Float64, peak: Float64) -> Float64:
    if peak <= 0.0 or current >= peak:
        return 0.0
    return (peak - current) / peak * 100.0


fn _total_pnl_pct(current: Float64, initial: Float64) -> Float64:
    if initial <= 0.0:
        return 0.0
    return (current - initial) / initial * 100.0


# ── Python-exported functions (PythonObject in/out) ──

fn _py_win_rate(
    py_winning: PythonObject, py_total: PythonObject,
) raises -> PythonObject:
    var winning = Int(py=py_winning)
    var total = Int(py=py_total)
    return PythonObject(_win_rate(winning, total))


fn _py_profit_factor(
    py_profit: PythonObject, py_loss: PythonObject,
) raises -> PythonObject:
    var profit = Float64(py=py_profit)
    var loss = Float64(py=py_loss)
    return PythonObject(_profit_factor(profit, loss))


fn _py_current_drawdown(
    py_current: PythonObject, py_peak: PythonObject,
) raises -> PythonObject:
    var current = Float64(py=py_current)
    var peak = Float64(py=py_peak)
    return PythonObject(_current_dd(current, peak))


fn _py_total_pnl_pct(
    py_current: PythonObject, py_initial: PythonObject,
) raises -> PythonObject:
    var current = Float64(py=py_current)
    var initial = Float64(py=py_initial)
    return PythonObject(_total_pnl_pct(current, initial))


# ── Module initialization ──

@export
fn PyInit_performance_metrics() -> PythonObject:
    try:
        var m = PythonModuleBuilder("performance_metrics")
        m.def_function[_py_win_rate](
            "win_rate",
            docstring="Win rate as percentage.",
        )
        m.def_function[_py_profit_factor](
            "profit_factor",
            docstring="Profit factor = gross_profit / abs(gross_loss).",
        )
        m.def_function[_py_current_drawdown](
            "current_drawdown",
            docstring="Current drawdown percentage from peak.",
        )
        m.def_function[_py_total_pnl_pct](
            "total_pnl_pct",
            docstring="Total P&L percentage.",
        )
        return m.finalize()
    except e:
        abort(String("error creating performance_metrics module:", e))
