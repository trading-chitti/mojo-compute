"""Risk Manager - Guard rails for sandbox trading (Mojo-accelerated)."""

from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort


# ── Internal computation (pure Mojo, zero Python overhead) ──

fn _can_open(
    dd: Float64, max_dd: Float64,
    open_pos: Int, max_pos: Int,
    sector_exp: Float64, max_sector: Float64,
    avail: Float64, min_trade: Float64,
    active: Bool,
) -> Bool:
    if not active:
        return False
    if dd >= max_dd:
        return False
    if open_pos >= max_pos:
        return False
    if sector_exp >= max_sector:
        return False
    if avail < min_trade:
        return False
    return True


# ── Python-exported functions (PythonObject in/out) ──
# can_open_trade has 9 params -> uses dict (max 6 PythonObject args per def_function).

fn _py_can_open_trade(params: PythonObject) raises -> PythonObject:
    """Takes dict: dd, max_dd, open_pos, max_pos, sector_exp, max_sector, avail, min_trade, active."""
    var dd = Float64(py=params["dd"])
    var max_dd = Float64(py=params["max_dd"])
    var open_pos = Int(py=params["open_pos"])
    var max_pos = Int(py=params["max_pos"])
    var sector_exp = Float64(py=params["sector_exp"])
    var max_sector = Float64(py=params["max_sector"])
    var avail = Float64(py=params["avail"])
    var min_trade = Float64(py=params["min_trade"])
    var active_int = Int(py=params["active"])
    var active = active_int != 0

    return PythonObject(_can_open(
        dd, max_dd, open_pos, max_pos,
        sector_exp, max_sector, avail, min_trade, active,
    ))


fn _py_should_stop(
    py_dd: PythonObject, py_max_dd: PythonObject,
) raises -> PythonObject:
    var dd = Float64(py=py_dd)
    var max_dd = Float64(py=py_max_dd)
    return PythonObject(dd >= max_dd)


fn _py_available_capital(
    py_current: PythonObject, py_allocated: PythonObject,
) raises -> PythonObject:
    var current = Float64(py=py_current)
    var allocated = Float64(py=py_allocated)
    var avail = current - allocated
    if avail < 0.0:
        return PythonObject(0.0)
    return PythonObject(avail)


fn _py_sector_exposure_pct(
    py_sector: PythonObject, py_total: PythonObject,
) raises -> PythonObject:
    var sector = Float64(py=py_sector)
    var total = Float64(py=py_total)
    if total <= 0.0:
        return PythonObject(0.0)
    return PythonObject(sector / total * 100.0)


# ── Module initialization ──

@export
fn PyInit_risk_manager() -> PythonObject:
    try:
        var m = PythonModuleBuilder("risk_manager")
        m.def_function[_py_can_open_trade](
            "can_open_trade",
            docstring="Check all risk conditions. Takes dict param.",
        )
        m.def_function[_py_should_stop](
            "should_stop_session",
            docstring="Check if session should auto-stop due to drawdown.",
        )
        m.def_function[_py_available_capital](
            "calculate_available_capital",
            docstring="Capital available for new trades.",
        )
        m.def_function[_py_sector_exposure_pct](
            "calculate_sector_exposure_pct",
            docstring="Sector exposure as percentage of total capital.",
        )
        return m.finalize()
    except e:
        abort(String("error creating risk_manager module:", e))
