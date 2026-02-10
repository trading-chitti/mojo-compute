"""
GPU Batch Technical Indicators - Python Callable Version
NO C FFI - works around compiler crash

Computes EMA(9,12,21,26,50), RSI(14), BB(20) on Apple Metal GPU
Called from Python via MAX framework
"""

from math import ceildiv, sqrt
from sys import has_accelerator
from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from memory import UnsafePointer
from time import perf_counter_ns
from python import Python

# Fixed allocation for up to 2500 stocks × 100 prices
comptime float_dtype = DType.float32
comptime MAX_STOCKS = 2500
comptime PRICE_LEN = 100
comptime TOTAL = MAX_STOCKS * PRICE_LEN
comptime layout = Layout.row_major(TOTAL)
comptime block_size = 256
comptime stock_blocks = ceildiv(MAX_STOCKS, block_size)
comptime total_blocks = ceildiv(TOTAL, block_size)

comptime ElemT = LayoutTensor[float_dtype, layout, MutAnyOrigin].element_type


# ============================================================
# GPU Kernels (same as original)
# ============================================================

fn ema_kernel[period: Int](
    prices: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    result: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    """Each thread computes EMA for one stock. Period is compile-time."""
    var stock_idx = block_idx.x * block_dim.x + thread_idx.x
    if stock_idx < MAX_STOCKS:
        var base = stock_idx * PRICE_LEN
        comptime alpha = ElemT(2.0 / (Float64(period) + 1.0))
        comptime one_minus = ElemT(1.0 - 2.0 / (Float64(period) + 1.0))

        var ema = prices[base]
        result[base] = ema
        for i in range(1, PRICE_LEN):
            ema = alpha * prices[base + i] + one_minus * ema
            result[base + i] = ema


fn rsi_deltas_kernel(
    prices: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    gains: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    losses: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    """Calculate price gains and losses in parallel."""
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < TOTAL:
        var i = tid % PRICE_LEN
        if i == 0:
            gains[tid] = ElemT(0)
            losses[tid] = ElemT(0)
        else:
            var delta = prices[tid] - prices[tid - 1]
            comptime zero = ElemT(0)
            if delta > zero:
                gains[tid] = delta
                losses[tid] = zero
            else:
                gains[tid] = zero
                losses[tid] = zero - delta


fn rsi_compute_kernel(
    gains: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    losses: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rsi: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    """Each thread computes RSI(14) smoothing for one stock."""
    var stock_idx = block_idx.x * block_dim.x + thread_idx.x
    if stock_idx < MAX_STOCKS:
        var base = stock_idx * PRICE_LEN
        comptime period = 14
        comptime fifty = ElemT(50)
        comptime hundred = ElemT(100)
        comptime one = ElemT(1)
        comptime pm1 = ElemT(period - 1)
        comptime p_inv = ElemT(1.0 / Float64(period))

        var ag = gains[base + 1]
        for ii in range(2, period + 1):
            ag = ag + gains[base + ii]
        ag = ag * p_inv

        var al = losses[base + 1]
        for ii in range(2, period + 1):
            al = al + losses[base + ii]
        al = al * p_inv

        rsi[base + period] = fifty
        for i in range(period + 1, PRICE_LEN):
            ag = (ag * pm1 + gains[base + i]) * p_inv
            al = (al * pm1 + losses[base + i]) * p_inv
            if al < ElemT(1e-9):
                rsi[base + i] = hundred
            else:
                var rs = ag / al
                rsi[base + i] = hundred - (hundred / (one + rs))


fn bollinger_kernel(
    prices: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    bb_upper: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    bb_mid: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    bb_lower: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    """Each thread computes BB(20) for one stock."""
    var stock_idx = block_idx.x * block_dim.x + thread_idx.x
    if stock_idx < MAX_STOCKS:
        var base = stock_idx * PRICE_LEN
        comptime period = 20
        comptime p_inv = ElemT(1.0 / Float64(period))
        comptime two = ElemT(2)

        for i in range(period, PRICE_LEN):
            var sum = ElemT(0)
            for j in range(period):
                sum = sum + prices[base + i - period + 1 + j]
            var sma = sum * p_inv

            var variance = ElemT(0)
            for j in range(period):
                var diff = prices[base + i - period + 1 + j] - sma
                variance = variance + diff * diff
            var std = sqrt(variance * p_inv)

            bb_mid[base + i] = sma
            bb_upper[base + i] = sma + two * std
            bb_lower[base + i] = sma - two * std


# ============================================================
# Python-Callable Interface
# ============================================================

def compute_batch_indicators_gpu(prices_flat: List[Float32], num_stocks: Int) -> List[Float32]:
    """
    Compute batch indicators on GPU.

    Args:
        prices_flat: Flattened price array (num_stocks * 100 prices)
        num_stocks: Number of stocks

    Returns:
        Flattened result array (num_stocks * 100 * 9 indicators)
    """
    @parameter
    if not has_accelerator():
        print("ERROR: No GPU accelerator available")
        var empty = List[Float32]()
        return empty^

    if num_stocks <= 0 or num_stocks > MAX_STOCKS:
        print("ERROR: num_stocks out of range")
        var empty = List[Float32]()
        return empty^

    var t_start = perf_counter_ns()

    # Initialize device context
    var ctx = DeviceContext()

    # Allocate host buffer and copy input data
    var h_p = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    ctx.synchronize()

    for i in range(TOTAL):
        h_p[i] = Float32(0.0)

    var data_count = num_stocks * PRICE_LEN
    for i in range(data_count):
        h_p[i] = prices_flat[i]

    # Allocate GPU buffers
    var d_p = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_ema9 = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_ema12 = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_ema21 = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_ema26 = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_ema50 = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_g = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_l = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_rsi = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_bu = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_bm = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_bl = ctx.enqueue_create_buffer[float_dtype](TOTAL)

    # Copy prices to GPU
    ctx.enqueue_copy(d_p, h_p)
    ctx.synchronize()

    # Create tensor views
    var tp = LayoutTensor[float_dtype, layout](d_p)
    var te9 = LayoutTensor[float_dtype, layout](d_ema9)
    var te12 = LayoutTensor[float_dtype, layout](d_ema12)
    var te21 = LayoutTensor[float_dtype, layout](d_ema21)
    var te26 = LayoutTensor[float_dtype, layout](d_ema26)
    var te50 = LayoutTensor[float_dtype, layout](d_ema50)
    var tg = LayoutTensor[float_dtype, layout](d_g)
    var tl = LayoutTensor[float_dtype, layout](d_l)
    var tr = LayoutTensor[float_dtype, layout](d_rsi)
    var tbu = LayoutTensor[float_dtype, layout](d_bu)
    var tbm = LayoutTensor[float_dtype, layout](d_bm)
    var tbl = LayoutTensor[float_dtype, layout](d_bl)

    var t_gpu_start = perf_counter_ns()

    # Launch all GPU kernels (v25.7 uses enqueue_function_checked)
    ctx.enqueue_function_checked[ema_kernel[9], ema_kernel[9]](tp, te9, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function_checked[ema_kernel[12], ema_kernel[12]](tp, te12, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function_checked[ema_kernel[21], ema_kernel[21]](tp, te21, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function_checked[ema_kernel[26], ema_kernel[26]](tp, te26, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function_checked[ema_kernel[50], ema_kernel[50]](tp, te50, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function_checked[rsi_deltas_kernel, rsi_deltas_kernel](tp, tg, tl, grid_dim=total_blocks, block_dim=block_size)
    ctx.enqueue_function_checked[rsi_compute_kernel, rsi_compute_kernel](tg, tl, tr, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function_checked[bollinger_kernel, bollinger_kernel](tp, tbu, tbm, tbl, grid_dim=stock_blocks, block_dim=block_size)

    ctx.synchronize()
    var gpu_us = (perf_counter_ns() - t_gpu_start) / 1000

    # Allocate output host buffers
    var h_ema9 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_ema12 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_ema21 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_ema26 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_ema50 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_rsi = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_bu = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_bm = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_bl = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    ctx.synchronize()

    # Copy results back
    ctx.enqueue_copy(h_ema9, d_ema9)
    ctx.enqueue_copy(h_ema12, d_ema12)
    ctx.enqueue_copy(h_ema21, d_ema21)
    ctx.enqueue_copy(h_ema26, d_ema26)
    ctx.enqueue_copy(h_ema50, d_ema50)
    ctx.enqueue_copy(h_rsi, d_rsi)
    ctx.enqueue_copy(h_bu, d_bu)
    ctx.enqueue_copy(h_bm, d_bm)
    ctx.enqueue_copy(h_bl, d_bl)
    ctx.synchronize()

    # Pack results into output list
    var out_size = num_stocks * PRICE_LEN
    var total_out = out_size * 9
    var out_list = List[Float32](capacity=total_out)

    for i in range(out_size):
        out_list.append(Float32(h_ema9[i]))
    for i in range(out_size):
        out_list.append(Float32(h_ema12[i]))
    for i in range(out_size):
        out_list.append(Float32(h_ema21[i]))
    for i in range(out_size):
        out_list.append(Float32(h_ema26[i]))
    for i in range(out_size):
        out_list.append(Float32(h_ema50[i]))
    for i in range(out_size):
        out_list.append(Float32(h_rsi[i]))
    for i in range(out_size):
        out_list.append(Float32(h_bu[i]))
    for i in range(out_size):
        out_list.append(Float32(h_bm[i]))
    for i in range(out_size):
        out_list.append(Float32(h_bl[i]))

    var total_us = (perf_counter_ns() - t_start) / 1000
    print("GPU_BATCH_OK", num_stocks, gpu_us, total_us)

    return out_list^


def main():
    """Test function."""
    print("Mojo GPU Batch Indicators - Python Callable")
    print("✅ Compilation successful!")
    print("Import from Python: from mojo_gpu import compute_batch_indicators_gpu")
