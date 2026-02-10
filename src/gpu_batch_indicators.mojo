"""
GPU Batch Technical Indicators - Production Binary

Reads price data for multiple stocks from binary file,
computes EMA(9,12,21,26,50), RSI(14), BB(20) on Apple Metal GPU,
writes results to binary output file.

Usage: mojo run gpu_batch_indicators.mojo <input.bin> <output.bin>

Input format:  [num_stocks: uint32] [prices: float32 × num_stocks × 100]
Output format: [float32 × num_stocks × 100 × 9]
  Arrays: ema9, ema12, ema21, ema26, ema50, rsi14, bb_upper, bb_mid, bb_lower
"""

from math import ceildiv, sqrt
from sys import has_accelerator, argv
from sys.ffi import external_call
from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from memory import UnsafePointer
from time import perf_counter_ns

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
# GPU Kernels
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
        result[base] = prices[base]
        for i in range(1, PRICE_LEN):
            result[base + i] = alpha * prices[base + i] + one_minus * result[base + i - 1]


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

        for ii in range(period):
            rsi[base + ii] = fifty

        comptime zero = ElemT(0)
        if al > zero:
            rsi[base + period] = hundred - (hundred / (one + ag / al))
        else:
            rsi[base + period] = hundred

        for ii in range(period + 1, PRICE_LEN):
            ag = (ag * pm1 + gains[base + ii]) * p_inv
            al = (al * pm1 + losses[base + ii]) * p_inv
            if al > zero:
                rsi[base + ii] = hundred - (hundred / (one + ag / al))
            else:
                rsi[base + ii] = hundred


fn bollinger_kernel(
    prices: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    upper: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    mid: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    lower: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    """Each thread computes BB(20) for one stock."""
    var stock_idx = block_idx.x * block_dim.x + thread_idx.x
    if stock_idx < MAX_STOCKS:
        var base = stock_idx * PRICE_LEN
        comptime period = 20
        comptime zero = ElemT(0)
        comptime two = ElemT(2)
        comptime p_inv = ElemT(1.0 / Float64(period))

        for i in range(PRICE_LEN):
            if i < period - 1:
                upper[base + i] = zero
                mid[base + i] = zero
                lower[base + i] = zero
            else:
                var sma = prices[base + i]
                for j in range(1, period):
                    sma = sma + prices[base + i - j]
                sma = sma * p_inv

                var var_acc = zero
                for j in range(period):
                    var d = prices[base + i - j] - sma
                    var_acc = var_acc + d * d
                var_acc = var_acc * p_inv

                mid[base + i] = sma
                upper[base + i] = sma + two * sqrt(var_acc)
                lower[base + i] = sma - two * sqrt(var_acc)


# ============================================================
# Main - File I/O + GPU Orchestration
# ============================================================

def main():
    var args = argv()

    if len(args) < 3:
        print("Usage: mojo run gpu_batch_indicators.mojo <input.bin> <output.bin>")
        print("Input:  [num_stocks:u32][float32 × num_stocks × 100]")
        print("Output: [float32 × num_stocks × 100 × 9] (ema9,12,21,26,50,rsi14,bb_u,m,l)")
        return

    var input_path = String(args[1])
    var output_path = String(args[2])

    @parameter
    if not has_accelerator():
        print("ERROR: No GPU found")
        return

    # ---- Read input binary file using C FFI ----
    var rb_mode = String("rb")
    var fp = external_call["fopen", UnsafePointer[NoneType]](
        input_path.unsafe_cstr_ptr(), rb_mode.unsafe_cstr_ptr()
    )
    # Read header (4 bytes = uint32 num_stocks)
    var header_list = List[UInt32](capacity=1)
    header_list.append(0)
    _ = external_call["fread", Int](header_list.unsafe_ptr().bitcast[NoneType](), 4, 1, fp)
    var num_stocks = Int(header_list[0])

    if num_stocks <= 0 or num_stocks > MAX_STOCKS:
        print("ERROR: num_stocks", num_stocks, "out of range (1-", MAX_STOCKS, ")")
        _ = external_call["fclose", Int](fp)
        return

    # Read price data into List buffer
    var data_count = num_stocks * PRICE_LEN
    var price_list = List[Float32](capacity=data_count)
    for _ in range(data_count):
        price_list.append(Float32(0.0))
    _ = external_call["fread", Int](price_list.unsafe_ptr().bitcast[NoneType](), 4, data_count, fp)
    _ = external_call["fclose", Int](fp)

    var t_start = perf_counter_ns()

    # Initialize GPU
    var ctx = DeviceContext()

    # Create host buffer and fill with price data
    var h_p = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    ctx.synchronize()

    # Zero out unused slots
    for i in range(TOTAL):
        h_p[i] = Float32(0.0)

    # Copy price data from List buffer to host buffer
    for i in range(data_count):
        h_p[i] = price_list[i]

    # Allocate GPU buffers
    var d_p = ctx.enqueue_create_buffer[float_dtype](TOTAL)

    # 5 EMA buffers
    var d_ema9 = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_ema12 = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_ema21 = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_ema26 = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_ema50 = ctx.enqueue_create_buffer[float_dtype](TOTAL)

    # RSI buffers
    var d_g = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_l = ctx.enqueue_create_buffer[float_dtype](TOTAL)
    var d_rsi = ctx.enqueue_create_buffer[float_dtype](TOTAL)

    # BB buffers
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

    # Launch all GPU kernels
    # EMA kernels (5 periods)
    ctx.enqueue_function[ema_kernel[9], ema_kernel[9]](tp, te9, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function[ema_kernel[12], ema_kernel[12]](tp, te12, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function[ema_kernel[21], ema_kernel[21]](tp, te21, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function[ema_kernel[26], ema_kernel[26]](tp, te26, grid_dim=stock_blocks, block_dim=block_size)
    ctx.enqueue_function[ema_kernel[50], ema_kernel[50]](tp, te50, grid_dim=stock_blocks, block_dim=block_size)

    # RSI kernels (2-pass)
    ctx.enqueue_function[rsi_deltas_kernel, rsi_deltas_kernel](tp, tg, tl, grid_dim=total_blocks, block_dim=block_size)
    ctx.enqueue_function[rsi_compute_kernel, rsi_compute_kernel](tg, tl, tr, grid_dim=stock_blocks, block_dim=block_size)

    # Bollinger Bands kernel
    ctx.enqueue_function[bollinger_kernel, bollinger_kernel](tp, tbu, tbm, tbl, grid_dim=stock_blocks, block_dim=block_size)

    ctx.synchronize()

    var gpu_us = (perf_counter_ns() - t_start) / 1000

    # Copy results back to host
    var h_ema9 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_ema12 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_ema21 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_ema26 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_ema50 = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_rsi = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_bu = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_bm = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
    var h_bl = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)

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

    # ---- Write output binary file using C FFI ----
    var out_size = num_stocks * PRICE_LEN
    var total_out = out_size * 9

    # Allocate output buffer using List
    var out_list = List[Float32](capacity=total_out)
    for _ in range(total_out):
        out_list.append(Float32(0.0))

    # Pack 9 indicator arrays sequentially
    for i in range(out_size):
        out_list[0 * out_size + i] = Float32(h_ema9[i])
        out_list[1 * out_size + i] = Float32(h_ema12[i])
        out_list[2 * out_size + i] = Float32(h_ema21[i])
        out_list[3 * out_size + i] = Float32(h_ema26[i])
        out_list[4 * out_size + i] = Float32(h_ema50[i])
        out_list[5 * out_size + i] = Float32(h_rsi[i])
        out_list[6 * out_size + i] = Float32(h_bu[i])
        out_list[7 * out_size + i] = Float32(h_bm[i])
        out_list[8 * out_size + i] = Float32(h_bl[i])

    # Write to file
    var wb_mode = String("wb")
    var ofp = external_call["fopen", UnsafePointer[NoneType]](
        output_path.unsafe_cstr_ptr(), wb_mode.unsafe_cstr_ptr()
    )
    _ = external_call["fwrite", Int](out_list.unsafe_ptr().bitcast[NoneType](), 4, total_out, ofp)
    _ = external_call["fclose", Int](ofp)

    var total_us = (perf_counter_ns() - t_start) / 1000
    print("GPU_BATCH_OK", num_stocks, gpu_us, total_us)
