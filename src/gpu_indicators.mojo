"""
GPU-Accelerated Technical Indicators for Trading

Processes EMA, RSI, Bollinger Bands for 1000+ stocks on Apple Metal GPU.
Each GPU thread processes one stock independently.
"""

from math import ceildiv, sqrt
from sys import has_accelerator
from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from time import perf_counter_ns

comptime float_dtype = DType.float32
comptime NUM_STOCKS = 1000
comptime PRICE_LEN = 100
comptime TOTAL = NUM_STOCKS * PRICE_LEN
comptime layout = Layout.row_major(TOTAL)
comptime block_size = 256
comptime stock_blocks = ceildiv(NUM_STOCKS, block_size)
comptime total_blocks = ceildiv(TOTAL, block_size)

# Type alias for our tensor element
alias ElemT = LayoutTensor[float_dtype, layout, MutAnyOrigin].element_type


fn ema_kernel(
    prices: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    result: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    """Each thread computes EMA(14) for one stock."""
    var stock_idx = block_idx.x * block_dim.x + thread_idx.x
    if stock_idx < NUM_STOCKS:
        var base = stock_idx * PRICE_LEN
        comptime alpha = ElemT(2.0 / 15.0)
        comptime one_minus = ElemT(1.0 - 2.0 / 15.0)
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
    if stock_idx < NUM_STOCKS:
        var base = stock_idx * PRICE_LEN
        comptime period = 14
        comptime fifty = ElemT(50)
        comptime hundred = ElemT(100)
        comptime one = ElemT(1)
        comptime pm1 = ElemT(period - 1)
        comptime p_inv = ElemT(1.0 / Float64(period))

        # Initial averages
        var ag = gains[base + 1]
        for ii in range(2, period + 1):
            ag = ag + gains[base + ii]
        ag = ag * p_inv

        var al = losses[base + 1]
        for ii in range(2, period + 1):
            al = al + losses[base + ii]
        al = al * p_inv

        # Fill initial values
        for ii in range(period):
            rsi[base + ii] = fifty

        # RSI at period
        comptime zero = ElemT(0)
        if al > zero:
            rsi[base + period] = hundred - (hundred / (one + ag / al))
        else:
            rsi[base + period] = hundred

        # Wilder's smoothing
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
    if stock_idx < NUM_STOCKS:
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
                # SMA
                var sma = prices[base + i]
                for j in range(1, period):
                    sma = sma + prices[base + i - j]
                sma = sma * p_inv

                # StdDev
                var var_acc = zero
                for j in range(period):
                    var d = prices[base + i - j] - sma
                    var_acc = var_acc + d * d
                var_acc = var_acc * p_inv

                # BB = SMA +/- 2*std
                mid[base + i] = sma
                upper[base + i] = sma + two * sqrt(var_acc)
                lower[base + i] = sma - two * sqrt(var_acc)


def main():
    @parameter
    if not has_accelerator():
        print("No GPU found")
    else:
        var ctx = DeviceContext()
        print("=" * 60)
        print("GPU Technical Indicators - Apple M1 Pro Metal")
        print("Device:", ctx.name(), "| API:", ctx.api())
        print("=" * 60)
        print("Stocks:", NUM_STOCKS, "| Prices:", PRICE_LEN, "| Total:", TOTAL)
        print()

        # Host data
        var h_p = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
        ctx.synchronize()
        for s in range(NUM_STOCKS):
            var bp = Float32(100 + s % 50) * 10.0
            for i in range(PRICE_LEN):
                var n = Float32((s * 7 + i * 13) % 100 - 50) * 0.01
                h_p[s * PRICE_LEN + i] = bp * (1.0 + n)

        # GPU buffers
        var d_p = ctx.enqueue_create_buffer[float_dtype](TOTAL)
        var d_ema = ctx.enqueue_create_buffer[float_dtype](TOTAL)
        var d_g = ctx.enqueue_create_buffer[float_dtype](TOTAL)
        var d_l = ctx.enqueue_create_buffer[float_dtype](TOTAL)
        var d_rsi = ctx.enqueue_create_buffer[float_dtype](TOTAL)
        var d_bu = ctx.enqueue_create_buffer[float_dtype](TOTAL)
        var d_bm = ctx.enqueue_create_buffer[float_dtype](TOTAL)
        var d_bl = ctx.enqueue_create_buffer[float_dtype](TOTAL)

        ctx.enqueue_copy(d_p, h_p)
        ctx.synchronize()

        var tp = LayoutTensor[float_dtype, layout](d_p)
        var te = LayoutTensor[float_dtype, layout](d_ema)
        var tg = LayoutTensor[float_dtype, layout](d_g)
        var tl = LayoutTensor[float_dtype, layout](d_l)
        var tr = LayoutTensor[float_dtype, layout](d_rsi)
        var tbu = LayoutTensor[float_dtype, layout](d_bu)
        var tbm = LayoutTensor[float_dtype, layout](d_bm)
        var tbl = LayoutTensor[float_dtype, layout](d_bl)

        # Warmup
        ctx.enqueue_function[ema_kernel, ema_kernel](tp, te, grid_dim=stock_blocks, block_dim=block_size)
        ctx.synchronize()

        # ===== EMA Benchmark =====
        var t0 = perf_counter_ns()
        for _ in range(100):
            ctx.enqueue_function[ema_kernel, ema_kernel](tp, te, grid_dim=stock_blocks, block_dim=block_size)
        ctx.synchronize()
        var ema_us = (perf_counter_ns() - t0) / 100 / 1000
        print("EMA(14) x 1000 stocks:    ", ema_us, "us")

        # ===== RSI Benchmark =====
        ctx.enqueue_function[rsi_deltas_kernel, rsi_deltas_kernel](tp, tg, tl, grid_dim=total_blocks, block_dim=block_size)
        ctx.enqueue_function[rsi_compute_kernel, rsi_compute_kernel](tg, tl, tr, grid_dim=stock_blocks, block_dim=block_size)
        ctx.synchronize()

        t0 = perf_counter_ns()
        for _ in range(100):
            ctx.enqueue_function[rsi_deltas_kernel, rsi_deltas_kernel](tp, tg, tl, grid_dim=total_blocks, block_dim=block_size)
            ctx.enqueue_function[rsi_compute_kernel, rsi_compute_kernel](tg, tl, tr, grid_dim=stock_blocks, block_dim=block_size)
        ctx.synchronize()
        var rsi_us = (perf_counter_ns() - t0) / 100 / 1000
        print("RSI(14) x 1000 stocks:    ", rsi_us, "us")

        # ===== Bollinger Bands =====
        ctx.enqueue_function[bollinger_kernel, bollinger_kernel](tp, tbu, tbm, tbl, grid_dim=stock_blocks, block_dim=block_size)
        ctx.synchronize()

        t0 = perf_counter_ns()
        for _ in range(100):
            ctx.enqueue_function[bollinger_kernel, bollinger_kernel](tp, tbu, tbm, tbl, grid_dim=stock_blocks, block_dim=block_size)
        ctx.synchronize()
        var bb_us = (perf_counter_ns() - t0) / 100 / 1000
        print("BB(20)  x 1000 stocks:    ", bb_us, "us")

        # ===== Full Pipeline =====
        t0 = perf_counter_ns()
        for _ in range(100):
            ctx.enqueue_function[ema_kernel, ema_kernel](tp, te, grid_dim=stock_blocks, block_dim=block_size)
            ctx.enqueue_function[rsi_deltas_kernel, rsi_deltas_kernel](tp, tg, tl, grid_dim=total_blocks, block_dim=block_size)
            ctx.enqueue_function[rsi_compute_kernel, rsi_compute_kernel](tg, tl, tr, grid_dim=stock_blocks, block_dim=block_size)
            ctx.enqueue_function[bollinger_kernel, bollinger_kernel](tp, tbu, tbm, tbl, grid_dim=stock_blocks, block_dim=block_size)
        ctx.synchronize()
        var full_us = (perf_counter_ns() - t0) / 100 / 1000
        var full_ms = Float64(full_us) / 1000.0
        print()
        print("FULL PIPELINE (EMA+RSI+BB) x 1000 stocks:", full_us, "us (", full_ms, "ms)")

        # Verify
        var h_r = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
        var h_e = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
        var h_m = ctx.enqueue_create_host_buffer[float_dtype](TOTAL)
        ctx.enqueue_copy(h_r, d_rsi)
        ctx.enqueue_copy(h_e, d_ema)
        ctx.enqueue_copy(h_m, d_bm)
        ctx.synchronize()

        print()
        print("=== Verification (Stock 0) ===")
        print("Price[0..4]:", h_p[0], h_p[1], h_p[2], h_p[3], h_p[4])
        print("EMA[0..4]:  ", h_e[0], h_e[1], h_e[2], h_e[3], h_e[4])
        print("RSI[14..18]:", h_r[14], h_r[15], h_r[16], h_r[17], h_r[18])
        print("BB_Mid[20..24]:", h_m[20], h_m[21], h_m[22], h_m[23], h_m[24])
        print()
        var sps = Float64(1000) / (Float64(full_us) / 1_000_000.0)
        print("Throughput:", sps, "stocks/second")
