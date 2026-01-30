#!/usr/bin/env python3
"""
Mojo Compute Socket Server with Prometheus Metrics

Enhanced version of server.py with comprehensive monitoring.
Tracks request count, duration, and indicator computation time.
"""

import asyncio
import json
import os
import socket
import struct
import sys
from pathlib import Path
from typing import Any, Dict
from aiohttp import web

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Import metrics
from prometheus_metrics import (
  track_request,
  track_indicator_compute,
  track_connection,
  track_time,
  record_error,
  set_service_info,
  get_metrics,
  active_connections
)

SOCKET_PATH = "/tmp/mojo-compute.sock"
METRICS_PORT = 9090


class MojoComputeServer:
  """Unix socket server with Prometheus metrics integration."""

  def __init__(self, socket_path: str = SOCKET_PATH):
    self.socket_path = socket_path
    self.server_socket = None

    # Set service info
    set_service_info(
      version='1.0.0',
      mojo_version='24.5.0',
      simd_enabled=True
    )

  async def start(self):
    """Start the Unix socket server."""
    # Remove existing socket file
    if os.path.exists(self.socket_path):
      os.unlink(self.socket_path)

    self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.server_socket.bind(self.socket_path)
    self.server_socket.listen(5)
    self.server_socket.setblocking(False)

    # Set socket permissions
    os.chmod(self.socket_path, 0o666)

    print(f"🚀 Mojo Compute Server listening on {self.socket_path}")
    print(f"📊 Metrics available on http://localhost:{METRICS_PORT}/metrics")

    loop = asyncio.get_event_loop()

    while True:
      client_socket, _ = await loop.sock_accept(self.server_socket)
      asyncio.create_task(self.handle_client(client_socket))

  async def handle_client(self, client_socket: socket.socket):
    """Handle a client connection with metrics tracking."""
    with track_connection():
      try:
        loop = asyncio.get_event_loop()

        # Receive length prefix (4 bytes, big-endian)
        length_bytes = await loop.sock_recv(client_socket, 4)
        if not length_bytes:
          return

        request_length = struct.unpack(">I", length_bytes)[0]

        # Receive request JSON
        request_bytes = await loop.sock_recv(client_socket, request_length)
        request_json = request_bytes.decode("utf-8")
        request = json.loads(request_json)

        action = request.get('action', 'unknown')
        print(f"📥 Request: {action}")

        # Process request
        response = await self.process_request(request)

        # Send response (length-prefixed JSON)
        response_json = json.dumps(response, ensure_ascii=False)
        response_bytes = response_json.encode("utf-8")
        response_length = struct.pack(">I", len(response_bytes))

        await loop.sock_sendall(client_socket, response_length + response_bytes)

        print(f"📤 Response sent: {len(response_bytes)} bytes")

      except Exception as e:
        print(f"❌ Error handling client: {e}")
        record_error(type(e).__name__)

        error_response = {"error": str(e), "status": "error"}
        response_json = json.dumps(error_response)
        response_bytes = response_json.encode("utf-8")
        response_length = struct.pack(">I", len(response_bytes))

        try:
          await loop.sock_sendall(client_socket, response_length + response_bytes)
        except:
          pass

      finally:
        client_socket.close()

  async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a compute request with metrics tracking."""
    action = request.get("action")

    if action == "ping":
      return {"status": "ok", "message": "pong"}

    elif action == "compute_sma":
      return await self.compute_sma(request)

    elif action == "compute_rsi":
      return await self.compute_rsi(request)

    elif action == "compute_ema":
      return await self.compute_ema(request)

    elif action == "compute_macd":
      return await self.compute_macd(request)

    elif action == "compute_bollinger":
      return await self.compute_bollinger(request)

    else:
      record_error('UnknownAction')
      return {"error": f"Unknown action: {action}", "status": "error"}

  @track_request('compute_sma')
  async def compute_sma(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Simple Moving Average with metrics tracking."""
    prices = request.get("prices", [])
    period = request.get("period", 20)
    symbol = request.get("symbol", "UNKNOWN")

    if not prices:
      record_error('EmptyPrices')
      return {"error": "No prices provided", "status": "error"}

    # Track indicator computation time
    with track_time('indicator_compute', {'indicator': 'sma'}):
      # Simple Python SMA implementation (will be replaced with Mojo call)
      sma_values = []
      for i in range(len(prices)):
        if i < period - 1:
          sma_values.append(0.0)
        else:
          window = prices[i - period + 1 : i + 1]
          sma_values.append(sum(window) / period)

    return {
      "status": "ok",
      "symbol": symbol,
      "indicator": "sma",
      "period": period,
      "values": sma_values,
      "computed_by": "mojo-compute (Python wrapper)",
    }

  @track_request('compute_rsi')
  async def compute_rsi(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Relative Strength Index with metrics tracking."""
    prices = request.get("prices", [])
    period = request.get("period", 14)
    symbol = request.get("symbol", "UNKNOWN")

    if not prices:
      record_error('EmptyPrices')
      return {"error": "No prices provided", "status": "error"}

    with track_time('indicator_compute', {'indicator': 'rsi'}):
      # Mock RSI implementation
      rsi_values = [0.0] * len(prices)
      if len(prices) >= period + 1:
        for i in range(period, len(prices)):
          rsi_values[i] = 50.0 + (i % 40) - 20  # Mock data

    return {
      "status": "ok",
      "symbol": symbol,
      "indicator": "rsi",
      "period": period,
      "values": rsi_values,
      "computed_by": "mojo-compute (Python wrapper)",
    }

  @track_request('compute_ema')
  async def compute_ema(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Exponential Moving Average with metrics tracking."""
    prices = request.get("prices", [])
    period = request.get("period", 12)
    symbol = request.get("symbol", "UNKNOWN")

    if not prices:
      record_error('EmptyPrices')
      return {"error": "No prices provided", "status": "error"}

    with track_time('indicator_compute', {'indicator': 'ema'}):
      # EMA implementation
      multiplier = 2.0 / (period + 1)
      ema_values = [0.0] * len(prices)

      if len(prices) >= period:
        # Start with SMA for first value
        ema_values[period - 1] = sum(prices[:period]) / period

        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
          ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]

    return {
      "status": "ok",
      "symbol": symbol,
      "indicator": "ema",
      "period": period,
      "values": ema_values,
      "computed_by": "mojo-compute (Python wrapper)",
    }

  @track_request('compute_macd')
  async def compute_macd(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute MACD indicator with metrics tracking."""
    with track_time('indicator_compute', {'indicator': 'macd'}):
      pass  # Mock implementation

    return {
      "status": "ok",
      "indicator": "macd",
      "macd_line": [0.0],
      "signal_line": [0.0],
      "histogram": [0.0],
      "computed_by": "mojo-compute (Python wrapper - TODO)",
    }

  @track_request('compute_bollinger')
  async def compute_bollinger(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Bollinger Bands with metrics tracking."""
    with track_time('indicator_compute', {'indicator': 'bollinger'}):
      pass  # Mock implementation

    return {
      "status": "ok",
      "indicator": "bollinger",
      "upper_band": [0.0],
      "middle_band": [0.0],
      "lower_band": [0.0],
      "computed_by": "mojo-compute (Python wrapper - TODO)",
    }


# ============================================================================
# Metrics HTTP Server
# ============================================================================

async def metrics_handler(request):
  """HTTP handler for Prometheus metrics endpoint."""
  metrics = get_metrics()
  return web.Response(body=metrics, content_type='text/plain; charset=utf-8')


async def health_handler(request):
  """Health check endpoint."""
  return web.json_response({
    "status": "healthy",
    "active_connections": active_connections._value._value
  })


async def start_metrics_server(port: int = METRICS_PORT):
  """Start HTTP server for Prometheus metrics."""
  app = web.Application()
  app.router.add_get('/metrics', metrics_handler)
  app.router.add_get('/health', health_handler)

  runner = web.AppRunner(app)
  await runner.setup()

  site = web.TCPSite(runner, '0.0.0.0', port)
  await site.start()

  print(f"📊 Metrics server started on http://0.0.0.0:{port}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
  """Main entry point."""
  # Start metrics HTTP server
  await start_metrics_server(METRICS_PORT)

  # Start compute server
  server = MojoComputeServer()

  try:
    await server.start()

  except KeyboardInterrupt:
    print("\n🛑 Server stopped")

  finally:
    if os.path.exists(SOCKET_PATH):
      os.unlink(SOCKET_PATH)


if __name__ == "__main__":
  asyncio.run(main())
