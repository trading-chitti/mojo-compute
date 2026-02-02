#!/usr/bin/env python3
"""
Mojo Compute Socket Server

Listens on Unix socket and processes indicator requests.
Calls Mojo-implemented indicators for maximum performance.
"""

import asyncio
import json
import os
import socket
import struct
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path for importing indicators module
sys.path.insert(0, str(Path(__file__).parent / "src"))

SOCKET_PATH = "/tmp/mojo-compute.sock"


class MojoComputeServer:
  """Unix socket server for Mojo compute service."""

  def __init__(self, socket_path: str = SOCKET_PATH):
    self.socket_path = socket_path
    self.server_socket = None

  async def start(self):
    """Start the Unix socket server."""
    # Remove existing socket file
    if os.path.exists(self.socket_path):
      os.unlink(self.socket_path)

    self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.server_socket.bind(self.socket_path)
    self.server_socket.listen(5)
    self.server_socket.setblocking(False)

    # Set socket permissions to allow core-api to connect
    os.chmod(self.socket_path, 0o666)

    print(f"🚀 Mojo Compute Server listening on {self.socket_path}")

    loop = asyncio.get_event_loop()

    while True:
      client_socket, _ = await loop.sock_accept(self.server_socket)
      asyncio.create_task(self.handle_client(client_socket))

  async def recv_exact(self, loop, client_socket: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket, handling partial reads."""
    data = b''
    while len(data) < n:
      chunk = await loop.sock_recv(client_socket, n - len(data))
      if not chunk:
        raise ConnectionError("Socket closed before receiving all data")
      data += chunk
    return data

  async def handle_client(self, client_socket: socket.socket):
    """Handle a client connection."""
    try:
      loop = asyncio.get_event_loop()

      # Receive length prefix (4 bytes, big-endian) - ensure complete read
      length_bytes = await self.recv_exact(loop, client_socket, 4)

      request_length = struct.unpack(">I", length_bytes)[0]

      # Receive request JSON - ensure complete read
      request_bytes = await self.recv_exact(loop, client_socket, request_length)
      request_json = request_bytes.decode("utf-8")
      request = json.loads(request_json)

      print(f"📥 Request: {request.get('action', 'unknown')}")

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
      error_response = {"error": str(e), "status": "error"}
      response_json = json.dumps(error_response)
      response_bytes = response_json.encode("utf-8")
      response_length = struct.pack(">I", len(response_bytes))
      try:
        await loop.sock_sendall(client_socket, response_length + response_bytes)
      except:
        pass

    finally:
      # Always close the client socket when done
      client_socket.close()

  async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a compute request.

    Args:
      request: JSON request with action and parameters.

    Returns:
      JSON response with results.
    """
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
      return {"error": f"Unknown action: {action}", "status": "error"}

  async def compute_sma(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Simple Moving Average.

    TODO: Call Mojo implementation via FFI or subprocess.
    For now, using Python implementation.
    """
    prices = request.get("prices", [])
    period = request.get("period", 20)
    symbol = request.get("symbol", "UNKNOWN")

    if not prices:
      return {"error": "No prices provided", "status": "error"}

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

  async def compute_rsi(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Relative Strength Index."""
    prices = request.get("prices", [])
    period = request.get("period", 14)
    symbol = request.get("symbol", "UNKNOWN")

    if not prices:
      return {"error": "No prices provided", "status": "error"}

    # Simple Python RSI implementation (placeholder)
    # TODO: Call Mojo implementation
    rsi_values = [0.0] * len(prices)
    if len(prices) >= period + 1:
      # Mock RSI values
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

  async def compute_ema(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Exponential Moving Average."""
    prices = request.get("prices", [])
    period = request.get("period", 12)
    symbol = request.get("symbol", "UNKNOWN")

    if not prices:
      return {"error": "No prices provided", "status": "error"}

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

  async def compute_macd(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute MACD (Moving Average Convergence Divergence).

    TODO: Call Mojo implementation for better performance.
    For now, using Python implementation for MVP.
    """
    prices = request.get("prices", [])
    fast_period = request.get("fast_period", 12)
    slow_period = request.get("slow_period", 26)
    signal_period = request.get("signal_period", 9)
    symbol = request.get("symbol", "UNKNOWN")

    if not prices:
      return {"error": "No prices provided", "status": "error"}

    # Compute EMA helper function
    def compute_ema_values(data, period):
      ema_values = [0.0] * len(data)
      if len(data) < period:
        return ema_values

      multiplier = 2.0 / (period + 1)
      # First EMA is SMA
      ema_values[period - 1] = sum(data[:period]) / period

      # Subsequent EMAs
      for i in range(period, len(data)):
        ema_values[i] = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]

      return ema_values

    # Calculate fast and slow EMAs
    fast_ema = compute_ema_values(prices, fast_period)
    slow_ema = compute_ema_values(prices, slow_period)

    # Calculate MACD line (fast EMA - slow EMA)
    macd_line = [fast_ema[i] - slow_ema[i] for i in range(len(prices))]

    # Calculate signal line (EMA of MACD line)
    signal_line = compute_ema_values(macd_line, signal_period)

    # Calculate histogram (MACD line - signal line)
    histogram = [macd_line[i] - signal_line[i] for i in range(len(prices))]

    return {
      "status": "ok",
      "symbol": symbol,
      "indicator": "macd",
      "macd_line": macd_line,
      "signal_line": signal_line,
      "histogram": histogram,
      "fast_period": fast_period,
      "slow_period": slow_period,
      "signal_period": signal_period,
      "computed_by": "mojo-compute (Python wrapper - will migrate to Mojo)",
    }

  async def compute_bollinger(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Bollinger Bands.

    TODO: Call Mojo implementation for better performance.
    For now, using Python implementation for MVP.
    """
    prices = request.get("prices", [])
    period = request.get("period", 20)
    std_dev_multiplier = request.get("std_dev", 2.0)
    symbol = request.get("symbol", "UNKNOWN")

    if not prices:
      return {"error": "No prices provided", "status": "error"}

    import math

    # Calculate middle band (SMA)
    middle_band = []
    for i in range(len(prices)):
      if i < period - 1:
        middle_band.append(0.0)
      else:
        window = prices[i - period + 1 : i + 1]
        middle_band.append(sum(window) / period)

    # Calculate upper and lower bands
    upper_band = [0.0] * len(prices)
    lower_band = [0.0] * len(prices)

    for i in range(period - 1, len(prices)):
      window = prices[i - period + 1 : i + 1]
      mean = middle_band[i]

      # Calculate standard deviation
      variance = sum((x - mean) ** 2 for x in window) / period
      std_dev = math.sqrt(variance)

      # Calculate bands
      upper_band[i] = mean + (std_dev_multiplier * std_dev)
      lower_band[i] = mean - (std_dev_multiplier * std_dev)

    return {
      "status": "ok",
      "symbol": symbol,
      "indicator": "bollinger",
      "upper_band": upper_band,
      "middle_band": middle_band,
      "lower_band": lower_band,
      "period": period,
      "std_dev": std_dev_multiplier,
      "computed_by": "mojo-compute (Python wrapper - will migrate to Mojo)",
    }


async def main():
  """Main entry point."""
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
