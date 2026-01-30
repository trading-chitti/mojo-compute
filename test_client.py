#!/usr/bin/env python3
"""Test client for mojo-compute server."""

import json
import socket
import struct

SOCKET_PATH = "/tmp/mojo-compute.sock"


def send_request(request: dict) -> dict:
  """Send request to mojo-compute server and get response."""
  # Connect to Unix socket
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  sock.connect(SOCKET_PATH)

  try:
    # Serialize request to JSON
    request_json = json.dumps(request)
    request_bytes = request_json.encode("utf-8")

    # Send length prefix (4 bytes, big-endian)
    length = struct.pack(">I", len(request_bytes))
    sock.sendall(length + request_bytes)

    # Receive length prefix
    length_bytes = sock.recv(4)
    response_length = struct.unpack(">I", length_bytes)[0]

    # Receive response
    response_bytes = sock.recv(response_length)
    response_json = response_bytes.decode("utf-8")

    return json.loads(response_json)

  finally:
    sock.close()


def main():
  """Run tests."""
  print("=" * 60)
  print("Testing Mojo Compute Server")
  print("=" * 60)

  # Test 1: Ping
  print("\n1. Testing ping...")
  response = send_request({"action": "ping"})
  print(f"Response: {response}")
  assert response["status"] == "ok", "Ping failed!"
  print("✅ Ping successful")

  # Test 2: SMA
  print("\n2. Testing SMA computation...")
  prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0]
  response = send_request({
    "action": "compute_sma",
    "symbol": "TCS",
    "prices": prices,
    "period": 5
  })
  print(f"Response status: {response['status']}")
  print(f"SMA values: {response['values'][:5]}... (showing first 5)")
  assert response["status"] == "ok", "SMA computation failed!"
  print("✅ SMA computation successful")

  # Test 3: RSI
  print("\n3. Testing RSI computation...")
  response = send_request({
    "action": "compute_rsi",
    "symbol": "TCS",
    "prices": prices * 2,  # Need more data for RSI
    "period": 14
  })
  print(f"Response status: {response['status']}")
  print(f"RSI indicator: {response['indicator']}")
  assert response["status"] == "ok", "RSI computation failed!"
  print("✅ RSI computation successful")

  # Test 4: EMA
  print("\n4. Testing EMA computation...")
  response = send_request({
    "action": "compute_ema",
    "symbol": "TCS",
    "prices": prices,
    "period": 5
  })
  print(f"Response status: {response['status']}")
  print(f"EMA values: {response['values'][:5]}... (showing first 5)")
  assert response["status"] == "ok", "EMA computation failed!"
  print("✅ EMA computation successful")

  print("\n" + "=" * 60)
  print("✅ All tests passed!")
  print("=" * 60)


if __name__ == "__main__":
  main()
