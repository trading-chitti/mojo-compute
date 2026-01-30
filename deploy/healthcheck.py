#!/usr/bin/env python3
"""
Health check script for mojo-compute service.
Returns 0 if healthy, 1 if unhealthy.
"""

import json
import os
import socket
import struct
import sys


def check_health(socket_path="/tmp/mojo-compute.sock", timeout=5):
  """Check if mojo-compute service is healthy.

  Args:
    socket_path: Path to Unix socket
    timeout: Connection timeout in seconds

  Returns:
    True if healthy, False otherwise
  """
  try:
    # Check if socket file exists
    if not os.path.exists(socket_path):
      print(f"ERROR: Socket file not found: {socket_path}", file=sys.stderr)
      return False

    # Connect to socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(socket_path)

    # Send ping request
    request = json.dumps({"action": "ping"}).encode("utf-8")
    length = struct.pack(">I", len(request))
    sock.sendall(length + request)

    # Receive response
    length_bytes = sock.recv(4)
    if not length_bytes:
      print("ERROR: Empty response from server", file=sys.stderr)
      sock.close()
      return False

    response_length = struct.unpack(">I", length_bytes)[0]
    response_bytes = sock.recv(response_length)
    response = json.loads(response_bytes.decode("utf-8"))

    sock.close()

    # Check response
    if response.get("status") == "ok" and response.get("message") == "pong":
      print("OK: mojo-compute is healthy")
      return True
    else:
      print(f"ERROR: Unexpected response: {response}", file=sys.stderr)
      return False

  except socket.timeout:
    print(f"ERROR: Connection timeout after {timeout}s", file=sys.stderr)
    return False
  except ConnectionRefusedError:
    print("ERROR: Connection refused - service not running?", file=sys.stderr)
    return False
  except Exception as e:
    print(f"ERROR: Health check failed: {e}", file=sys.stderr)
    return False


def main():
  """Main entry point."""
  # Get socket path from environment or use default
  socket_path = os.getenv("SOCKET_PATH", "/tmp/mojo-compute.sock")

  # Run health check
  if check_health(socket_path):
    sys.exit(0)  # Healthy
  else:
    sys.exit(1)  # Unhealthy


if __name__ == "__main__":
  main()
