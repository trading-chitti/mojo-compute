#!/bin/bash
#
# Mojo Compute Service - Production Installation Script
# Run as root or with sudo
#

set -e  # Exit on error

echo "=========================================="
echo "Mojo Compute Service - Installation"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "ERROR: This script must be run as root"
  echo "Usage: sudo ./install.sh"
  exit 1
fi

# Configuration
APP_DIR="/opt/trading-chitti/mojo-compute"
USER="trading-chitti"
GROUP="trading-chitti"
SOCKET_DIR="/var/run/mojo-compute"
LOG_DIR="/var/log/mojo-compute"

echo "1. Creating application directory..."
mkdir -p "$APP_DIR"
echo "   ✓ Created $APP_DIR"

echo ""
echo "2. Creating user and group..."
if ! getent group "$GROUP" > /dev/null 2>&1; then
  groupadd -r "$GROUP"
  echo "   ✓ Created group: $GROUP"
else
  echo "   → Group already exists: $GROUP"
fi

if ! getent passwd "$USER" > /dev/null 2>&1; then
  useradd -r -g "$GROUP" -s /bin/bash -d /opt/trading-chitti "$USER"
  echo "   ✓ Created user: $USER"
else
  echo "   → User already exists: $USER"
fi

echo ""
echo "3. Copying application files..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"

cp -r "$SOURCE_DIR"/* "$APP_DIR/"
echo "   ✓ Copied application files"

echo ""
echo "4. Setting permissions..."
chown -R "$USER:$GROUP" "$APP_DIR"
chmod 750 "$APP_DIR"
chmod 640 "$APP_DIR"/server.py
chmod 755 "$APP_DIR"/deploy/healthcheck.py
echo "   ✓ Set file permissions"

echo ""
echo "5. Creating runtime directories..."
mkdir -p "$SOCKET_DIR"
chown "$USER:$GROUP" "$SOCKET_DIR"
chmod 755 "$SOCKET_DIR"
echo "   ✓ Created $SOCKET_DIR"

mkdir -p "$LOG_DIR"
chown "$USER:$GROUP" "$LOG_DIR"
chmod 755 "$LOG_DIR"
echo "   ✓ Created $LOG_DIR"

echo ""
echo "6. Installing systemd service..."
cp "$APP_DIR/deploy/mojo-compute.service" /etc/systemd/system/
systemctl daemon-reload
echo "   ✓ Service file installed"

echo ""
echo "7. Enabling service..."
systemctl enable mojo-compute
echo "   ✓ Service enabled (will start on boot)"

echo ""
echo "=========================================="
echo "Installation Complete! ✅"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start the service:     sudo systemctl start mojo-compute"
echo "  2. Check status:          sudo systemctl status mojo-compute"
echo "  3. View logs:             sudo journalctl -u mojo-compute -f"
echo "  4. Run health check:      sudo -u $USER $APP_DIR/deploy/healthcheck.py"
echo ""
echo "Documentation: $APP_DIR/deploy/DEPLOYMENT.md"
echo ""
