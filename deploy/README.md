# Mojo Compute - Deployment Files

This directory contains all production deployment files for the mojo-compute service.

---

## 📁 **Files**

| File | Description |
|------|-------------|
| **mojo-compute.service** | Systemd service file for Linux production deployment |
| **Dockerfile** | Docker container configuration |
| **docker-compose.yml** | Docker Compose orchestration file |
| **healthcheck.py** | Health check script (used by systemd & Docker) |
| **install.sh** | Automated installation script for production |
| **DEPLOYMENT.md** | Complete deployment guide and documentation |

---

## 🚀 **Quick Start**

### Option 1: Systemd (Linux Production)

```bash
# Run installation script
sudo ./install.sh

# Start service
sudo systemctl start mojo-compute

# Check status
sudo systemctl status mojo-compute
```

### Option 2: Docker

```bash
# Using Docker Compose
docker-compose up -d

# Or build manually
docker build -f Dockerfile -t mojo-compute ..
docker run -d --name mojo-compute mojo-compute
```

### Option 3: Development

```bash
# Just run the Python server directly
cd ..
python3 server.py
```

---

## 📖 **Documentation**

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Complete setup instructions
- Configuration options
- Health monitoring
- Troubleshooting guide
- Security hardening

---

## ✅ **Health Check**

Test if the service is running:

```bash
./healthcheck.py
```

Expected output:
```
OK: mojo-compute is healthy
```

Exit code `0` = healthy, `1` = unhealthy

---

## 🔧 **Service Management**

### Systemd

```bash
sudo systemctl start mojo-compute    # Start
sudo systemctl stop mojo-compute     # Stop
sudo systemctl restart mojo-compute  # Restart
sudo systemctl status mojo-compute   # Status
sudo journalctl -u mojo-compute -f   # Logs
```

### Docker

```bash
docker-compose up -d                 # Start
docker-compose down                  # Stop
docker-compose restart               # Restart
docker-compose logs -f               # Logs
```

---

## 📊 **Files Created**

After installation with `install.sh`:

```
/opt/trading-chitti/mojo-compute/   # Application directory
/var/run/mojo-compute/              # Socket directory
/var/log/mojo-compute/              # Log directory (if configured)
/etc/systemd/system/mojo-compute.service  # Service file
```

---

## 🔐 **Security**

All deployment files include:
- ✅ Dedicated user/group (trading-chitti)
- ✅ Minimal permissions
- ✅ Read-only system directories
- ✅ No privilege escalation
- ✅ Resource limits
- ✅ Isolated temp directories

---

## 📝 **Notes**

- **2-space indentation**: All files follow project standards
- **Socket path**: Default is `/tmp/mojo-compute.sock` (configurable)
- **Python version**: Requires Python 3.11+
- **Mojo SDK**: Optional (for future performance improvements)

---

**Status**: ✅ Production Ready

See parent directory [README.md](../README_NEW.md) for service overview.
