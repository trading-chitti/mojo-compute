# Production Monitoring with Prometheus

Comprehensive monitoring solution for Mojo Compute Service using Prometheus metrics.

## Overview

This monitoring system tracks key performance indicators (KPIs) for the Mojo compute service:

- **Request Metrics**: Total requests, duration, status
- **Indicator Metrics**: Computation time, data points processed
- **Error Metrics**: Error counts by type
- **Connection Metrics**: Active connections
- **System Metrics**: Service info, SIMD configuration

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Mojo Compute Service                    │
│                                                 │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │ Unix Socket  │────────>│ Prometheus       │  │
│  │ Server       │         │ Metrics Module   │  │
│  │ (Port 8080)  │         │                  │  │
│  └──────────────┘         └─────────────────┘  │
│                                  │              │
│                                  v              │
│                           ┌─────────────────┐  │
│                           │ HTTP Server     │  │
│                           │ /metrics        │  │
│                           │ (Port 9090)     │  │
│                           └─────────────────┘  │
└─────────────────────────────────────────────────┘
                                  │
                                  v
                        ┌──────────────────┐
                        │   Prometheus     │
                        │   Server         │
                        │   (Port 9091)    │
                        └──────────────────┘
                                  │
                                  v
                        ┌──────────────────┐
                        │     Grafana      │
                        │   Dashboards     │
                        │   (Port 3000)    │
                        └──────────────────┘
```

## Components

### 1. Prometheus Metrics Module (`prometheus_metrics.py`)

Core metrics library providing:
- Metric definitions (Counter, Histogram, Gauge, Summary)
- Decorators for automatic tracking
- Context managers for time tracking
- Helper functions for manual instrumentation

### 2. Metrics-Enabled Server (`server_with_metrics.py`)

Enhanced version of the compute server with:
- Automatic request tracking
- Indicator computation time measurement
- Error tracking
- Connection monitoring
- HTTP endpoint for Prometheus scraping

## Metrics Reference

### Request Metrics

#### `mojo_compute_requests_total`

**Type**: Counter
**Labels**: `action`, `status`
**Description**: Total number of requests processed

```python
# Example values
mojo_compute_requests_total{action="compute_sma", status="success"} 1523
mojo_compute_requests_total{action="compute_rsi", status="error"} 3
```

#### `mojo_compute_request_duration_seconds`

**Type**: Histogram
**Labels**: `action`
**Buckets**: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
**Description**: Request processing duration in seconds

```python
# Example query (PromQL)
histogram_quantile(0.95, rate(mojo_compute_request_duration_seconds_bucket[5m]))
```

### Indicator Metrics

#### `mojo_compute_indicator_time_seconds`

**Type**: Histogram
**Labels**: `indicator`
**Buckets**: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
**Description**: Indicator computation time in seconds

```python
# Example values
mojo_compute_indicator_time_seconds{indicator="sma"} 0.0012
mojo_compute_indicator_time_seconds{indicator="rsi"} 0.0023
```

#### `mojo_compute_indicator_data_points`

**Type**: Summary
**Labels**: `indicator`
**Description**: Number of data points processed per calculation

### Error Metrics

#### `mojo_compute_errors_total`

**Type**: Counter
**Labels**: `error_type`
**Description**: Total number of errors by type

```python
# Example values
mojo_compute_errors_total{error_type="ValueError"} 5
mojo_compute_errors_total{error_type="TimeoutError"} 2
mojo_compute_errors_total{error_type="EmptyPrices"} 8
```

### Connection Metrics

#### `mojo_compute_active_connections`

**Type**: Gauge
**Description**: Current number of active connections

```python
# Example value
mojo_compute_active_connections 12
```

### System Metrics

#### `mojo_compute_info`

**Type**: Info
**Description**: Service configuration and version information

```python
# Example value
mojo_compute_info{version="1.0.0", mojo_version="24.5.0", simd_enabled="true", simd_width="4"} 1
```

#### `mojo_compute_indicator_speedup`

**Type**: Gauge
**Labels**: `indicator`
**Description**: Speedup factor vs Python baseline

```python
# Example values
mojo_compute_indicator_speedup{indicator="sma"} 1000.0
mojo_compute_indicator_speedup{indicator="ema"} 750.0
mojo_compute_indicator_speedup{indicator="rsi"} 666.0
```

## Usage

### Starting the Metrics-Enabled Server

```bash
cd /Users/hariprasath/trading-chitti/mojo-compute/monitoring
python server_with_metrics.py
```

**Output:**
```
🚀 Mojo Compute Server listening on /tmp/mojo-compute.sock
📊 Metrics available on http://localhost:9090/metrics
```

### Accessing Metrics

#### HTTP Endpoints

1. **Metrics Endpoint**: `http://localhost:9090/metrics`
   - Returns Prometheus-formatted metrics
   - Scraped by Prometheus server

2. **Health Check**: `http://localhost:9090/health`
   - Returns service health status
   - Shows active connections

```bash
# View metrics
curl http://localhost:9090/metrics

# Health check
curl http://localhost:9090/health
```

### Using Decorators

#### Track Request Processing

```python
from prometheus_metrics import track_request

@track_request('compute_sma')
async def compute_sma(request):
  # Your implementation
  return result
```

#### Track Indicator Computation

```python
from prometheus_metrics import track_indicator_compute

@track_indicator_compute('sma')
def compute_sma_mojo(prices, period):
  # Your implementation
  return sma_values
```

### Using Context Managers

#### Track Execution Time

```python
from prometheus_metrics import track_time

with track_time('indicator_compute', {'indicator': 'sma'}):
  result = expensive_computation()
```

#### Track Connections

```python
from prometheus_metrics import track_connection

with track_connection():
  await handle_client(socket)
```

### Manual Instrumentation

```python
from prometheus_metrics import record_error, record_indicator_speedup

# Record error
try:
  result = risky_operation()
except ValueError as e:
  record_error('ValueError')

# Record speedup measurement
record_indicator_speedup('sma', 1000.0)
```

## Prometheus Configuration

### Installation

```bash
# macOS
brew install prometheus

# Linux
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*
```

### Configuration File (`prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mojo-compute'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
```

### Running Prometheus

```bash
# Start Prometheus
prometheus --config.file=prometheus.yml --storage.tsdb.path=./data
```

Access Prometheus UI at `http://localhost:9091`

## Grafana Dashboards

### Installation

```bash
# macOS
brew install grafana

# Linux
sudo apt-get install grafana
```

### Starting Grafana

```bash
# macOS
brew services start grafana

# Linux
sudo systemctl start grafana-server
```

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

### Dashboard Setup

1. **Add Prometheus Data Source**
   - URL: `http://localhost:9091`
   - Access: Server (default)

2. **Import Dashboard**
   - Use the provided `grafana_dashboard.json` (see below)

### Sample Dashboard Panels

#### 1. Request Rate

```promql
rate(mojo_compute_requests_total[5m])
```

#### 2. Request Duration (p95)

```promql
histogram_quantile(0.95,
  rate(mojo_compute_request_duration_seconds_bucket[5m])
)
```

#### 3. Indicator Performance

```promql
rate(mojo_compute_indicator_time_seconds_sum[5m])
/
rate(mojo_compute_indicator_time_seconds_count[5m])
```

#### 4. Error Rate

```promql
rate(mojo_compute_errors_total[5m])
```

#### 5. Active Connections

```promql
mojo_compute_active_connections
```

## Alerting Rules

### Sample Alert Configuration

Create `alerts.yml`:

```yaml
groups:
  - name: mojo_compute_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(mojo_compute_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      # Slow request processing
      - alert: SlowRequests
        expr: |
          histogram_quantile(0.95,
            rate(mojo_compute_request_duration_seconds_bucket[5m])
          ) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow request processing"
          description: "P95 latency is {{ $value }} seconds"

      # Too many active connections
      - alert: HighConnectionCount
        expr: mojo_compute_active_connections > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High number of active connections"
          description: "{{ $value }} active connections"

      # Low speedup (performance degradation)
      - alert: LowSpeedup
        expr: mojo_compute_indicator_speedup{indicator="sma"} < 500
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Performance degradation detected"
          description: "SMA speedup dropped to {{ $value }}x"
```

## Performance Monitoring Queries

### Top 10 Most Common Errors

```promql
topk(10, sum by (error_type) (mojo_compute_errors_total))
```

### Request Success Rate

```promql
sum(rate(mojo_compute_requests_total{status="success"}[5m]))
/
sum(rate(mojo_compute_requests_total[5m])) * 100
```

### Average Indicator Computation Time

```promql
avg by (indicator) (
  rate(mojo_compute_indicator_time_seconds_sum[5m])
  /
  rate(mojo_compute_indicator_time_seconds_count[5m])
)
```

### Requests Per Second by Action

```promql
sum by (action) (rate(mojo_compute_requests_total[5m]))
```

## Best Practices

### 1. Metric Naming

- Use `_total` suffix for counters
- Use `_seconds` for time measurements
- Include units in metric names

### 2. Label Cardinality

- Keep label cardinality low (< 100 unique values)
- Avoid high-cardinality labels (IDs, timestamps)
- Use labels for dimensions, not for data values

### 3. Histogram Buckets

- Choose buckets based on expected value ranges
- Include buckets for outliers
- Use exponential bucket spacing

### 4. Scrape Intervals

- Default: 15s for most metrics
- High-frequency: 5s for critical metrics
- Low-frequency: 60s for slow-changing metrics

## Troubleshooting

### Issue: Metrics not appearing

**Check:**
```bash
# Verify metrics endpoint is accessible
curl http://localhost:9090/metrics

# Check Prometheus targets
# Go to http://localhost:9091/targets
```

### Issue: High cardinality warnings

**Solution:**
- Review label usage
- Reduce unique label values
- Use relabeling in Prometheus config

### Issue: Missing historical data

**Solution:**
- Check Prometheus retention settings
- Verify storage disk space
- Review scrape interval configuration

## Integration with Existing Services

### Core API Integration

The core API can query metrics programmatically:

```python
import requests

# Get current metrics
response = requests.get('http://localhost:9090/metrics')
metrics = response.text

# Parse metrics (using prometheus_client)
from prometheus_client.parser import text_string_to_metric_families

for family in text_string_to_metric_families(metrics):
  for sample in family.samples:
    print(f"{sample.name}{sample.labels} = {sample.value}")
```

## Security Considerations

### 1. Restrict Metrics Access

```yaml
# prometheus.yml - Add basic auth
scrape_configs:
  - job_name: 'mojo-compute'
    basic_auth:
      username: prometheus
      password: secure_password
```

### 2. Firewall Rules

```bash
# Allow only local access to metrics port
sudo ufw allow from 127.0.0.1 to any port 9090
```

### 3. HTTPS for Metrics

Use reverse proxy (nginx) with TLS:

```nginx
server {
  listen 443 ssl;
  server_name metrics.example.com;

  ssl_certificate /path/to/cert.pem;
  ssl_certificate_key /path/to/key.pem;

  location /metrics {
    proxy_pass http://localhost:9090;
  }
}
```

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [prometheus_client Library](https://github.com/prometheus/client_python)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)

---

**Last Updated**: 2026-01-30
**Version**: 1.0.0
**Python Version**: 3.9+
