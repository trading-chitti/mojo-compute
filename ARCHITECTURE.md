# Trading-Chitti Architecture: All-Mojo Backend

**Last Updated**: 2024-01-30

---

## 🎯 Design Philosophy

**Maximum Performance with Pragmatic Web Layer**

- **Frontend**: React (mature UI framework)
- **API Gateway**: FastAPI Python (handles HTTP/SSE/WebSocket complexity)
- **Business Logic**: Mojo (ALL services - 35,000x faster)
- **Database**: PostgreSQL + TimescaleDB

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              dashboard-app (React + TypeScript)        │    │
│  │                                                         │    │
│  │  - TradingView Charts                                  │    │
│  │  - Real-time SSE streaming                             │    │
│  │  - Signal dashboard                                    │    │
│  │  - Backtest visualization                              │    │
│  └────────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/SSE/WebSocket
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │           core-api (FastAPI + Python 3.11)             │    │
│  │                                                         │    │
│  │  Responsibilities:                                     │    │
│  │  - HTTP request routing                                │    │
│  │  - SSE event streaming                                 │    │
│  │  - WebSocket management                                │    │
│  │  - Authentication/Authorization                        │    │
│  │  - CORS handling                                       │    │
│  │  - Request validation (Pydantic)                       │    │
│  │  - Response serialization (JSON)                       │    │
│  │                                                         │    │
│  │  Does NOT contain business logic!                      │    │
│  │  Just routes to Mojo services via:                     │    │
│  │  - Unix sockets (local IPC)                            │    │
│  │  - TCP sockets (distributed)                           │    │
│  │  - gRPC (future - when Mojo supports it)               │    │
│  └────────────────────────────────────────────────────────┘    │
└───────┬──────────────┬──────────────┬─────────────────────────┘
        │              │              │
        │ IPC/TCP      │ IPC/TCP      │ IPC/TCP
        ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MOJO SERVICES LAYER                          │
│                     (All business logic)                         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  news-nlp    │  │signal-service│  │mojo-compute  │         │
│  │   (MOJO)     │  │   (MOJO)     │  │   (MOJO)     │         │
│  │              │  │              │  │              │         │
│  │ RSS ingest   │  │ Alert gen    │  │ Indicators   │         │
│  │ XML parsing  │  │ Pattern match│  │ Backtesting  │         │
│  │ NLP/sentiment│  │ Signal logic │  │ ML inference │         │
│  │ Entity recog │  │ Event stream │  │ Feature eng  │         │
│  │ DB write     │  │ DB query     │  │ Optimization │         │
│  │              │  │              │  │              │         │
│  │ Socket API   │  │ Socket API   │  │ Socket API   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ libpq (C FFI)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATABASE LAYER                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │         PostgreSQL 15+ with TimescaleDB                │    │
│  │                                                         │    │
│  │  Schemas:                                              │    │
│  │  - news.*    (articles, entities, sentiments)          │    │
│  │  - md.*      (market data, EOD, indicators)            │    │
│  │  - ml.*      (models, predictions, features)           │    │
│  │  - signals.* (alerts, patterns, backtests)             │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Service Responsibilities

### 1. dashboard-app (React) - UNCHANGED

**Technology**: React 18+ TypeScript, Vite, TailwindCSS

**Responsibilities**:
- User interface rendering
- Chart visualization (TradingView Lightweight Charts)
- Real-time updates (SSE subscription)
- User interactions
- Client-side state management

**Communication**: HTTP + SSE to core-api

**Port**: 5173 (dev), 80/443 (prod)

---

### 2. core-api (FastAPI Python) - NEW THIN GATEWAY

**Technology**: FastAPI, Python 3.11, Pydantic

**Responsibilities**:
- ✅ HTTP request routing
- ✅ SSE event streaming (Server-Sent Events)
- ✅ WebSocket management (future)
- ✅ Authentication (JWT, OAuth2)
- ✅ CORS configuration
- ✅ Request validation (Pydantic schemas)
- ✅ Response serialization (JSON)
- ✅ Rate limiting
- ✅ API documentation (OpenAPI/Swagger)
- ❌ **NO business logic** (just routing!)

**Endpoints**:
```
GET  /health                    → Check all services
GET  /api/alerts                → news-nlp (Mojo)
GET  /api/alerts/stream         → signal-service (Mojo) SSE
POST /api/compute/sma           → mojo-compute
POST /api/compute/batch         → mojo-compute
GET  /api/backtest/results/{id} → mojo-compute
```

**Communication**:
- Inbound: HTTP/SSE from dashboard
- Outbound: Unix sockets to Mojo services (or TCP if distributed)

**Port**: 6001

---

### 3. news-nlp (MOJO) - REWRITTEN

**Technology**: Mojo, C FFI (libxml2, libpq, libcurl)

**Responsibilities**:
- ✅ RSS feed ingestion (4 Google News feeds)
- ✅ XML/RSS parsing (FFI to libxml2 or manual)
- ✅ Article extraction (title, summary, link, date)
- ✅ NLP sentiment analysis (Mojo implementation)
- ✅ Named Entity Recognition (stock symbols, sectors)
- ✅ Direction classification (bullish, bearish, neutral)
- ✅ Database writes (PostgreSQL via libpq FFI)
- ✅ Scheduled jobs (cron-like in Mojo)

**Performance Gains**:
- RSS parsing: 100x faster than feedparser
- Sentiment analysis: 1000x faster than transformers
- Database writes: 10x faster (batching + native)

**API** (Socket-based):
```
Request:  {"action": "ingest_rss", "url": "..."}
Response: {"status": "ok", "articles": 42, "time_ms": 5.2}

Request:  {"action": "analyze_sentiment", "text": "..."}
Response: {"sentiment": "bullish", "score": 0.82}
```

**Port**: Unix socket `/tmp/news-nlp.sock` (or TCP 6002)

---

### 4. signal-service (MOJO) - REWRITTEN

**Technology**: Mojo, C FFI (libpq)

**Responsibilities**:
- ✅ Alert generation (from news + price data)
- ✅ Pattern matching (technical + fundamental)
- ✅ Signal scoring (impact, confidence)
- ✅ Event streaming (publish alerts)
- ✅ Database queries (read alerts)
- ✅ Real-time filtering

**Performance Gains**:
- Pattern matching: 500x faster
- Database queries: 20x faster (SIMD for filtering)
- Alert generation: 100x faster

**API** (Socket-based):
```
Request:  {"action": "get_alerts", "limit": 100}
Response: {"alerts": [...], "count": 42}

Request:  {"action": "generate_signals", "symbol": "TCS"}
Response: {"signals": [...], "generated": 5}

Stream:   {"event": "new_alert", "data": {...}}
```

**Port**: Unix socket `/tmp/signal-service.sock` (or TCP 6003)

---

### 5. mojo-compute (MOJO) - AS PLANNED

**Technology**: Mojo, SIMD, GPU (future)

**Responsibilities**:
- ✅ Technical indicators (100+)
- ✅ Backtesting engine (vectorized)
- ✅ ML model inference (LightGBM, sklearn via ONNX)
- ✅ Feature engineering (200+ features)
- ✅ Portfolio optimization
- ✅ Risk calculations

**Performance Gains**:
- Indicators: 100x faster than NumPy
- Backtesting: 60x faster than vectorbt
- ML inference: 1000x faster than sklearn

**API** (Socket-based):
```
Request:  {"action": "compute_sma", "prices": [...], "period": 20}
Response: {"values": [...], "time_ms": 0.15}

Request:  {"action": "backtest", "strategy": {...}, "data": {...}}
Response: {"sharpe": 1.42, "return": 0.156, "trades": [...]}
```

**Port**: Unix socket `/tmp/mojo-compute.sock` (or TCP 6004)

---

## 🔗 Inter-Service Communication

### Option 1: Unix Domain Sockets (Recommended for localhost)

**Pros**:
- ✅ Fastest IPC (no network overhead)
- ✅ More secure (file system permissions)
- ✅ Lower latency (<1μs)

**Cons**:
- ❌ Only works on same machine
- ❌ Harder to scale horizontally

**Implementation**:
```mojo
# Mojo service listens on Unix socket
let socket = UnixSocket("/tmp/news-nlp.sock")
socket.listen()

while True:
    let client = socket.accept()
    let request = client.recv()
    let response = handle_request(request)
    client.send(response)
```

```python
# Python core-api connects
import socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/news-nlp.sock")
sock.send(b'{"action":"get_alerts"}')
response = sock.recv(4096)
```

---

### Option 2: TCP Sockets (For distributed deployment)

**Pros**:
- ✅ Works across machines
- ✅ Easy to scale horizontally
- ✅ Load balancing possible

**Cons**:
- ❌ Slightly higher latency (~100μs)
- ❌ Network security needed

**Ports**:
- core-api: 6001 (FastAPI HTTP)
- news-nlp: 6002 (Mojo socket API)
- signal-service: 6003 (Mojo socket API)
- mojo-compute: 6004 (Mojo socket API)

---

## 🗄️ Database Architecture

**Single PostgreSQL instance, multiple schemas**

```sql
-- News data
CREATE SCHEMA news;
CREATE TABLE news.articles (...);
CREATE TABLE news.entities (...);

-- Market data
CREATE SCHEMA md;
CREATE TABLE md.eod_prices (...);
CREATE TABLE md.indicators (...);

-- ML data
CREATE SCHEMA ml;
CREATE TABLE ml.predictions (...);
CREATE TABLE ml.features (...);

-- Signals
CREATE SCHEMA signals;
CREATE TABLE signals.alerts (...);
CREATE TABLE signals.backtests (...);
```

**All Mojo services connect directly** using FFI to libpq (no ORMs, native speed).

---

## 🚀 Deployment

### Development (localhost)

```bash
# Start database
docker-compose up postgres

# Start Mojo services (each in own terminal)
mojo run news-nlp/main.mojo          # Port 6002 or /tmp/news-nlp.sock
mojo run signal-service/main.mojo    # Port 6003 or /tmp/signal-service.sock
mojo run mojo-compute/main.mojo      # Port 6004 or /tmp/mojo-compute.sock

# Start Python gateway
uvicorn core-api.app:app --port 6001 --reload

# Start React dashboard
cd dashboard-app && npm run dev      # Port 5173
```

### Production (Docker)

```yaml
# docker-compose.yml
services:
  postgres:
    image: timescale/timescaledb:latest-pg15

  news-nlp-mojo:
    build: ./news-nlp
    volumes:
      - /tmp:/tmp  # Unix sockets
    depends_on: [postgres]

  signal-service-mojo:
    build: ./signal-service
    volumes:
      - /tmp:/tmp
    depends_on: [postgres, news-nlp-mojo]

  mojo-compute:
    build: ./mojo-compute
    volumes:
      - /tmp:/tmp
    depends_on: [postgres]

  core-api:
    build: ./core-api
    ports:
      - "6001:6001"
    volumes:
      - /tmp:/tmp  # Connect to Unix sockets
    depends_on: [news-nlp-mojo, signal-service-mojo, mojo-compute]

  dashboard:
    build: ./dashboard-app
    ports:
      - "80:80"
    depends_on: [core-api]
```

---

## 📊 Performance Expectations

| Component | Python Baseline | Mojo Expected | Speedup |
|-----------|-----------------|---------------|---------|
| RSS Parsing | 500ms/feed | 5ms/feed | 100x |
| Sentiment Analysis | 200ms/article | 0.2ms/article | 1000x |
| Entity Recognition | 100ms/article | 1ms/article | 100x |
| Alert Generation | 50ms/symbol | 0.5ms/symbol | 100x |
| Pattern Matching | 20ms/pattern | 0.2ms/pattern | 100x |
| SMA (10K points) | 80ms | 0.8ms | 100x |
| RSI (10K points) | 120ms | 1.5ms | 80x |
| Backtesting (1yr) | 3000ms | 50ms | 60x |
| ML Inference | 10ms | 0.01ms | 1000x |

**Overall System Throughput**:
- Current (Python): ~100 requests/second
- Target (Mojo): ~10,000 requests/second (100x improvement)

---

## 🔒 Security

### API Gateway (core-api)
- JWT authentication
- Rate limiting (per IP, per user)
- CORS configuration
- Input validation (Pydantic)
- SQL injection prevention

### Mojo Services
- No direct external access (only via core-api)
- Unix socket permissions (0600)
- Input sanitization
- Prepared statements for DB queries

### Database
- Connection pooling
- Read-only users for query-only services
- Row-level security (future)

---

## 📈 Scalability

### Horizontal Scaling

```
                    ┌─────────────┐
                    │ Load Balancer│
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  core-api-1  │   │  core-api-2  │   │  core-api-3  │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ news-nlp    │   │ signal-svc  │   │ mojo-compute│
│ (Mojo) x3   │   │ (Mojo) x3   │   │ (Mojo) x5   │
└─────────────┘   └─────────────┘   └─────────────┘
```

### Vertical Scaling

- Mojo's SIMD: Use all CPU cores
- GPU acceleration (future): CUDA via Mojo
- Memory efficiency: 10x less RAM than Python

---

## 🎯 Migration Priority

1. **Phase 1**: Build mojo-compute (CURRENT)
2. **Phase 2**: Build core-api gateway (Python, connects to existing services + mojo-compute)
3. **Phase 3**: Rewrite signal-service in Mojo
4. **Phase 4**: Rewrite news-nlp in Mojo
5. **Phase 5**: Optimize and scale

**Timeline**: 12-18 months for full migration

---

## ✅ Success Criteria

- [ ] All services communicate via sockets
- [ ] 100x performance improvement overall
- [ ] <10ms p95 latency for all APIs
- [ ] 10,000+ requests/second throughput
- [ ] <500MB memory usage per service
- [ ] Zero downtime deployments
- [ ] 99.9% uptime
- [ ] Full test coverage (unit + integration)

---

This architecture gives you:
✅ **Performance** (Mojo for all business logic)
✅ **Simplicity** (FastAPI for web complexity)
✅ **Scalability** (stateless services, socket-based)
✅ **Maintainability** (clean separation of concerns)
