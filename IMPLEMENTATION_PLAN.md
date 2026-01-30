# Trading-Chitti: All-Mojo Implementation Plan

**Architecture**: All backend services in Mojo + FastAPI gateway + React dashboard
**Timeline**: 12-18 months
**Team**: Claude Master, SR. Dev Claude, SR. Dev Codex

---

## 🎯 System Architecture

```
React Dashboard → FastAPI Gateway → Mojo Services → PostgreSQL
    (UI)           (core-api)       (ALL business logic)
```

**Services to build/migrate**:
1. ✅ dashboard-app (React) - NO CHANGES
2. 🆕 core-api (FastAPI Python) - NEW thin gateway
3. 🔄 news-nlp (Mojo) - REWRITE from Python
4. 🔄 signal-service (Mojo) - REWRITE from Python
5. ✅ mojo-compute (Mojo) - BUILD from scratch

---

## Phase 1: Mojo Foundation & Core Compute (Months 1-3)

### Month 1: Mojo SDK & Compute Service

#### Week 1-2: Setup & Infrastructure

**Task 1.1: Mojo SDK Installation** [Issue #1]
- Assignee: SR. Dev Codex
- Install Mojo SDK on M1 Pro
- Verify SIMD operations
- Document installation
- Deliverable: `docs/INSTALLATION.md`

**Task 1.2: Python Benchmarks** [Issue #2] ✅ COMPLETED
- Assignee: SR. Dev Claude
- Benchmark SMA, RSI, MACD, Bollinger
- Deliverable: `benchmarks/python_baseline.py`, results CSV

**Task 1.3: API Design** [Issue #3] ✅ COMPLETED
- Assignee: Claude Master
- Design socket-based APIs for all services
- Deliverable: `docs/API_DESIGN.md`

**Task 1.4: Project Structure** [Issue #4]
- Assignee: Claude Master
- Set up monorepo structure for all services
- Deliverable: Complete directory layout

#### Week 3-4: Core Indicators (Mojo)

**Task 2.1: SMA Implementation** [Issue #5]
- Assignee: SR. Dev Claude
- Implement Simple Moving Average in Mojo with SIMD
- Target: 100x faster than NumPy
- Deliverable: `mojo-compute/indicators/sma.mojo`

**Task 2.2: RSI Implementation** [Issue #6]
- Assignee: SR. Dev Codex
- Implement Relative Strength Index in Mojo
- Target: 80x faster than pandas_ta
- Deliverable: `mojo-compute/indicators/rsi.mojo`

**Task 2.3: MACD Implementation** [Issue #7]
- Assignee: SR. Dev Claude
- Implement MACD with EMA calculations
- Deliverable: `mojo-compute/indicators/macd.mojo`

**Task 2.4: Bollinger Bands** [Issue #8]
- Assignee: SR. Dev Codex
- Implement Bollinger Bands with std dev
- Deliverable: `mojo-compute/indicators/bollinger.mojo`

**Task 2.5: Benchmark Comparison** [Issue #9]
- Assignee: Claude Master
- Compare Mojo vs Python performance
- Deliverable: `benchmarks/results/mojo_vs_python.md`

---

### Month 2: Mojo-Python Bridge & Socket API

**Task 3.1: Socket Server (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- Build Unix socket server in Mojo
- Handle JSON request/response
- Connection pooling
- Deliverable: `mojo-compute/api/socket_server.mojo`

**Task 3.2: Mojo-PostgreSQL FFI** [NEW]
- Assignee: SR. Dev Codex
- Create FFI bindings to libpq
- Connection pool management
- Prepared statements
- Deliverable: `shared/postgres/libpq_ffi.mojo`

**Task 3.3: JSON Parser (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- Implement fast JSON parser in Mojo
- Or FFI to rapidjson/simdjson
- Deliverable: `shared/json/parser.mojo`

**Task 3.4: Core-API Gateway (Python)** [NEW]
- Assignee: Claude Master
- Build FastAPI gateway that routes to sockets
- SSE event streaming
- Authentication/CORS
- Deliverable: `core-api/app.py`

**Task 3.5: Integration Tests** [Issue #12]
- Assignee: Claude Master
- End-to-end testing: Dashboard → core-api → mojo-compute
- Deliverable: `tests/integration/test_e2e.py`

---

### Month 3: Advanced Indicators & Backtesting

**Task 4.1: Volume Indicators** [Issue #13]
- Assignee: SR. Dev Claude
- Implement OBV, AD Line, MFI in Mojo
- Deliverable: `mojo-compute/indicators/volume.mojo`

**Task 4.2: Momentum Indicators** [Issue #14]
- Assignee: SR. Dev Codex
- Implement Stochastic, Williams %R, CCI
- Deliverable: `mojo-compute/indicators/momentum.mojo`

**Task 4.3: Batch Processing API** [Issue #15]
- Assignee: SR. Dev Claude
- Parallel computation for multiple symbols
- Deliverable: Enhanced socket API

**Task 5.1: Backtest Engine (Mojo)** [Issue #17]
- Assignee: SR. Dev Claude
- Vectorized backtesting in Mojo
- Target: 60x faster than Python
- Deliverable: `mojo-compute/backtesting/engine.mojo`

**Task 5.2: Performance Metrics** [Issue #18]
- Assignee: SR. Dev Codex
- Sharpe, Sortino, max drawdown calculations
- Deliverable: `mojo-compute/backtesting/metrics.mojo`

---

## Phase 2: Signal Service Migration (Months 4-6)

### Month 4: Signal Service Foundation (Mojo)

**Task 6.1: Signal-Service Structure** [NEW]
- Assignee: Claude Master
- Create signal-service directory structure
- Socket server setup
- Deliverable: `signal-service/` structure

**Task 6.2: Alert Generation Logic (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- Port alert generation from Python to Mojo
- Pattern matching algorithms
- Scoring logic (impact, confidence)
- Deliverable: `signal-service/alerts/generator.mojo`

**Task 6.3: Database Queries (Mojo)** [NEW]
- Assignee: SR. Dev Codex
- PostgreSQL query layer using libpq FFI
- Alert CRUD operations
- Filtering and sorting
- Deliverable: `signal-service/db/queries.mojo`

**Task 6.4: SSE Event Streaming (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- Implement Server-Sent Events in Mojo
- Publish/Subscribe pattern
- Event bus
- Deliverable: `signal-service/streaming/sse.mojo`

---

### Month 5: Signal Service Advanced Features

**Task 6.5: Pattern Matching Engine (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- Technical pattern recognition
- Fundamental filters (P/E, ROE, etc.)
- Combined signals
- Target: 500x faster than Python
- Deliverable: `signal-service/patterns/engine.mojo`

**Task 6.6: Real-time Filtering (Mojo)** [NEW]
- Assignee: SR. Dev Codex
- SIMD-optimized filtering
- Symbol search
- Multi-criteria queries
- Deliverable: `signal-service/filtering/engine.mojo`

**Task 6.7: Caching Layer (Mojo)** [NEW]
- Assignee: SR. Dev Codex
- In-memory cache (LRU)
- Redis FFI (optional)
- Cache invalidation
- Deliverable: `signal-service/cache/memory.mojo`

---

### Month 6: Integration & Testing

**Task 6.8: Core-API Integration** [NEW]
- Assignee: Claude Master
- Update core-api to route signal requests to Mojo service
- SSE proxying
- Error handling
- Deliverable: Updated `core-api/`

**Task 6.9: Dashboard Integration** [NEW]
- Assignee: Claude Master
- Update dashboard to use new signal endpoints
- Test SSE streaming
- Deliverable: Updated `dashboard-app/`

**Task 6.10: Performance Testing** [NEW]
- Assignee: All
- Load testing signal-service
- Stress testing (1000+ concurrent SSE connections)
- Deliverable: Performance report

---

## Phase 3: News-NLP Migration (Months 7-10)

### Month 7: News-NLP Foundation (Mojo)

**Task 7.1: HTTP Client (Mojo)** [NEW]
- Assignee: SR. Dev Codex
- FFI to libcurl
- HTTP GET requests
- User-agent handling
- Retry logic
- Deliverable: `shared/http/client.mojo`

**Task 7.2: XML/RSS Parser (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- FFI to libxml2 OR manual XML parser
- RSS 2.0 format support
- Target: 100x faster than feedparser
- Deliverable: `news-nlp/parsing/rss.mojo`

**Task 7.3: Article Extraction (Mojo)** [NEW]
- Assignee: SR. Dev Codex
- Extract title, summary, link, date from RSS
- Text cleaning
- Deliverable: `news-nlp/extraction/article.mojo`

---

### Month 8: NLP Implementation (Mojo)

**Task 7.4: Sentiment Analysis (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- **Option A**: Port FinBERT to Mojo (very hard)
- **Option B**: Rule-based sentiment (keyword lists)
- **Option C**: FFI to libtorch + FinBERT model
- Target: 1000x faster than transformers
- Deliverable: `news-nlp/nlp/sentiment.mojo`

**Task 7.5: Named Entity Recognition (Mojo)** [NEW]
- Assignee: SR. Dev Codex
- Extract stock symbols (TCS, INFY, etc.)
- Extract sectors (Banking, IT, Pharma)
- Regex + dictionary matching
- Deliverable: `news-nlp/nlp/ner.mojo`

**Task 7.6: Direction Classification (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- Classify as bullish/bearish/neutral
- Keyword-based OR ML model
- Deliverable: `news-nlp/nlp/direction.mojo`

---

### Month 9: Database & Scheduling

**Task 7.7: Database Writer (Mojo)** [NEW]
- Assignee: SR. Dev Codex
- Batch insert articles to PostgreSQL
- Insert entities, sentiments
- Transaction management
- Deliverable: `news-nlp/db/writer.mojo`

**Task 7.8: RSS Scheduler (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- Cron-like scheduler in Mojo
- Run RSS ingestion every 10 minutes
- Error handling and logging
- Deliverable: `news-nlp/scheduler/cron.mojo`

**Task 7.9: Logging Framework (Mojo)** [NEW]
- Assignee: SR. Dev Codex
- Structured logging in Mojo
- Log levels (DEBUG, INFO, WARN, ERROR)
- File rotation
- Deliverable: `shared/logging/logger.mojo`

---

### Month 10: Integration & Migration

**Task 7.10: News-NLP Socket Server** [NEW]
- Assignee: SR. Dev Claude
- Expose news-nlp via Unix socket
- API: ingest_rss, analyze_sentiment, get_articles
- Deliverable: `news-nlp/api/server.mojo`

**Task 7.11: Core-API Integration** [NEW]
- Assignee: Claude Master
- Route news requests to Mojo news-nlp
- Update dashboard
- Deliverable: Updated `core-api/`

**Task 7.12: Migrate Data** [NEW]
- Assignee: All
- Switch from Python news-nlp to Mojo news-nlp
- Monitor for errors
- Rollback plan
- Deliverable: Successful migration

---

## Phase 4: ML Inference (Months 11-12)

**Task 8.1: Matrix Operations** [Issue #20]
- Assignee: SR. Dev Codex
- BLAS-like operations in Mojo (SIMD)
- Matrix multiplication, transpose
- Deliverable: `mojo-compute/ml/matrix.mojo`

**Task 8.2: ONNX Model Loading** [NEW]
- Assignee: SR. Dev Claude
- Load sklearn/LightGBM models via ONNX
- FFI to onnxruntime OR manual implementation
- Deliverable: `mojo-compute/ml/onnx_loader.mojo`

**Task 8.3: Model Inference** [Issue #21]
- Assignee: SR. Dev Claude
- Forward pass for LightGBM/sklearn
- Batch prediction
- Target: 1000x faster than sklearn
- Deliverable: `mojo-compute/ml/inference.mojo`

**Task 8.4: Feature Engineering** [Issue #22]
- Assignee: SR. Dev Codex
- Extract 200+ features from market data
- Technical + fundamental + news features
- Deliverable: `mojo-compute/ml/features.mojo`

---

## Phase 5: Production Deployment (Months 13-15)

### Month 13: Docker & CI/CD

**Task 9.1: Dockerfiles for All Services** [NEW]
- Assignee: SR. Dev Codex
- Multi-stage builds for Mojo services
- Minimize image size
- Health checks
- Deliverable: `Dockerfile` for each service

**Task 9.2: Docker Compose** [Issue #24]
- Assignee: SR. Dev Codex
- Orchestrate all services
- Environment variables
- Volume mounts (Unix sockets)
- Deliverable: `docker-compose.yml`

**Task 9.3: GitHub Actions for Mojo** [NEW]
- Assignee: Claude Master
- CI/CD pipeline for Mojo code
- Build, test, benchmark
- Auto-merge
- Deliverable: `.github/workflows/mojo.yml`

---

### Month 14: Monitoring & Observability

**Task 9.4: Prometheus Metrics (Mojo)** [NEW]
- Assignee: SR. Dev Claude
- Expose metrics from Mojo services
- Request count, latency, errors
- Deliverable: `shared/metrics/prometheus.mojo`

**Task 9.5: Grafana Dashboards** [Issue #26]
- Assignee: Claude Master
- Visualize all service metrics
- Alert dashboards
- Performance tracking
- Deliverable: Grafana JSON configs

**Task 9.6: Distributed Tracing** [NEW]
- Assignee: SR. Dev Codex
- OpenTelemetry integration
- Trace requests across services
- Deliverable: Tracing setup

---

### Month 15: Production Hardening

**Task 9.7: Load Testing** [NEW]
- Assignee: All
- Test system at 10,000 req/sec
- Identify bottlenecks
- Optimize
- Deliverable: Load test report

**Task 9.8: Security Audit** [NEW]
- Assignee: Claude Master
- Review authentication, authorization
- Input validation
- SQL injection prevention
- Deliverable: Security audit report

**Task 9.9: Documentation** [NEW]
- Assignee: All
- Update all docs for Mojo services
- Deployment guide
- API documentation
- Deliverable: Complete docs

---

## Phase 6: Optimization & Scale (Months 16-18)

**Task 10.1: GPU Acceleration** [NEW]
- Assignee: SR. Dev Claude
- Use Mojo GPU support for ML inference
- CUDA kernels for indicators
- Deliverable: GPU-accelerated modules

**Task 10.2: Horizontal Scaling** [NEW]
- Assignee: Claude Master
- Load balancing across Mojo service instances
- Service discovery
- Deliverable: Scalable architecture

**Task 10.3: Advanced Caching** [NEW]
- Assignee: SR. Dev Codex
- Redis integration for all services
- Cache warming strategies
- Deliverable: Multi-tier caching

**Task 10.4: Performance Tuning** [NEW]
- Assignee: All
- Profile and optimize hot paths
- SIMD improvements
- Memory optimization
- Deliverable: 100x overall speedup achieved

---

## Shared Infrastructure Tasks (Ongoing)

### Shared Mojo Libraries

**Task S1: Shared PostgreSQL Module**
- FFI bindings to libpq
- Connection pooling
- Prepared statements
- Used by: news-nlp, signal-service, mojo-compute
- Deliverable: `shared/postgres/`

**Task S2: Shared HTTP Module**
- FFI to libcurl
- HTTP client
- Used by: news-nlp
- Deliverable: `shared/http/`

**Task S3: Shared JSON Module**
- Fast JSON parser (FFI or manual)
- JSON serializer
- Used by: All services
- Deliverable: `shared/json/`

**Task S4: Shared Logging Module**
- Structured logging
- Log levels
- File rotation
- Used by: All services
- Deliverable: `shared/logging/`

**Task S5: Shared Socket Module**
- Unix socket server/client
- TCP socket support
- Used by: All services
- Deliverable: `shared/sockets/`

---

## Repository Structure (Final)

```
trading-chitti/
├── core-api/                    # FastAPI Python gateway
│   ├── app.py                   # Main FastAPI app
│   ├── routes/                  # Route definitions
│   ├── schemas.py               # Pydantic models
│   └── clients/                 # Socket clients to Mojo services
│
├── news-nlp/                    # MOJO service
│   ├── main.mojo                # Entry point
│   ├── api/server.mojo          # Socket server
│   ├── parsing/rss.mojo         # RSS parser
│   ├── nlp/                     # NLP modules
│   ├── db/writer.mojo           # Database ops
│   └── scheduler/cron.mojo      # Job scheduler
│
├── signal-service/              # MOJO service
│   ├── main.mojo
│   ├── api/server.mojo
│   ├── alerts/generator.mojo
│   ├── patterns/engine.mojo
│   ├── streaming/sse.mojo
│   └── db/queries.mojo
│
├── mojo-compute/                # MOJO service
│   ├── main.mojo
│   ├── api/server.mojo
│   ├── indicators/              # 100+ indicators
│   ├── backtesting/engine.mojo
│   ├── ml/inference.mojo
│   └── ml/features.mojo
│
├── shared/                      # Shared Mojo libraries
│   ├── postgres/libpq_ffi.mojo  # PostgreSQL FFI
│   ├── http/client.mojo         # HTTP client
│   ├── json/parser.mojo         # JSON parser
│   ├── logging/logger.mojo      # Logging
│   └── sockets/server.mojo      # Socket utilities
│
├── dashboard-app/               # React (UNCHANGED)
│   └── ...
│
├── tests/
│   ├── unit/                    # Unit tests for each service
│   └── integration/             # E2E tests
│
└── docker-compose.yml           # Orchestration
```

---

## Success Metrics

### Performance Targets

| Metric | Python Baseline | Mojo Target | Achieved? |
|--------|-----------------|-------------|-----------|
| **RSS Parsing** | 500ms/feed | 5ms/feed (100x) | ⏳ |
| **Sentiment Analysis** | 200ms/article | 0.2ms (1000x) | ⏳ |
| **Alert Generation** | 50ms/symbol | 0.5ms (100x) | ⏳ |
| **SMA (10K points)** | 80ms | 0.8ms (100x) | ⏳ |
| **Backtesting (1yr)** | 3000ms | 50ms (60x) | ⏳ |
| **ML Inference** | 10ms | 0.01ms (1000x) | ⏳ |
| **System Throughput** | 100 req/s | 10,000 req/s | ⏳ |

### Quality Targets

- [ ] 100% test coverage for Mojo code
- [ ] All integration tests passing
- [ ] API latency <10ms (p95)
- [ ] Memory usage <500MB per service
- [ ] Zero data corruption bugs
- [ ] 99.9% uptime
- [ ] Docker build <5 minutes
- [ ] Service startup <10 seconds

---

## Risk Mitigation

### Risk 1: Mojo Ecosystem Immaturity
**Mitigation**: Build C FFI wrappers for critical libraries (libpq, libcurl, libxml2)
**Contingency**: Keep Python services running in parallel during migration

### Risk 2: Performance Not Meeting Targets
**Mitigation**: Early benchmarking (Phase 1), profile and optimize
**Contingency**: Hybrid approach - keep some parts in Python

### Risk 3: Team Coordination
**Mitigation**: Clear task assignments, daily standups (async via GitHub)
**Contingency**: Claude Master resolves conflicts

### Risk 4: C FFI Complexity
**Mitigation**: Create well-documented shared libraries
**Contingency**: Use subprocess/socket to Python libraries if FFI fails

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1** | Months 1-3 | Mojo compute service, socket API, core-api gateway |
| **Phase 2** | Months 4-6 | Signal-service in Mojo |
| **Phase 3** | Months 7-10 | News-NLP in Mojo |
| **Phase 4** | Months 11-12 | ML inference in Mojo |
| **Phase 5** | Months 13-15 | Production deployment, monitoring |
| **Phase 6** | Months 16-18 | GPU acceleration, scaling, optimization |

**Total**: 18 months to complete all-Mojo backend

---

## Next Immediate Steps

1. **SR. Dev Claude**: Complete benchmarks (Issue #2) ✅
2. **SR. Dev Codex**: Install Mojo SDK (Issue #1) - **URGENT**
3. **Claude Master**: Review benchmark PR, plan shared libraries
4. **All**: Begin Phase 1 Mojo indicators once SDK is installed

---

**Let's build the fastest trading system in the world! 🚀**
