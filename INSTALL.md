# Install and setup guide for Memory-MCP

## Requirements

- Python 3.11+
- Docker + Docker Compose (for PostgreSQL/Redis mode)
- Ollama (optional, for local LLM usage)

---

## Option 1: SQLite (quick start)

### 1) Install dependencies

```bash
pip install -e .
```

### 2) Configure environment

```bash
cp .env.local .env
```

### 3) Run server

```bash
python -m mcp_server.server
```

Done. Data is stored in `./memory.db`.

---

## Option 2: PostgreSQL + Redis (production-style)

### 1) Start infrastructure

```bash
# start PostgreSQL + Redis
./docker-compose.sh start

# verify health
./docker-compose.sh health
```

### 2) Configure environment

```bash
cp .env.docker .env
```

### 3) Run application

```bash
pip install -e .
python -m mcp_server.server
```

### 4) Daily operations

```bash
# stop
./docker-compose.sh stop

# restart
./docker-compose.sh restart

# backup
./docker-compose.sh backup

# logs
./docker-compose.sh logs ai_postgres
```

See `docker/README.md` for full Docker operations.

---

## Option 3: Hybrid mode (SQLite + Redis)

Use SQLite as primary storage and Redis for cache/rate limiting:

```bash
# start only Redis
docker compose up -d ai_redis

# set in .env
OMNIMIND_REDIS_ENABLED=true
OMNIMIND_REDIS_HOST=localhost
OMNIMIND_DB_PATH=./memory.db
```

---

## Verify installation

### Run tests

```bash
pip install -e .[dev]
pytest tests -v
```

### Lint + type checks

```bash
ruff check core/ mcp_server/
mypy core/security/ core/search/ core/llm/
```

### Health check

```bash
python -c "from core.memory import MemoryStore; import asyncio; print(asyncio.run(MemoryStore().health()))"
```

---

## Configuration

### Core settings

| Parameter | Default value | Description |
|-----------|---------------|-------------|
| `OMNIMIND_DB_TYPE` | `sqlite` | Database mode: `sqlite` or `postgres` |
| `OMNIMIND_DB_PATH` | `./memory.db` | SQLite file path |
| `OMNIMIND_REDIS_ENABLED` | `false` | Enable Redis cache/rate limiting |
| `OMNIMIND_LLM_PROVIDER` | `ollama` | LLM provider |
| `OMNIMIND_EMBEDDINGS_PROVIDER` | `fastembed` | Embeddings provider |

### LLM provider examples

#### Ollama (local)

```bash
OMNIMIND_LLM_PROVIDER=ollama
OMNIMIND_LLM_BASE_URL=http://localhost:11434
OMNIMIND_LLM_MODEL=llama3.2
```

#### DeepSeek (cloud)

```bash
OMNIMIND_LLM_PROVIDER=deepseek
OMNIMIND_LLM_API_KEY=sk-xxx
OMNIMIND_LLM_MODEL=deepseek-chat
```

#### OpenAI (cloud)

```bash
OMNIMIND_LLM_PROVIDER=openai
OMNIMIND_LLM_API_KEY=sk-xxx
OMNIMIND_LLM_MODEL=gpt-4
```

---

## Troubleshooting

### SQLite: "database is locked"

```bash
# remove WAL files
rm memory.db-wal memory.db-shm

# alternatively disable WAL in sqlite shell
PRAGMA journal_mode=DELETE;
```

### PostgreSQL: connection refused

```bash
docker ps | grep ai_postgres
docker compose logs ai_postgres
docker exec ai_postgres pg_isready -U memory_user -d memory
```

### Redis: connection issues

```bash
docker ps | grep ai_redis
docker exec ai_redis redis-cli ping
```

### Ollama: model not found

```bash
ollama pull llama3.2
ollama list
```

---

## Production deployment checklist

### 1) Security

- Replace default passwords in `docker-compose.yml` and `.env`.
- Store API keys in secrets manager or Docker secrets.
- Disable public ports unless explicitly required.

### 2) Scalability

- Use Docker Swarm/Kubernetes for orchestration.
- Configure PostgreSQL replication if needed.
- Use Redis Cluster under high load.

### 3) Monitoring

- Enable metrics (`OMNIMIND_METRICS_ENABLED=true`).
- Use Prometheus + Grafana.
- Use pgAdmin or DB dashboards for PostgreSQL observability.

### 4) Backups

```bash
# PostgreSQL
./docker-compose.sh backup

# Redis
docker cp ai_redis:/data/dump.rdb ./backup.rdb

# SQLite
cp memory.db memory.db.backup
```

---

## Additional resources

- Docker setup: `docker/README.md`
- Usage examples: `examples/`
- Environment presets: `ENV_CONFIGS.md`
