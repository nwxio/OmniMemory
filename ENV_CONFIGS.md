# Environment presets (`.env`)

This repository ships with ready-to-use environment presets for common deployment modes.

## Available presets

| File | Intended use | DB | DB flags | LLM | Redis |
|------|--------------|----|----------|-----|-------|
| `.env` | active runtime config | depends on file content | depends on file content | depends on file content | depends on file content |
| `.env.local` | local development | SQLite | `POSTGRES=false`, `SQLITE=true` | Ollama | no |
| `.env.docker` | Docker stack | PostgreSQL | `POSTGRES=true`, `SQLITE=false` | Ollama | yes |
| `.env.production` | production baseline | PostgreSQL | `POSTGRES=true`, `SQLITE=false` | OpenAI | yes |
| `.env.budget` | low-cost mode | SQLite | `POSTGRES=false`, `SQLITE=true` | DeepSeek | no |

## Quick setup recipes

### 1) Local development (fastest start)

```bash
cp .env.local .env
python -m mcp_server.server
```

Includes:

- SQLite (`memory.db`)
- Ollama (local)
- FastEmbed (local)

### 2) Docker infrastructure (PostgreSQL + Redis)

```bash
./docker-compose.sh start
cp .env.docker .env
python -m mcp_server.server
```

Includes:

- PostgreSQL (`ai_postgres`)
- Redis (`ai_redis`)
- Ollama (external/local host install)

### 3) Production-style preset (OpenAI)

```bash
cp .env.production .env

# then update secrets and hostnames in .env
python -m mcp_server.server
```

### 4) Budget preset (DeepSeek)

```bash
cp .env.budget .env

# then set your API key(s)
python -m mcp_server.server
```

## Switching profiles

```bash
# local
cp .env.local .env

# docker
cp .env.docker .env

# production
cp .env.production .env

# budget
cp .env.budget .env
```

Check active profile quickly:

```bash
grep "^OMNIMIND_DB_TYPE" .env
grep "^OMNIMIND_POSTGRES_ENABLED" .env
grep "^OMNIMIND_SQLITE_ENABLED" .env
grep "^OMNIMIND_LLM_PROVIDER" .env
grep "^OMNIMIND_REDIS_ENABLED" .env
```

DB toggle priority:

- Preferred: `OMNIMIND_POSTGRES_ENABLED` + `OMNIMIND_SQLITE_ENABLED`
- Backward compatibility: `OMNIMIND_DB_TYPE` is used when both toggles are omitted
- Ambiguous flags (`true/true` or `false/false`) fall back to `OMNIMIND_DB_TYPE`
- Optional strict mode: set `OMNIMIND_DB_STRICT_BACKEND=true` to fail startup on requested/effective mismatch.
- PostgreSQL backend requires a PostgreSQL Python driver (`psycopg2`/`psycopg`).
- Check active backend via `memory_health` (`db_backend.effective`).

## Critical variables to review before deployment

For production (`.env.production`):

1. `OMNIMIND_POSTGRES_PASSWORD`
2. `OMNIMIND_LLM_API_KEY`
3. `OPENAI_API_KEY` (if used)
4. `OMNIMIND_REDIS_PASSWORD`
5. DB/Redis hostnames and ports

For docker preset (`.env.docker`):

- Replace default passwords if exposed beyond local development.

## Connectivity checks

### PostgreSQL

```bash
# docker
docker exec -it ai_postgres psql -U memory_user -d memory

# remote
psql -h your-db.example.com -U memory_user -d memory_prod
```

### Redis

```bash
# docker
docker exec -it ai_redis redis-cli ping

# remote
redis-cli -h your-redis.example.com -a YourPassword ping
```

### LLM providers

```bash
# Ollama
curl http://localhost:11434/api/tags

# DeepSeek
curl https://api.deepseek.com/v1/models -H "Authorization: Bearer sk-YOUR_KEY"

# OpenAI
curl https://api.openai.com/v1/models -H "Authorization: Bearer sk-YOUR_KEY"
```

## Troubleshooting

### PostgreSQL: connection refused

```bash
docker ps | grep ai_postgres
docker compose logs ai_postgres
docker exec ai_postgres pg_isready -U memory_user -d memory
```

### Redis: connection refused

```bash
docker ps | grep ai_redis
docker exec ai_redis redis-cli ping
```

### Ollama unavailable

```bash
ollama list
ollama pull llama3.2

# macOS
brew services restart ollama

# Linux
sudo systemctl restart ollama
```

## Related docs

- Docker deployment: `docker/README.md`
- Installation notes: `INSTALL.md`
- Main project overview: `README.md`
