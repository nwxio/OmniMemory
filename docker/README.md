# Docker setup for Memory-MCP

This guide explains how to run Memory-MCP with Docker infrastructure
(`PostgreSQL + Redis`, optional `pgAdmin`).

## Quick start

### 1) Start infrastructure

```bash
docker compose up -d

# or use helper script
./docker-compose.sh start
```

### 2) Apply Docker environment preset

```bash
cp .env.docker .env
```

### 3) Start the MCP server

```bash
python -m mcp_server.server
```

### 4) Verify services

```bash
docker compose ps
docker exec -it ai_postgres psql -U memory_user -d memory
docker exec -it ai_redis redis-cli ping
```

## Services

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| PostgreSQL | `ai_postgres` | `5442` | primary database |
| Redis | `ai_redis` | `6379` | cache + rate limiting |
| pgAdmin (optional) | `ai_pgadmin` | `5050` | PostgreSQL web UI |

Start with pgAdmin profile:

```bash
docker compose --profile tools up -d
```

## Day-to-day commands

```bash
# stop
docker compose down

# stop + remove volumes (destructive)
docker compose down -v

# restart
docker compose restart

# recreate containers
docker compose up -d --force-recreate

# logs
docker compose logs -f
docker compose logs -f ai_postgres
docker compose logs -f ai_redis

# resource usage
docker stats ai_postgres ai_redis
```

## Backups

### PostgreSQL backup/restore

```bash
# backup
docker exec ai_postgres pg_dump -U memory_user memory > backup_$(date +%Y%m%d).sql

# restore
docker exec -i ai_postgres psql -U memory_user memory < backup_20240101.sql
```

### Redis backup

```bash
docker cp ai_redis:/data/dump.rdb ./redis_backup_$(date +%Y%m%d).rdb
```

## Tuned defaults in this repo

PostgreSQL (memory-focused profile):

- `shared_buffers=256MB`
- `effective_cache_size=768MB`
- `work_mem=4MB`
- `maintenance_work_mem=64MB`
- WAL durability and logging tuned in `docker-compose.yml`

Redis:

- `maxmemory=128mb`
- `maxmemory-policy=allkeys-lru`
- `appendonly=yes`
- `appendfsync=everysec`

## Security checklist

Before public or production deployment:

1. Change default passwords in `docker-compose.yml` and `.env`.
2. Do not expose DB/cache ports publicly unless required.
3. Use Docker Secrets (or Vault/KMS) for credentials.
4. Restrict network access with firewall/security groups.
5. Enable regular encrypted backups.

## Troubleshooting

### PostgreSQL fails to start

```bash
docker compose logs ai_postgres
docker volume ls | grep postgres
docker exec ai_postgres pg_isready -U memory_user -d memory
```

### Redis does not respond

```bash
docker compose logs ai_redis
docker exec ai_redis redis-cli ping
```

### High memory usage / storage growth

```bash
# clear Redis cache (destructive)
docker exec ai_redis redis-cli FLUSHALL

# vacuum PostgreSQL
docker exec ai_postgres psql -U memory_user -d memory -c "VACUUM FULL;"
```

## Production notes

For production use:

1. Run behind orchestration (`Docker Swarm` or `Kubernetes`).
2. Use managed secrets and encrypted volumes.
3. Add monitoring (`Prometheus/Grafana`) and alerting.
4. Add backup retention and restore drills.
5. Validate rolling upgrade and rollback procedure.
