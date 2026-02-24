# Docker Compose Setup для Memory-MCP

## Быстрый старт

### 1. Запуск PostgreSQL + Redis

```bash
# Запустить все сервисы
docker-compose up -d

# Проверить статус
docker-compose ps

# Посмотреть логи
docker-compose logs -f
```

### 2. Настройка приложения

Готовая конфигурация уже в `.env.docker` - все пароли настроены!

Просто скопируйте:

```bash
cp .env.docker .env
```

Или запустите с явным указанием файла:

```bash
export $(cat .env.docker | xargs) && python -m mcp_server.server
```

### 3. Проверка подключения

```bash
# PostgreSQL
docker exec -it ai_postgres psql -U memory_user -d memory

# Redis
docker exec -it ai_redis redis-cli ping

# Проверка health
docker-compose ps
```

## Сервисы

| Сервис | Контейнер | Порт | Описание |
|--------|-----------|------|----------|
| PostgreSQL | ai_postgres | 5442 | Основная БД |
| Redis | ai_redis | 6379 | Кэш + rate limiting |
| pgAdmin | ai_pgadmin | 5050 | Веб-интерфейс БД (опционально) |

## Управление

```bash
# Остановить все сервисы
docker-compose down

# Остановить с удалением данных (WARNING!)
docker-compose down -v

# Перезапустить сервисы
docker-compose restart

# Пересоздать контейнеры
docker-compose up -d --force-recreate

# Запустить только PostgreSQL
docker-compose up -d ai_postgres

# Запустить с pgAdmin
docker-compose --profile tools up -d
```

## Мониторинг

```bash
# Статистика использования ресурсов
docker stats ai_postgres ai_redis

# Логи PostgreSQL
docker-compose logs ai_postgres

# Логи Redis
docker-compose logs ai_redis

# Проверка здоровья
docker-compose ps
```

## Бэкапы

### PostgreSQL

```bash
# Создать дамп
docker exec ai_postgres pg_dump -U memory_user memory > backup_$(date +%Y%m%d).sql

# Восстановить из дампа
docker exec -i ai_postgres psql -U memory_user memory < backup_20240101.sql
```

### Redis

```bash
# Копировать RDB файл
docker cp ai_redis:/data/dump.rdb ./redis_backup_$(date +%Y%m%d).rdb
```

## Оптимизация

### PostgreSQL tuned для Memory-MCP:

- `shared_buffers = 256MB` - 25% от RAM контейнера
- `effective_cache_size = 768MB` - 75% от RAM
- `work_mem = 4MB` - для сложных запросов
- `maintenance_work_mem = 64MB` - для VACUUM/CREATE INDEX
- `autovacuum` настроен для частых UPDATE/DELETE

### Redis оптимизация:

- `maxmemory = 128mb` - лимит памяти
- `maxmemory-policy = allkeys-lru` - LRU eviction
- `appendonly = yes` - персистентность
- `appendfsync = everysec` - баланс производительности/надёжности

## Безопасность

1. **Смените пароли** в `docker-compose.yml`:
   - `POSTGRES_PASSWORD`
   - `PGADMIN_DEFAULT_PASSWORD`

2. **Не публикуйте порты** в production:
   ```yaml
   # Уберите секцию ports:
   # ports:
   #   - "5442:5442"
   ```

3. **Используйте secrets** для чувствительных данных:
   ```yaml
   secrets:
     - db_password
   
   services:
     ai_postgres:
       environment:
         POSTGRES_PASSWORD_FILE: /run/secrets/db_password
   ```

## Troubleshooting

### PostgreSQL не запускается

```bash
# Проверить логи
docker-compose logs ai_postgres

# Проверить volume
docker volume ls | grep memory_postgres_data
```

### Redis не подключается

```bash
# Проверить доступность
docker exec ai_redis redis-cli ping

# Проверить логи
docker-compose logs ai_redis
```

### Проблемы с памятью

```bash
# Очистить кэш Redis
docker exec ai_redis redis-cli FLUSHALL

# Vacuum PostgreSQL
docker exec ai_postgres psql -U memory_user -d memory -c "VACUUM FULL;"
```

## Production Deployment

Для production используйте:

1. **Docker Swarm/Kubernetes** для оркестрации
2. **Отдельные volume** для данных
3. **Secrets management** (Docker Secrets, Vault)
4. **Monitoring** (Prometheus + Grafana)
5. **Backup strategy** (регулярные бэкапы + репликация)
