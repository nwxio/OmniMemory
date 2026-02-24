# Установка и настройка Memory-MCP

## Требования

- Python 3.11+
- Docker + Docker Compose (для PostgreSQL/Redis варианта)
- Ollama (опционально, для локальной LLM)

---

## Вариант 1: SQLite (быстрый старт)

### 1. Установка зависимостей

```bash
pip install -e .
```

### 2. Настройка окружения

```bash
cp .env.example .env
```

### 3. Запуск

```bash
python -m mcp_server.server
```

**Готово!** Данные хранятся в `./memory.db`.

---

## Вариант 2: PostgreSQL + Redis (production)

### 1. Запуск инфраструктуры

```bash
# Запустить PostgreSQL + Redis
./docker-compose.sh start

# Проверить статус
./docker-compose.sh health
```

### 2. Настройка окружения

```bash
# Использовать готовый конфиг для Docker
cp .env.docker .env
```

### 3. Запуск приложения

```bash
pip install -e .
python -m mcp_server.server
```

### 4. Управление

```bash
# Остановить
./docker-compose.sh stop

# Перезапустить
./docker-compose.sh restart

# Бэкап
./docker-compose.sh backup

# Логи
./docker-compose.sh logs ai_postgres
```

См. [`docker/README.md`](docker/README.md) для деталей.

---

## Вариант 3: Гибридный (SQLite + Redis)

Используйте SQLite для данных, Redis для кэша:

```bash
# Запустить только Redis
docker-compose up -d ai_redis

# Настроить .env
OMNIMIND_REDIS_ENABLED=true
OMNIMIND_REDIS_HOST=localhost
OMNIMIND_DB_PATH=./memory.db
```

---

## Проверка установки

### Тесты

```bash
pip install -e .[dev]
pytest tests -v
```

### Lint + Type

```bash
ruff check core/ mcp_server/
mypy core/security/ core/search/ core/llm/
```

### Health check

```bash
python -c "from core.memory import MemoryStore; import asyncio; print(asyncio.run(MemoryStore().health()))"
```

---

## Конфигурация

### Основные параметры

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `OMNIMIND_DB_TYPE` | `sqlite` | Тип БД: `sqlite` или `postgres` |
| `OMNIMIND_DB_PATH` | `./memory.db` | Путь к SQLite файлу |
| `OMNIMIND_REDIS_ENABLED` | `false` | Включить Redis кэш |
| `OMNIMIND_LLM_PROVIDER` | `ollama` | LLM провайдер |
| `OMNIMIND_EMBEDDINGS_PROVIDER` | `fastembed` | Провайдер эмбеддингов |

### LLM провайдеры

#### Ollama (локально)

```bash
OMNIMIND_LLM_PROVIDER=ollama
OMNIMIND_LLM_BASE_URL=http://localhost:11434
OMNIMIND_LLM_MODEL=llama3.2
```

#### DeepSeek (облако)

```bash
OMNIMIND_LLM_PROVIDER=deepseek
OMNIMIND_LLM_API_KEY=sk-xxx
OMNIMIND_LLM_MODEL=deepseek-chat
```

#### OpenAI (облако)

```bash
OMNIMIND_LLM_PROVIDER=openai
OMNIMIND_LLM_API_KEY=sk-xxx
OMNIMIND_LLM_MODEL=gpt-4
```

---

## Troubleshooting

### SQLite: "database is locked"

```bash
# Удалить WAL файлы
rm memory.db-wal memory.db-shm

# Или отключить WAL
PRAGMA journal_mode=DELETE;
```

### PostgreSQL: connection refused

```bash
# Проверить что контейнер запущен
docker ps | grep ai_postgres

# Проверить логи
docker-compose logs ai_postgres

# Проверить подключение
docker exec ai_postgres pg_isready -U memory_user -d memory
```

### Redis: не подключается

```bash
# Проверить что контейнер запущен
docker ps | grep ai_redis

# Проверить ping
docker exec ai_redis redis-cli ping
```

### Ollama: model not found

```bash
# Загрузить модель
ollama pull llama3.2

# Проверить доступные модели
ollama list
```

---

## Production Deployment

### 1. Безопасность

- Смените пароли в `docker-compose.yml`
- Используйте secrets для API ключей
- Отключите публичные порты (если не нужно)

### 2. Масштабирование

- Используйте Docker Swarm/Kubernetes
- Настройте репликацию PostgreSQL
- Используйте Redis Cluster для высокого нагрузки

### 3. Мониторинг

- Включите метрики: `OMNIMIND_METRICS_ENABLED=true`
- Настройте Prometheus + Grafana
- Используйте pgAdmin для PostgreSQL

### 4. Бэкапы

```bash
# PostgreSQL
./docker-compose.sh backup

# Redis
docker cp ai_redis:/data/dump.rdb ./backup.rdb

# SQLite
cp memory.db memory.db.backup
```

---

## Дополнительные ресурсы

- [Docker Compose настройка](docker/README.md)
- [Примеры использования](examples/)
- [Конфигурация .env](.env.example)
