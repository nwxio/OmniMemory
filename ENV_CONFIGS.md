# Готовые конфигурации .env

## Доступные варианты

| Файл | Назначение | БД | LLM | Redis |
|------|-----------|-----|-----|-------|
| `.env` | **Активная** | SQLite | Ollama | ❌ |
| `.env.local` | Локальная разработка | SQLite | Ollama | ❌ |
| `.env.docker` | Docker (PostgreSQL+Redis) | PostgreSQL | Ollama | ✅ |
| `.env.production` | Production | PostgreSQL | OpenAI GPT-4 | ✅ |
| `.env.budget` | Бюджетный вариант | SQLite | DeepSeek | ❌ |

---

## Быстрый старт

### Вариант 1: Локальная разработка

```bash
cp .env.local .env
python -m mcp_server.server
```

**Что используется:**
- ✅ SQLite (файл `memory.db`)
- ✅ Ollama (локально, бесплатно)
- ✅ FastEmbed (локально, бесплатно)

---

### Вариант 2: Docker (PostgreSQL + Redis)

```bash
# Запустить инфраструктуру
./docker-compose.sh start

# Настроить окружение
cp .env.docker .env

# Запустить приложение
python -m mcp_server.server
```

**Что используется:**
- ✅ PostgreSQL (контейнер `ai_postgres`)
- ✅ Redis (контейнер `ai_redis`)
- ✅ Ollama (локально)

**Пароли по умолчанию:**
- PostgreSQL: `SecureP@ssw0rd_2024!MemoryDB`
- Redis: без пароля (локально)

---

### Вариант 3: Production (OpenAI)

```bash
# Скопировать production конфиг
cp .env.production .env

# Заменить API ключи на свои
# Отредактировать .env:
#   - OMNIMIND_LLM_API_KEY
#   - OPENAI_API_KEY
#   - PostgreSQL credentials

# Запустить
python -m mcp_server.server
```

**Что используется:**
- ✅ PostgreSQL (production сервер)
- ✅ Redis Cluster
- ✅ OpenAI GPT-4
- ✅ OpenAI Embeddings

---

### Вариант 4: Бюджетный (DeepSeek)

```bash
cp .env.budget .env

# Заменить API ключ DeepSeek
# Отредактировать .env:
#   - OMNIMIND_LLM_API_KEY=sk-YOUR_KEY

python -m mcp_server.server
```

**Что используется:**
- ✅ SQLite (бесплатно)
- ✅ DeepSeek (~$0.14/1M tokens)
- ✅ FastEmbed (бесплатно)

**Экономия:** в 10-20 раз дешевле OpenAI

---

## Смена конфигурации

### Переключение между режимами

```bash
# На локальную
cp .env.local .env

# На Docker
cp .env.docker .env

# На production
cp .env.production .env

# На бюджетную
cp .env.budget .env
```

### Проверка активной конфигурации

```bash
# Показать текущие настройки
grep "^OMNIMIND_DB_TYPE" .env
grep "^OMNIMIND_LLM_PROVIDER" .env
grep "^OMNIMIND_REDIS_ENABLED" .env
```

---

## Безопасность

### Production (.env.production)

**Обязательно замените:**

1. **PostgreSQL пароль:**
   ```
   OMNIMIND_POSTGRES_PASSWORD=YourSecurePassword
   ```

2. **OpenAI API ключ:**
   ```
   OMNIMIND_LLM_API_KEY=sk-proj-YourKey
   OPENAI_API_KEY=sk-proj-YourKey
   ```

3. **Redis пароль:**
   ```
   OMNIMIND_REDIS_PASSWORD=YourRedisPassword
   ```

### Docker (.env.docker)

Пароли уже настроены, но для production замените:

```yaml
# docker-compose.yml
POSTGRES_PASSWORD: YourSecurePassword
```

---

## Тестирование подключения

### PostgreSQL

```bash
# Для Docker
docker exec -it ai_postgres psql -U memory_user -d memory

# Для Production
psql -h your-db.example.com -U memory_user -d memory_prod
```

### Redis

```bash
# Для Docker
docker exec -it ai_redis redis-cli ping

# Для Production
redis-cli -h your-redis.example.com -a YourPassword ping
```

### LLM

```bash
# Ollama
curl http://localhost:11434/api/tags

# DeepSeek
curl https://api.deepseek.com/v1/models -H "Authorization: Bearer sk-YOUR_KEY"

# OpenAI
curl https://api.openai.com/v1/models -H "Authorization: Bearer sk-YOUR_KEY"
```

---

## Troubleshooting

### "Connection refused" к PostgreSQL

```bash
# Проверить что БД запущена
docker ps | grep ai_postgres

# Проверить логи
docker-compose logs ai_postgres

# Проверить подключение
docker exec ai_postgres pg_isready -U memory_user -d memory
```

### "Connection refused" к Redis

```bash
# Проверить что Redis запущен
docker ps | grep ai_redis

# Проверить ping
docker exec ai_redis redis-cli ping
```

### Ollama не отвечает

```bash
# Проверить статус
ollama list

# Загрузить модель
ollama pull llama3.2

# Перезапустить Ollama
brew services restart ollama  # macOS
sudo systemctl restart ollama  # Linux
```

---

## Дополнительные ресурсы

- [Docker Compose документация](docker/README.md)
- [Инструкция по установке](INSTALL.md)
- [Основной README](README.md)
