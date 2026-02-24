#!/bin/bash
# ===========================================
# Memory-MCP Docker Compose Helper Script
# ===========================================

set -e

COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="memory-mcp"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===========================================
# Functions
# ===========================================

print_header() {
    echo -e "${BLUE}===========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker не найден. Установите Docker."
        exit 1
    fi
    
    if ! command -v docker compose &> /dev/null; then
        print_error "Docker Compose не найден. Установите Docker Compose."
        exit 1
    fi
    
    print_success "Docker и Docker Compose доступны"
}

start_services() {
    print_header "Запуск сервисов..."
    
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d
    
    print_success "Сервисы запущены"
    echo ""
    echo "Проверка статуса:"
    echo "  docker compose -f $COMPOSE_FILE -p $PROJECT_NAME ps"
    echo ""
    echo "Просмотр логов:"
    echo "  docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f"
}

stop_services() {
    print_header "Остановка сервисов..."
    
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down
    
    print_success "Сервисы остановлены"
}

restart_services() {
    print_header "Перезапуск сервисов..."
    
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME restart
    
    print_success "Сервисы перезапущены"
}

status_services() {
    print_header "Статус сервисов"
    echo ""
    
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
}

logs_services() {
    local service=$1
    
    if [ -n "$service" ]; then
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f $service
    else
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f
    fi
}

clean_all() {
    print_warning "ВНИМАНИЕ: Это удалит все данные!"
    read -p "Вы уверены? (y/N): " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        print_header "Удаление всех сервисов и данных..."
        
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v
        
        print_success "Все сервисы и данные удалены"
    else
        print_warning "Отменено"
    fi
}

backup_postgres() {
    local filename="backup_$(date +%Y%m%d_%H%M%S).sql"
    
    print_header "Создание бэкапа PostgreSQL..."
    
    docker exec ai_postgres pg_dump -U memory_user memory > $filename
    
    print_success "Бэкап создан: $filename"
}

backup_redis() {
    local filename="redis_backup_$(date +%Y%m%d_%H%M%S).rdb"
    
    print_header "Создание бэкапа Redis..."
    
    docker exec ai_redis redis-cli BGSAVE
    sleep 2
    docker cp ai_redis:/data/dump.rdb $filename
    
    print_success "Бэкап создан: $filename"
}

health_check() {
    print_header "Проверка здоровья сервисов"
    echo ""
    
    # PostgreSQL
    if docker exec ai_postgres pg_isready -U memory_user -d memory &> /dev/null; then
        print_success "PostgreSQL: OK"
    else
        print_error "PostgreSQL: НЕДОСТУПЕН"
    fi
    
    # Redis
    if [ "$(docker exec ai_redis redis-cli ping)" = "PONG" ]; then
        print_success "Redis: OK"
    else
        print_error "Redis: НЕДОСТУПЕН"
    fi
    
    echo ""
}

show_help() {
    print_header "Memory-MCP Docker Compose Helper"
    echo ""
    echo "Использование:"
    echo "  $0 <command> [options]"
    echo ""
    echo "Команды:"
    echo "  start       - Запустить все сервисы"
    echo "  stop        - Остановить все сервисы"
    echo "  restart     - Перезапустить сервисы"
    echo "  status      - Показать статус сервисов"
    echo "  logs [svc]  - Показать логи (опционально сервис)"
    echo "  health      - Проверка здоровья"
    echo "  clean       - Удалить всё (сервисы + данные)"
    echo "  backup      - Создать бэкапы (postgres + redis)"
    echo "  help        - Показать эту справку"
    echo ""
    echo "Примеры:"
    echo "  $0 start"
    echo "  $0 logs ai_postgres"
    echo "  $0 backup"
    echo ""
}

# ===========================================
# Main
# ===========================================

check_docker

case "${1:-help}" in
    start)
        start_services
        sleep 3
        health_check
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        status_services
        ;;
    logs)
        logs_services "$2"
        ;;
    health)
        health_check
        ;;
    clean)
        clean_all
        ;;
    backup)
        backup_postgres
        backup_redis
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Неизвестная команда: $1"
        show_help
        exit 1
        ;;
esac
