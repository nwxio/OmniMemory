#!/bin/bash
# ===========================================
# Memory-MCP Docker Compose Helper Script
# ===========================================

set -e

COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="memory-mcp"

# Output colors
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
        print_error "Docker not found. Please install Docker."
        exit 1
    fi

    if ! command -v docker compose &> /dev/null; then
        print_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi

    print_success "Docker and Docker Compose are available"
}

start_services() {
    print_header "Starting services..."
    
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d
    
    print_success "Services started"
    echo ""
    echo "Check status:"
    echo "  docker compose -f $COMPOSE_FILE -p $PROJECT_NAME ps"
    echo ""
    echo "View logs:"
    echo "  docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f"
}

stop_services() {
    print_header "Stopping services..."
    
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down
    
    print_success "Services stopped"
}

restart_services() {
    print_header "Restarting services..."
    
    docker compose -f $COMPOSE_FILE -p $PROJECT_NAME restart
    
    print_success "Services restarted"
}

status_services() {
    print_header "Service status"
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
    print_warning "WARNING: This will delete all data!"
    read -p "Are you sure? (y/N): " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        print_header "Deleting all services and data..."
        
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v
        
        print_success "All services and data have been removed"
    else
        print_warning "Cancelled"
    fi
}

backup_postgres() {
    local filename="backup_$(date +%Y%m%d_%H%M%S).sql"
    
    print_header "Creating PostgreSQL backup..."
    
    docker exec ai_postgres pg_dump -U memory_user memory > $filename
    
    print_success "Backup created: $filename"
}

backup_redis() {
    local filename="redis_backup_$(date +%Y%m%d_%H%M%S).rdb"
    
    print_header "Creating Redis backup..."
    
    docker exec ai_redis redis-cli BGSAVE
    sleep 2
    docker cp ai_redis:/data/dump.rdb $filename
    
    print_success "Backup created: $filename"
}

health_check() {
    print_header "Service health check"
    echo ""
    
    # PostgreSQL
    if docker exec ai_postgres pg_isready -U memory_user -d memory &> /dev/null; then
        print_success "PostgreSQL: OK"
    else
        print_error "PostgreSQL: UNAVAILABLE"
    fi
    
    # Redis
    if [ "$(docker exec ai_redis redis-cli ping)" = "PONG" ]; then
        print_success "Redis: OK"
    else
        print_error "Redis: UNAVAILABLE"
    fi
    
    echo ""
}

show_help() {
    print_header "Memory-MCP Docker Compose Helper"
    echo ""
    echo "Usage:"
    echo "  $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start       - Start all services"
    echo "  stop        - Stop all services"
    echo "  restart     - Restart services"
    echo "  status      - Show service status"
    echo "  logs [svc]  - Show logs (optional service)"
    echo "  health      - Health check"
    echo "  clean       - Remove everything (services + data)"
    echo "  backup      - Create backups (postgres + redis)"
    echo "  help        - Show this help"
    echo ""
    echo "Examples:"
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
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
