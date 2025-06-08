.PHONY: build start stop restart status logs shell clean check-gpu check-redis

# Build all containers
build:
	docker-compose build

# Start all services
start:
	docker-compose up -d

# Start in interactive mode
start-interactive:
	docker-compose up

# Stop all services
stop:
	docker-compose down

# Restart all services
restart:
	docker-compose restart

# Check status of services
status:
	docker-compose ps

# View logs of all services
logs:
	docker-compose logs -f

# View logs of a specific service
logs-%:
	docker-compose logs -f $*

# Open shell in a container
shell-%:
	docker-compose exec $* /bin/bash

# Clean up (stop and remove containers, networks, volumes)
clean:
	docker-compose down -v

# Remove all containers and images
clean-all:
	docker-compose down -v --rmi all

# Check GPU status
check-gpu:
	docker-compose exec doctr nvidia-smi

# Check Redis connection
check-redis:
	docker-compose exec redis redis-cli ping

# Help command
help:
	@echo "OCR Pipeline System Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make build              - Build all containers"
	@echo "  make start              - Start all services in detached mode"
	@echo "  make start-interactive  - Start all services in interactive mode"
	@echo "  make stop               - Stop all services"
	@echo "  make restart            - Restart all services"
	@echo "  make status             - Check status of services"
	@echo "  make logs               - View logs of all services"
	@echo "  make logs-SERVICE       - View logs of a specific service (e.g., make logs-api)"
	@echo "  make shell-SERVICE      - Open shell in a specific container (e.g., make shell-doctr)"
	@echo "  make clean              - Clean up containers, networks, volumes"
	@echo "  make clean-all          - Clean up containers, networks, volumes, and images"
	@echo "  make check-gpu          - Check GPU status"
	@echo "  make check-redis        - Check Redis connection"
	@echo "  make help               - Show this help message"