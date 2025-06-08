#!/bin/bash

# OCR Pipeline System Management Script

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print header
print_header() {
  echo -e "${BLUE}============================================${NC}"
  echo -e "${BLUE}      OCR Pipeline System Management        ${NC}"
  echo -e "${BLUE}============================================${NC}"
}

# Function to check if Docker and docker-compose are installed
check_dependencies() {
  echo -e "${YELLOW}Checking dependencies...${NC}"
  
  if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
  fi
  
  if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
  }
  
  echo -e "${GREEN}All dependencies are installed.${NC}"
}

# Function to start the system
start_system() {
  echo -e "${YELLOW}Starting OCR Pipeline system...${NC}"
  docker-compose up -d
  echo -e "${GREEN}System started successfully.${NC}"
  echo -e "Access the UI at: ${BLUE}http://localhost:7860${NC}"
}

# Function to stop the system
stop_system() {
  echo -e "${YELLOW}Stopping OCR Pipeline system...${NC}"
  docker-compose down
  echo -e "${GREEN}System stopped successfully.${NC}"
}

# Function to show container status
show_status() {
  echo -e "${YELLOW}Checking container status...${NC}"
  docker-compose ps
}

# Function to show container logs
show_logs() {
  if [ -z "$1" ]; then
    echo -e "${YELLOW}Showing logs from all containers...${NC}"
    docker-compose logs
  else
    echo -e "${YELLOW}Showing logs from $1 container...${NC}"
    docker-compose logs "$1"
  fi
}

# Function to restart a specific container or all containers
restart_containers() {
  if [ -z "$1" ]; then
    echo -e "${YELLOW}Restarting all containers...${NC}"
    docker-compose restart
  else
    echo -e "${YELLOW}Restarting $1 container...${NC}"
    docker-compose restart "$1"
  fi
  echo -e "${GREEN}Restart completed.${NC}"
}

# Function to check GPU status
check_gpu() {
  echo -e "${YELLOW}Checking GPU status...${NC}"
  docker-compose exec doctr nvidia-smi
}

# Function to check Redis
check_redis() {
  echo -e "${YELLOW}Checking Redis connection...${NC}"
  docker-compose exec redis redis-cli ping
}

# Function to display help
show_help() {
  echo "Usage: $0 [command]"
  echo
  echo "Commands:"
  echo "  start           Start all services"
  echo "  stop            Stop all services"
  echo "  restart [name]  Restart all services or a specific service"
  echo "  status          Show status of all services"
  echo "  logs [name]     Show logs of all services or a specific service"
  echo "  gpu             Check GPU status"
  echo "  redis           Check Redis connection"
  echo "  help            Show this help message"
}

# Main script execution
print_header
check_dependencies

# Process command line arguments
case "$1" in
  start)
    start_system
    ;;
  stop)
    stop_system
    ;;
  restart)
    restart_containers "$2"
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs "$2"
    ;;
  gpu)
    check_gpu
    ;;
  redis)
    check_redis
    ;;
  help|*)
    show_help
    ;;
esac