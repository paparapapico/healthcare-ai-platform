#!/bin/bash
# Deployment Script
# 파일 위치: scripts/deploy.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="healthcare-app"
ENVIRONMENT=${1:-staging}  # Default to staging

echo -e "${GREEN}Starting deployment for ${PROJECT_NAME} - ${ENVIRONMENT}${NC}"

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check requirements
echo -e "${YELLOW}Checking requirements...${NC}"
for cmd in docker docker-compose git; do
    if ! command_exists $cmd; then
        echo -e "${RED}Error: $cmd is not installed${NC}"
        exit 1
    fi
done
echo -e "${GREEN}All requirements met${NC}"

# Pull latest code
echo -e "${YELLOW}Pulling latest code...${NC}"
git pull origin main

# Load environment variables
if [ -f ".env.$ENVIRONMENT" ]; then
    echo -e "${YELLOW}Loading environment variables...${NC}"
    export $(cat .env.$ENVIRONMENT | grep -v '^#' | xargs)
else
    echo -e "${RED}Warning: .env.$ENVIRONMENT file not found${NC}"
fi

# Build and start containers
echo -e "${YELLOW}Building and starting containers...${NC}"
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
else
    docker-compose up -d --build
fi

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Run database migrations
echo -e "${YELLOW}Running database migrations...${NC}"
docker-compose exec -T backend alembic upgrade head

# Collect static files (if applicable)
# docker-compose exec -T backend python manage.py collectstatic --noinput

# Run tests
if [ "$ENVIRONMENT" != "production" ]; then
    echo -e "${YELLOW}Running tests...${NC}"
    docker-compose exec -T backend pytest tests/ -v
fi

# Health check
echo -e "${YELLOW}Performing health check...${NC}"
HEALTH_CHECK=$(curl -s http://localhost:8000/health | grep -o '"status":"healthy"')
if [ -n "$HEALTH_CHECK" ]; then
    echo -e "${GREEN}Health check passed${NC}"
else
    echo -e "${RED}Health check failed${NC}"
    exit 1
fi

# Clean up
echo -e "${YELLOW}Cleaning up...${NC}"
docker system prune -f

echo -e "${GREEN}Deployment completed successfully!${NC}"

# Show running containers
echo -e "${YELLOW}Running containers:${NC}"
docker-compose ps

# Show logs (last 50 lines)
echo -e "${YELLOW}Recent logs:${NC}"
docker-compose logs --tail=50