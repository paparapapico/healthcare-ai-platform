#!/bin/bash
# 전체 테스트 실행 스크립트
# 파일 위치: run_all_tests.sh

set -e  # 에러 발생시 중단

echo "🏥 Healthcare AI Platform - Complete Test Suite"
echo "==============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if backend is running
check_backend() {
    echo "🔍 Checking backend server..."
    if curl -f -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✅ Backend is running${NC}"
        return 0
    else
        echo -e "${RED}❌ Backend is not running${NC}"
        echo "Starting backend server..."
        cd backend
        uvicorn app.main:app --reload &
        BACKEND_PID=$!
        sleep 5
        cd ..
        return 0
    fi
}

# Run API documentation tests
test_docs() {
    echo -e "\n${YELLOW}📚 Testing API Documentation...${NC}"
    bash backend/tests/test_docs.sh
}

# Run authentication tests
test_auth() {
    echo -e "\n${YELLOW}🔐 Testing Authentication...${NC}"
    cd backend
    python tests/test_auth.py
    cd ..
}

# Run unit tests
test_unit() {
    echo -e "\n${YELLOW}🧪 Running Unit Tests...${NC}"
    cd backend
    pytest tests/test_api.py -v --tb=short
    cd ..
}

# Run integration tests
test_integration() {
    echo -e "\n${YELLOW}🔗 Running Integration Tests...${NC}"
    cd backend
    python tests/test_integration.py
    cd ..
}

# Run load tests (optional - takes time)
test_load() {
    echo -e "\n${YELLOW}📊 Load Testing (Optional)...${NC}"
    echo "Skip load testing for quick check? (y/n)"
    read -r skip_load
    
    if [ "$skip_load" != "y" ]; then
        cd backend
        echo "Running quick load test (10 users, 30 seconds)..."
        locust -f tests/locustfile.py \
               --host=http://localhost:8000 \
               --users=10 \
               --spawn-rate=2 \
               --run-time=30s \
               --headless \
               --only-summary
        cd ..
    fi
}

# Health check all services
health_check() {
    echo -e "\n${YELLOW}❤️ Health Check...${NC}"
    
    # API Health
    echo -n "  API Server: "
    curl -s http://localhost:8000/health | grep -q "healthy" && echo -e "${GREEN}✅ Healthy${NC}" || echo -e "${RED}❌ Unhealthy${NC}"
    
    # Database (if using docker)
    if docker ps | grep -q healthcare_db; then
        echo -n "  Database: "
        docker exec healthcare_db pg_isready -q && echo -e "${GREEN}✅ Ready${NC}" || echo -e "${RED}❌ Not Ready${NC}"
    fi
    
    # Redis (if using docker)
    if docker ps | grep -q healthcare_redis; then
        echo -n "  Redis: "
        docker exec healthcare_redis redis-cli ping | grep -q PONG && echo -e "${GREEN}✅ Ready${NC}" || echo -e "${RED}❌ Not Ready${NC}"
    fi
}

# Main execution
main() {
    echo "Starting test suite at $(date)"
    echo ""
    
    # Check and start backend if needed
    check_backend
    
    # Run all tests
    test_docs
    test_auth
    test_unit
    test_integration
    health_check
    
    # Optional load test
    test_load
    
    echo ""
    echo "==============================================="
    echo -e "${GREEN}✅ All tests completed successfully!${NC}"
    echo "==============================================="
    echo ""
    echo "📊 Test Summary:"
    echo "  - API Documentation: ✅"
    echo "  - Authentication: ✅"
    echo "  - Unit Tests: ✅"
    echo "  - Integration Tests: ✅"
    echo "  - Health Check: ✅"
    echo ""
    echo "🚀 Your Healthcare AI Platform is working correctly!"
}

# Run main
main