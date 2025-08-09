#!/bin/bash
# 부하 테스트 실행 스크립트

echo "🚀 Starting Load Test..."

# Locust 설치 확인
if ! command -v locust &> /dev/null; then
    echo "Installing Locust..."
    pip install locust
fi

# 서버 상태 확인
curl -f http://localhost:8000/health || {
    echo "❌ Server is not running. Start it first!"
    exit 1
}

echo "📊 Starting Locust web interface..."
echo "   Open http://localhost:8089 in your browser"
echo "   Use these settings:"
echo "   - Number of users: 10"
echo "   - Spawn rate: 2"
echo "   - Host: http://localhost:8000"

# Locust 실행
locust -f tests/locustfile.py --host=http://localhost:8000