#!/bin/bash
# API 문서 테스트
echo "🧪 Testing API Documentation..."

# API 서버가 실행 중인지 확인
curl -f http://localhost:8000/health || {
    echo "❌ API 서버가 실행되지 않음. 먼저 서버를 시작하세요:"
    echo "cd backend && uvicorn app.main:app --reload"
    exit 1
}

# Swagger UI 확인
echo "✅ Checking Swagger UI..."
curl -s http://localhost:8000/docs | grep -q "swagger-ui" && echo "✅ Swagger UI is working" || echo "❌ Swagger UI failed"

# ReDoc 확인
echo "✅ Checking ReDoc..."
curl -s http://localhost:8000/redoc | grep -q "redoc" && echo "✅ ReDoc is working" || echo "❌ ReDoc failed"

# OpenAPI JSON 확인
echo "✅ Checking OpenAPI JSON..."
curl -s http://localhost:8000/api/openapi.json | python -m json.tool > /dev/null && echo "✅ OpenAPI JSON is valid" || echo "❌ OpenAPI JSON failed"

echo "📊 API Documentation test completed!"