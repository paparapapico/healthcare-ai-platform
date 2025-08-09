// 파일: ~/HealthcareAI/start_full_stack.sh
#!/bin/bash

# Day 8: Full Stack Startup Script
# 위치: ~/HealthcareAI/start_full_stack.sh

echo "🚀 Healthcare AI 전체 스택 시작..."

# 터미널 세션별로 실행
echo "📦 Docker 서비스 시작..."
docker-compose up -d

echo "⏳ 서비스 준비 대기 (10초)..."
sleep 10

echo "🔧 백엔드 시작..."
cd ~/HealthcareAI/backend
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

echo "⏳ 백엔드 준비 대기 (5초)..."
sleep 5

echo "🎨 프론트엔드 시작..."
cd ~/HealthcareAI/frontend
npm run dev &
FRONTEND_PID=$!

echo "📱 모바일 앱 시작 (선택사항)..."
# cd ~/HealthcareAI/mobile/HealthcareApp
# npx react-native run-ios &

echo "✅ 전체 스택 실행 완료!"
echo ""
echo "🌐 서비스 접근 정보:"
echo "  - 프론트엔드 대시보드: http://localhost:3000"
echo "  - 백엔드 API 문서: http://localhost:8000/docs"
echo "  - Grafana 모니터링: http://localhost:3000"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo ""
echo "🔑 로그인 정보:"
echo "  - Email: admin@healthcare.ai"
echo "  - Password: admin123"
echo ""
echo "⏹️ 종료하려면 Ctrl+C 누르세요"

# 트랩으로 정리 작업
trap 'echo "🛑 서비스 종료 중..."; kill $BACKEND_PID $FRONTEND_PID; docker-compose down; exit' INT

# 메인 프로세스 유지
wait