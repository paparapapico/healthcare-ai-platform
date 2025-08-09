// 파일: ~/HealthcareAI/run_frontend_tests.sh
#!/bin/bash

# Day 8: Frontend Test Runner
# 위치: ~/HealthcareAI/run_frontend_tests.sh

echo "🧪 Healthcare AI Frontend 테스트 실행..."

cd ~/HealthcareAI/frontend

echo "📦 의존성 확인..."
if [ ! -d "node_modules" ]; then
    echo "📥 패키지 설치 중..."
    npm install
fi

echo "🔍 린트 검사..."
npm run lint

echo "🧪 유닛 테스트 실행..."
npm run test -- --coverage --watchAll=false

echo "🏗️ 빌드 테스트..."
npm run build

echo "✅ 모든 테스트 완료!"
echo "📊 커버리지 리포트: coverage/lcov-report/index.html"
echo "🚀 개발 서버 실행: npm run dev"

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