#!/bin/bash

# Day 8: Healthcare AI Frontend 설정 스크립트
# 위치: ~/HealthcareAI/setup_frontend.sh

echo "🚀 Healthcare AI Frontend (Day 8) 설정 시작..."

# frontend 디렉토리로 이동
cd ~/HealthcareAI

# Vite + React + TypeScript 프로젝트 생성
echo "📦 Vite React TypeScript 프로젝트 생성..."
npm create vite@latest frontend -- --template react-ts

cd frontend

# 필요한 패키지 설치
echo "📚 필수 패키지 설치..."
npm install

# 추가 라이브러리 설치
echo "🔧 추가 라이브러리 설치..."
npm install \
  @tanstack/react-query \
  @tanstack/react-table \
  react-router-dom \
  axios \
  socket.io-client \
  recharts \
  @headlessui/react \
  @heroicons/react \
  clsx \
  tailwindcss-animate \
  date-fns \
  react-hook-form \
  @hookform/resolvers \
  yup \
  react-hot-toast \
  framer-motion

# 개발 의존성 설치
npm install -D \
  @types/node \
  tailwindcss \
  postcss \
  autoprefixer \
  @vitejs/plugin-react

# Tailwind CSS 설정
echo "🎨 Tailwind CSS 설정..."
npx tailwindcss init -p

echo "✅ Frontend 기본 설정 완료!"
echo "📁 위치: ~/HealthcareAI/frontend/"
echo "🚀 실행: cd frontend && npm run dev"