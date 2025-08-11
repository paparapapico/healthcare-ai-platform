# 파일: HealthcareAI/backend/app/websocket_server.py
# WebSocket 서버 설정 수정

import socketio
from fastapi import FastAPI
import uvicorn
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Socket.IO 서버 생성 (CORS 허용)
sio = socketio.AsyncServer(
    cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    async_mode='asgi',
    logger=True,
    engineio_logger=True
)

# Socket.IO ASGI 앱 생성
socket_app = socketio.ASGIApp(sio)

@sio.event
async def connect(sid, environ, auth):
    """클라이언트 연결 이벤트"""
    try:
        logger.info(f"Client {sid} attempting to connect")
        
        # 인증 토큰 확인 (선택사항)
        if auth and 'token' in auth:
            # 여기서 토큰 검증 로직을 추가할 수 있음
            logger.info(f"Client {sid} provided token")
        
        logger.info(f"Client {sid} connected successfully")
        
        # 연결 성공 메시지 전송
        await sio.emit('connected', {'message': 'Connected to Healthcare AI'}, to=sid)
        
    except Exception as e:
        logger.error(f"Connection error for {sid}: {str(e)}")
        return False  # 연결 거부

@sio.event
async def disconnect(sid):
    """클라이언트 연결 해제 이벤트"""
    logger.info(f"Client {sid} disconnected")

@sio.event
async def join_room(sid, data):
    """룸 참가 이벤트"""
    try:
        room = data.get('room', 'general')
        await sio.enter_room(sid, room)
        logger.info(f"Client {sid} joined room {room}")
        await sio.emit('room_joined', {'room': room}, to=sid)
    except Exception as e:
        logger.error(f"Error joining room: {str(e)}")

@sio.event
async def request_realtime_stats(sid, data):
    """실시간 통계 요청"""
    try:
        # 모의 실시간 데이터 생성
        stats = {
            'timestamp': datetime.now().isoformat(),
            'activeUsers': 42,
            'workoutSessions': 15,
            'onlineUsers': 8
        }
        
        await sio.emit('realtime_stats', stats, to=sid)
        logger.info(f"Sent realtime stats to {sid}")
        
    except Exception as e:
        logger.error(f"Error sending realtime stats: {str(e)}")

# 실시간 데이터 브로드캐스트 함수
async def broadcast_realtime_data():
    """주기적으로 실시간 데이터 브로드캐스트"""
    while True:
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'activeUsers': 42 + (datetime.now().second % 10),
                'workoutSessions': 15 + (datetime.now().second % 5),
                'onlineUsers': 8 + (datetime.now().second % 3)
            }
            
            await sio.emit('realtime_stats', stats)
            await asyncio.sleep(10)  # 10초마다 업데이트
            
        except Exception as e:
            logger.error(f"Error broadcasting data: {str(e)}")
            await asyncio.sleep(5)