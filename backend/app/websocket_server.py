"""
WebSocket Server for Real-time Pose Analysis
실시간 운동 분석을 위한 WebSocket 서버
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio
import logging
from datetime import datetime
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

from app.services.ai.pose_analyzer import pose_analyzer, ExerciseType

logger = logging.getLogger(__name__)

class ConnectionManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_sessions: Dict[str, Dict] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """새 연결 수락"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_sessions[client_id] = {
            "websocket": websocket,
            "exercise_type": None,
            "start_time": datetime.now(),
            "frame_count": 0,
            "fps": 0
        }
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, websocket: WebSocket, client_id: str):
        """연결 종료"""
        self.active_connections.remove(websocket)
        if client_id in self.user_sessions:
            del self.user_sessions[client_id]
        logger.info(f"Client {client_id} disconnected")
        
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """개인 메시지 전송"""
        await websocket.send_json(message)
        
    async def broadcast(self, message: dict):
        """전체 브로드캐스트"""
        for connection in self.active_connections:
            await connection.send_json(message)

# 연결 관리자 인스턴스
manager = ConnectionManager()

class WorkoutSession:
    """운동 세션 관리"""
    
    def __init__(self, user_id: str, exercise_type: str):
        self.user_id = user_id
        self.exercise_type = ExerciseType(exercise_type)
        self.start_time = datetime.now()
        self.frame_buffer = []
        self.analysis_results = []
        self.is_recording = False
        
        # 성능 메트릭
        self.total_frames = 0
        self.processed_frames = 0
        self.avg_processing_time = 0
        
        # 운동 통계
        self.max_form_score = 0
        self.min_form_score = 100
        self.total_reps = 0
        self.total_calories = 0
        
    async def process_frame(self, frame_data: str) -> dict:
        """프레임 처리 및 분석"""
        try:
            import time
            start_time = time.time()
            
            # Base64 이미지 분석
            result = pose_analyzer.analyze_base64_image(
                frame_data, 
                self.exercise_type.value
            )
            
            if result["success"]:
                analysis_data = result["data"]
                
                # 통계 업데이트
                form_score = analysis_data["form_score"]
                self.max_form_score = max(self.max_form_score, form_score)
                self.min_form_score = min(self.min_form_score, form_score)
                self.total_reps = analysis_data["rep_count"]
                self.total_calories = analysis_data["calories_burned"]
                
                # 처리 시간 계산
                processing_time = (time.time() - start_time) * 1000  # ms
                self.processed_frames += 1
                self.avg_processing_time = (
                    (self.avg_processing_time * (self.processed_frames - 1) + processing_time) 
                    / self.processed_frames
                )
                
                # 결과 저장
                self.analysis_results.append({
                    "timestamp": datetime.now().isoformat(),
                    "data": analysis_data
                })
                
                return {
                    "type": "analysis_result",
                    "success": True,
                    "data": analysis_data,
                    "stats": {
                        "processing_time_ms": round(processing_time, 2),
                        "avg_processing_time_ms": round(self.avg_processing_time, 2),
                        "frames_processed": self.processed_frames,
                        "max_form_score": round(self.max_form_score, 1),
                        "min_form_score": round(self.min_form_score, 1)
                    }
                }
            else:
                return {
                    "type": "analysis_error",
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {
                "type": "error",
                "success": False,
                "error": str(e)
            }
    
    def get_session_summary(self) -> dict:
        """세션 요약 정보"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "type": "session_summary",
            "data": {
                "exercise_type": self.exercise_type.value,
                "duration_seconds": round(duration, 1),
                "total_reps": self.total_reps,
                "total_calories": round(self.total_calories, 1),
                "avg_form_score": round((self.max_form_score + self.min_form_score) / 2, 1),
                "max_form_score": round(self.max_form_score, 1),
                "min_form_score": round(self.min_form_score, 1),
                "frames_processed": self.processed_frames,
                "avg_fps": round(self.processed_frames / duration, 1) if duration > 0 else 0
            }
        }

# 활성 세션 저장
workout_sessions: Dict[str, WorkoutSession] = {}

async def handle_websocket_message(websocket: WebSocket, client_id: str, data: dict):
    """WebSocket 메시지 처리"""
    
    message_type = data.get("type")
    
    try:
        if message_type == "start_workout":
            # 운동 시작
            exercise_type = data.get("exercise_type", "squat")
            workout_sessions[client_id] = WorkoutSession(client_id, exercise_type)
            
            # 포즈 분석기 초기화
            pose_analyzer.reset_exercise_state(ExerciseType(exercise_type))
            
            await manager.send_personal_message({
                "type": "workout_started",
                "data": {
                    "exercise_type": exercise_type,
                    "timestamp": datetime.now().isoformat()
                }
            }, websocket)
            
        elif message_type == "frame":
            # 프레임 분석
            if client_id not in workout_sessions:
                await manager.send_personal_message({
                    "type": "error",
                    "error": "No active workout session"
                }, websocket)
                return
            
            session = workout_sessions[client_id]
            frame_data = data.get("frame")
            
            if frame_data:
                # 비동기 프레임 처리
                result = await session.process_frame(frame_data)
                await manager.send_personal_message(result, websocket)
            
        elif message_type == "stop_workout":
            # 운동 종료
            if client_id in workout_sessions:
                session = workout_sessions[client_id]
                summary = session.get_session_summary()
                
                await manager.send_personal_message(summary, websocket)
                
                # 세션 정리
                del workout_sessions[client_id]
                
        elif message_type == "ping":
            # 연결 유지용 ping
            await manager.send_personal_message({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }, websocket)
            
        else:
            await manager.send_personal_message({
                "type": "error",
                "error": f"Unknown message type: {message_type}"
            }, websocket)
            
    except Exception as e:
        logger.error(f"Message handling error: {e}")
        await manager.send_personal_message({
            "type": "error",
            "error": str(e)
        }, websocket)

# FastAPI WebSocket 엔드포인트
from fastapi import FastAPI

app = FastAPI()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket 엔드포인트"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_json()
            
            # 메시지 처리
            await handle_websocket_message(websocket, client_id, data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        
        # 세션 정리
        if client_id in workout_sessions:
            del workout_sessions[client_id]
            
        logger.info(f"Client {client_id} disconnected")
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, client_id)

# ========================
# Utility Functions
# ========================

async def analyze_video_file(video_path: str, exercise_type: str) -> dict:
    """비디오 파일 분석 (테스트/디버깅용)"""
    cap = cv2.VideoCapture(video_path)
    
    results = []
    frame_count = 0
    
    exercise_enum = ExerciseType(exercise_type)
    pose_analyzer.reset_exercise_state(exercise_enum)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 매 3프레임마다 분석 (성능 최적화)
        if frame_count % 3 == 0:
            result = pose_analyzer.analyze_frame(frame, exercise_enum)
            results.append({
                "frame": frame_count,
                "timestamp": frame_count / 30.0,  # 30 FPS 가정
                "analysis": result
            })
    
    cap.release()
    
    return {
        "total_frames": frame_count,
        "analyzed_frames": len(results),
        "results": results
    }

# ========================
# Performance Monitoring
# ========================

class PerformanceMonitor:
    """성능 모니터링"""
    
    def __init__(self):
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "total_frames_processed": 0,
            "avg_processing_time_ms": 0,
            "errors": 0
        }
    
    def update_metrics(self, metric: str, value: float):
        """메트릭 업데이트"""
        if metric == "processing_time":
            # 이동 평균 계산
            self.metrics["total_frames_processed"] += 1
            n = self.metrics["total_frames_processed"]
            self.metrics["avg_processing_time_ms"] = (
                (self.metrics["avg_processing_time_ms"] * (n - 1) + value) / n
            )
        else:
            self.metrics[metric] = value
    
    def get_metrics(self) -> dict:
        """현재 메트릭 반환"""
        return {
            **self.metrics,
            "active_connections": len(manager.active_connections),
            "active_sessions": len(workout_sessions)
        }

# 성능 모니터 인스턴스
performance_monitor = PerformanceMonitor()

@app.get("/api/v1/ws/metrics")
async def get_websocket_metrics():
    """WebSocket 서버 메트릭"""
    return performance_monitor.get_metrics()