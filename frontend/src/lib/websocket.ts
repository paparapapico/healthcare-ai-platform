// 파일: HealthcareAI/frontend/src/lib/websocket.ts (수정된 버전)
import { io, Socket } from 'socket.io-client';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  
  connect() {
    try {
      // Socket.IO 클라이언트 연결
      this.socket = io('http://localhost:8000', {
        transports: ['websocket', 'polling'], // fallback 추가
        timeout: 20000,
        forceNew: true,
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: 1000,
        auth: {
          token: localStorage.getItem('access_token') || ''
        }
      });
      
      this.socket.on('connect', () => {
        console.log('✅ WebSocket connected successfully');
        this.reconnectAttempts = 0;
      });
      
      this.socket.on('disconnect', (reason) => {
        console.log('❌ WebSocket disconnected:', reason);
      });
      
      this.socket.on('connect_error', (error) => {
        console.log('❌ WebSocket connection error:', error.message);
        this.reconnectAttempts++;
        
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          console.log('❌ Max reconnection attempts reached');
        }
      });
      
      this.socket.on('connected', (data) => {
        console.log('📡 Server confirmed connection:', data);
      });
      
      return this.socket;
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
      return null;
    }
  }
  
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      console.log('🔌 WebSocket disconnected');
    }
  }
  
  subscribeToRealtime(callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on('realtime_stats', callback);
      
      // 실시간 데이터 요청
      this.socket.emit('request_realtime_stats', {});
    }
  }
  
  subscribeToWorkouts(callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on('workout_update', callback);
    }
  }
  
  subscribeToUserActivity(callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on('user_activity', callback);
    }
  }
  
  joinRoom(room: string) {
    if (this.socket) {
      this.socket.emit('join_room', { room });
    }
  }
  
  isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

export const wsService = new WebSocketService();