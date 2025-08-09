// 파일: ~/HealthcareAI/frontend/src/lib/websocket.ts
import { io, Socket } from 'socket.io-client';

class WebSocketService {
  private socket: Socket | null = null;
  
  connect(token: string) {
    this.socket = io('ws://localhost:8000', {
      auth: { token },
      transports: ['websocket']
    });
    
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
    });
    
    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });
    
    return this.socket;
  }
  
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
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
}

export const wsService = new WebSocketService();