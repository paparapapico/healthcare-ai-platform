import { io, Socket } from 'socket.io-client';

class WebSocketService {
  private socket: Socket | null = null;
  private url: string;

  constructor() {
    this.url = process.env.REACT_APP_WS_URL || 'http://localhost:8000';
  }

  connect(token: string): void {
    this.socket = io(this.url, {
      auth: {
        token: token
      },
      transports: ['websocket']
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  // 실시간 시스템 메트릭 구독
  subscribeToMetrics(callback: (data: any) => void): void {
    if (this.socket) {
      this.socket.on('system_metrics', callback);
    }
  }

  // 실시간 사용자 활동 구독
  subscribeToUserActivity(callback: (data: any) => void): void {
    if (this.socket) {
      this.socket.on('user_activity', callback);
    }
  }

  // 실시간 운동 세션 구독
  subscribeToExerciseSessions(callback: (data: any) => void): void {
    if (this.socket) {
      this.socket.on('exercise_session', callback);
    }
  }

  unsubscribeAll(): void {
    if (this.socket) {
      this.socket.off('system_metrics');
      this.socket.off('user_activity');
      this.socket.off('exercise_session');
    }
  }
}

export const wsService = new WebSocketService();