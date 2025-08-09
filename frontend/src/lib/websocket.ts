// 파일: ~/HealthcareAI/frontend/src/lib/websocket.ts
import { io, Socket } from 'socket.io-client';

interface WorkoutUpdate {
  userId: string;
  exerciseType: string;
  duration: number;
  accuracy: number;
}

interface UserActivity {
  userId: string;
  activity: string;
  timestamp: string;
}

// RealtimeData 인터페이스를 실제로 사용하거나 제거
interface RealtimeData {
  timestamp: string;
  activeUsers: number;
  workoutSessions: number;
}

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
  
  // RealtimeData 타입을 실제로 사용
  subscribeToRealtimeData(callback: (data: RealtimeData) => void) {
    if (this.socket) {
      this.socket.on('realtime_stats', callback);
    }
  }
  
  subscribeToWorkouts(callback: (data: WorkoutUpdate) => void) {
    if (this.socket) {
      this.socket.on('workout_update', callback);
    }
  }
  
  subscribeToUserActivity(callback: (data: UserActivity) => void) {
    if (this.socket) {
      this.socket.on('user_activity', callback);
    }
  }
}

export const wsService = new WebSocketService();