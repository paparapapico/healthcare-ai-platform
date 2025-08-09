// 파일: ~/HealthcareAI/frontend/src/lib/api.ts
import axios from 'axios';
import type { User, WorkoutSession, HealthData, Challenge, DashboardStats } from '@/types';

const API_BASE_URL = 'http://localhost:8000/api/v1'

// Axios 인스턴스 생성
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 토큰 인터셉터
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 응답 인터셉터 (토큰 만료 처리)
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: async (email: string, password: string) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    
    const response = await api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
    return response.data;
  },
  
  getCurrentUser: async (): Promise<User> => {
    const response = await api.get('/auth/me');  // /api/ 제거
    return response.data;
  },
  
  logout: () => {
    localStorage.removeItem('access_token');
  }
};

// Users API 수정
export const usersAPI = {
  // 현재 사용자만 가져오기 (목록 대신)
  getCurrentUser: async (): Promise<User> => {
    const response = await api.get('/auth/me');
    return response.data;
  },
  
  getUserById: async (userId: string): Promise<User> => {
    const response = await api.get(`/users/${userId}`);
    return response.data;
  },
  
  // 사용자 목록은 임시로 현재 사용자만 반환
  getUsers: async (skip = 0, limit = 100): Promise<User[]> => {
    try {
      const currentUser = await api.get('/auth/me');
      return [currentUser.data]; // 배열로 감싸서 반환
    } catch (error) {
      console.error('Failed to get users:', error);
      return [];
    }
  },
  
  updateUser: async (userId: string, userData: Partial<User>): Promise<User> => {
    const response = await api.put(`/users/${userId}`, userData);
    return response.data;
  },
  
  deleteUser: async (userId: string): Promise<void> => {
    await api.delete(`/users/${userId}`);
  }
};

// Workouts API
export const workoutsAPI = {
  getWorkouts: async (skip = 0, limit = 100): Promise<WorkoutSession[]> => {
    const response = await api.get(`/api/workouts/?skip=${skip}&limit=${limit}`);
    return response.data;
  },
  
  getUserWorkouts: async (userId: string): Promise<WorkoutSession[]> => {
    const response = await api.get(`/api/workouts/user/${userId}`);
    return response.data;
  },
  
  getWorkoutStats: async (period: '7d' | '30d' | '90d' = '30d') => {
    const response = await api.get(`/api/workouts/stats?period=${period}`);
    return response.data;
  }
};

// Health Data API
export const healthAPI = {
  getHealthData: async (userId: string, dataType?: string): Promise<HealthData[]> => {
    const params = dataType ? `?data_type=${dataType}` : '';
    const response = await api.get(`/api/health/${userId}${params}`);
    return response.data;
  },
  
  addHealthData: async (healthData: Omit<HealthData, 'id' | 'recorded_at'>): Promise<HealthData> => {
    const response = await api.post('/api/health/', healthData);
    return response.data;
  }
};

// Challenges API
export const challengesAPI = {
  getChallenges: async (): Promise<Challenge[]> => {
    const response = await api.get('/api/challenges/');
    return response.data;
  },
  
  createChallenge: async (challenge: Omit<Challenge, 'id' | 'current_value' | 'participants_count'>): Promise<Challenge> => {
    const response = await api.post('/api/challenges/', challenge);
    return response.data;
  },
  
  joinChallenge: async (challengeId: string): Promise<void> => {
    await api.post(`/api/challenges/${challengeId}/join`);
  }
};

// Dashboard API
export const dashboardAPI = {
  getStats: async (): Promise<DashboardStats> => {
    const response = await api.get('/api/dashboard/stats');
    return response.data;
  },
  
  getRealtimeData: async () => {
    const response = await api.get('/api/dashboard/realtime');
    return response.data;
  }
};