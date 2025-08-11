import axios from 'axios';
import { User, DashboardStats } from '../types';

// API 기본 URL 설정 (v1 제거!)
const API_BASE_URL = 'http://localhost:8000';

// Axios 인스턴스 생성
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

// 요청 인터셉터
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  
  console.log(' API 요청:', {
    method: config.method,
    url: config.url,
    baseURL: config.baseURL,
    fullURL: `${config.baseURL}${config.url}`
  });
  
  return config;
});

// 응답 인터셉터
api.interceptors.response.use(
  (response) => {
    console.log(' API 응답 성공:', response.status, response.data);
    return response;
  },
  async (error) => {
    console.error(' API 응답 실패:', error.response?.status, error.message);
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API (올바른 경로)
export const authAPI = {
  login: async (email: string, password: string) => {
    console.log(' 로그인 시도:', { email });
    
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    
    try {
      // 올바른 경로 사용 (v1 없음)
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        body: formData,
      });
      
      console.log(' 로그인 응답:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(' 로그인 실패:', errorText);
        throw new Error(`Login failed: ${response.status}`);
      }
      
      const data = await response.json();
      console.log(' 로그인 성공:', data);
      return data;
      
    } catch (error) {
      console.error(' 로그인 오류:', error);
      throw error;
    }
  },
  
  getCurrentUser: async (): Promise<User> => {
    const response = await api.get('/api/auth/me');
    return response.data;
  },
  
  logout: () => {
    localStorage.removeItem('access_token');
  }
};

// Dashboard API (올바른 경로)
export const dashboardAPI = {
  getStats: async (): Promise<DashboardStats> => {
    const response = await api.get('/api/dashboard/stats');
    return response.data;
  }
};
