import axios from 'axios';
import { User, DashboardStats } from '../types';

// API �⺻ URL ���� (v1 ����!)
const API_BASE_URL = 'http://localhost:8000';

// Axios �ν��Ͻ� ����
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

// ��û ���ͼ���
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  
  console.log(' API ��û:', {
    method: config.method,
    url: config.url,
    baseURL: config.baseURL,
    fullURL: `${config.baseURL}${config.url}`
  });
  
  return config;
});

// ���� ���ͼ���
api.interceptors.response.use(
  (response) => {
    console.log(' API ���� ����:', response.status, response.data);
    return response;
  },
  async (error) => {
    console.error(' API ���� ����:', error.response?.status, error.message);
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API (�ùٸ� ���)
export const authAPI = {
  login: async (email: string, password: string) => {
    console.log(' �α��� �õ�:', { email });
    
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    
    try {
      // �ùٸ� ��� ��� (v1 ����)
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        body: formData,
      });
      
      console.log(' �α��� ����:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(' �α��� ����:', errorText);
        throw new Error(`Login failed: ${response.status}`);
      }
      
      const data = await response.json();
      console.log(' �α��� ����:', data);
      return data;
      
    } catch (error) {
      console.error(' �α��� ����:', error);
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

// Dashboard API (�ùٸ� ���)
export const dashboardAPI = {
  getStats: async (): Promise<DashboardStats> => {
    const response = await api.get('/api/dashboard/stats');
    return response.data;
  }
};
