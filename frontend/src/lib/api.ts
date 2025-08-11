// API 기본 설정
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const api = {
  // 기본 API 함수들
  get: async (endpoint: string) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    return response.json();
  },
  
  post: async (endpoint: string, data: any) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return response.json();
  }
};

export const dashboardAPI = {
  getStats: () => api.get('/dashboard/stats'),
  getPatients: () => api.get('/dashboard/patients'),
  getAppointments: () => api.get('/dashboard/appointments'),
  getRevenue: () => api.get('/dashboard/revenue'),
};