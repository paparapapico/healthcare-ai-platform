import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { ApiResponse, User, Exercise, HealthData, Challenge, DashboardStats } from '../types';
import { SubscriptionPlan, SubscriptionData, Payment } from '../types/payment';

class ApiService {
  public api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 요청 인터셉터 - JWT 토큰 추가
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // 응답 인터셉터 - 에러 처리
    this.api.interceptors.response.use(
      (response: AxiosResponse) => response,
      async (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('access_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // 인증 관련
  async login(email: string, password: string): Promise<ApiResponse<{ access_token: string; user: User }>> {
    const response = await this.api.post('/auth/login', { email, password });
    return response.data;
  }

  async register(userData: { email: string; password: string; name: string }): Promise<ApiResponse<User>> {
    const response = await this.api.post('/auth/register', userData);
    return response.data;
  }

  async getCurrentUser(): Promise<ApiResponse<User>> {
    const response = await this.api.get('/users/me');
    return response.data;
  }

  // 관리자 - 사용자 관리
  async getUsers(page: number = 1, limit: number = 20): Promise<ApiResponse<{ users: User[]; total: number }>> {
    const response = await this.api.get(`/admin/users?page=${page}&limit=${limit}`);
    return response.data;
  }

  async getUserById(userId: string): Promise<ApiResponse<User>> {
    const response = await this.api.get(`/admin/users/${userId}`);
    return response.data;
  }

  async updateUser(userId: string, userData: Partial<User>): Promise<ApiResponse<User>> {
    const response = await this.api.put(`/admin/users/${userId}`, userData);
    return response.data;
  }

  async deleteUser(userId: string): Promise<ApiResponse<void>> {
    const response = await this.api.delete(`/admin/users/${userId}`);
    return response.data;
  }

  // 운동 데이터
  async getExercises(userId?: string, page: number = 1): Promise<ApiResponse<{ exercises: Exercise[]; total: number }>> {
    const url = userId ? `/admin/exercises?user_id=${userId}&page=${page}` : `/exercises?page=${page}`;
    const response = await this.api.get(url);
    return response.data;
  }

  // 건강 데이터
  async getHealthData(userId?: string, startDate?: string, endDate?: string): Promise<ApiResponse<HealthData[]>> {
    let url = userId ? `/admin/health-data?user_id=${userId}` : '/health-data';
    if (startDate && endDate) {
      url += `&start_date=${startDate}&end_date=${endDate}`;
    }
    const response = await this.api.get(url);
    return response.data;
  }

  // 챌린지
  async getChallenges(): Promise<ApiResponse<Challenge[]>> {
    const response = await this.api.get('/challenges');
    return response.data;
  }

  async createChallenge(challengeData: Omit<Challenge, 'id' | 'participants'>): Promise<ApiResponse<Challenge>> {
    const response = await this.api.post('/admin/challenges', challengeData);
    return response.data;
  }

  async updateChallenge(challengeId: string, challengeData: Partial<Challenge>): Promise<ApiResponse<Challenge>> {
    const response = await this.api.put(`/admin/challenges/${challengeId}`, challengeData);
    return response.data;
  }

  // 대시보드 통계
  async getDashboardStats(): Promise<ApiResponse<DashboardStats>> {
    const response = await this.api.get('/admin/dashboard/stats');
    return response.data;
  }

  // 시스템 모니터링
  async getSystemMetrics(): Promise<ApiResponse<any>> {
    const response = await this.api.get('/admin/system/metrics');
    return response.data;
  }

  // 알림 발송
  async sendNotification(notificationData: {
    user_id?: string;
    title: string;
    message: string;
    type: 'push' | 'email';
  }): Promise<ApiResponse<void>> {
    const response = await this.api.post('/admin/notifications', notificationData);
    return response.data;
  }

  // 결제 관련 API 메서드들
  async getSubscriptionPlans(): Promise<ApiResponse<SubscriptionPlan[]>> {
    const response = await this.api.get('/payment/plans');
    return response.data;
  }

  async createCheckoutSession(planId: string): Promise<ApiResponse<{ checkout_url: string; session_id: string }>> {
    const response = await this.api.post('/payment/checkout', { plan_id: planId });
    return response.data;
  }

  async handlePaymentSuccess(sessionId: string): Promise<ApiResponse<any>> {
    const response = await this.api.get(`/payment/success?session_id=${sessionId}`);
    return response.data;
  }

  async getUserSubscription(): Promise<ApiResponse<SubscriptionData>> {
    const response = await this.api.get('/payment/subscription');
    return response.data;
  }

  async cancelSubscription(immediate: boolean = false): Promise<ApiResponse<any>> {
    const response = await this.api.post('/payment/cancel', { immediate });
    return response.data;
  }

  async getPaymentHistory(): Promise<ApiResponse<Payment[]>> {
    const response = await this.api.get('/payment/payments');
    return response.data;
  }
}

export const apiService = new ApiService();

// 편의를 위한 별도 export (기존 paymentApi와 호환)
export const paymentApi = {
  getSubscriptionPlans: () => apiService.getSubscriptionPlans().then(res => res.data),
  createCheckoutSession: (planId: string) => apiService.createCheckoutSession(planId).then(res => res.data),
  handlePaymentSuccess: (sessionId: string) => apiService.handlePaymentSuccess(sessionId),
  getUserSubscription: () => apiService.getUserSubscription().then(res => res.data),
  cancelSubscription: (immediate?: boolean) => apiService.cancelSubscription(immediate),
  getPaymentHistory: () => apiService.getPaymentHistory().then(res => res.data),
};