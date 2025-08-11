// frontend/src/services/paymentApi.ts
import { apiService } from './api';
import { SubscriptionPlan, SubscriptionData, Payment } from '../types/payment';

class PaymentApiService {
  // 구독 플랜 목록 조회
  async getSubscriptionPlans(): Promise<SubscriptionPlan[]> {
    const response = await apiService.api.get('/payment/plans');
    return response.data.data;
  }

  // 체크아웃 세션 생성
  async createCheckoutSession(planId: string): Promise<{ checkout_url: string; session_id: string }> {
    const response = await apiService.api.post('/payment/checkout', { plan_id: planId });
    return response.data.data;
  }

  // 결제 성공 처리
  async handlePaymentSuccess(sessionId: string): Promise<any> {
    const response = await apiService.api.get(`/payment/success?session_id=${sessionId}`);
    return response.data;
  }

  // 사용자 구독 정보 조회
  async getUserSubscription(): Promise<SubscriptionData> {
    const response = await apiService.api.get('/payment/subscription');
    return response.data.data;
  }

  // 구독 취소
  async cancelSubscription(immediate: boolean = false): Promise<any> {
    const response = await apiService.api.post('/payment/cancel', { immediate });
    return response.data;
  }

  // 결제 내역 조회
  async getPaymentHistory(): Promise<Payment[]> {
    const response = await apiService.api.get('/payment/payments');
    return response.data.data;
  }
}

export const paymentApi = new PaymentApiService();