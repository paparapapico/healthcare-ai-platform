import { apiService } from './api';

export const paymentApi = {
  getSubscriptionPlans: () => apiService.getSubscriptionPlans().then(res => res.data),
  createCheckoutSession: (planId: string) => apiService.createCheckoutSession(planId).then(res => res.data),
  handlePaymentSuccess: (sessionId: string) => apiService.handlePaymentSuccess(sessionId),
  getUserSubscription: () => apiService.getUserSubscription().then(res => res.data),
  cancelSubscription: (immediate?: boolean) => apiService.cancelSubscription(immediate),
  getPaymentHistory: () => apiService.getPaymentHistory().then(res => res.data),
};