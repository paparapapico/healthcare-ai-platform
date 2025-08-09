// 파일: ~/HealthcareAI/frontend/src/lib/payment.ts
import { loadStripe } from '@stripe/stripe-js';
import type { Stripe } from '@stripe/stripe-js';

let stripePromise: Promise<Stripe | null>;

const getStripe = () => {
  if (!stripePromise) {
    const key = import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY;
    if (!key) {
      console.warn('Stripe publishable key not found');
      return Promise.resolve(null);
    }
    stripePromise = loadStripe(key);
  }
  return stripePromise;
};

export interface PaymentIntentData {
  amount: number;
  currency: string;
  customer_id?: string;
  metadata?: Record<string, string>;
}

export interface SubscriptionData {
  price_id: string;
  customer_id: string;
  trial_days?: number;
}

// Payment API functions
export const paymentAPI = {
  createPaymentIntent: async (data: PaymentIntentData) => {
    const response = await fetch('/api/payments/create-intent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      },
      body: JSON.stringify(data)
    });
    return response.json();
  },

  createSubscription: async (data: SubscriptionData) => {
    const response = await fetch('/api/payments/create-subscription', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      },
      body: JSON.stringify(data)
    });
    return response.json();
  },

  getSubscriptionStatus: async (customer_id: string) => {
    const response = await fetch(`/api/payments/subscription/${customer_id}`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    });
    return response.json();
  },

  cancelSubscription: async (subscription_id: string) => {
    const response = await fetch(`/api/payments/cancel/${subscription_id}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    });
    return response.json();
  }
};

export default getStripe;