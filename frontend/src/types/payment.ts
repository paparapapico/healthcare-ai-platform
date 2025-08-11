export interface SubscriptionPlan {
  id: string;
  name: string;
  price: number;
  currency: string;
  interval: string;
  features: string;  // JSON string
  max_exercises_per_day: number;
  max_ai_analysis: number;
  premium_content_access: boolean;
  priority_support: boolean;
}

export interface UserSubscription {
  id: string;
  status: 'active' | 'canceled' | 'past_due' | 'unpaid';
  current_period_start: string;
  current_period_end: string;
  cancel_at_period_end: boolean;
}

export interface Payment {
  id: string;
  amount: number;
  currency: string;
  status: string;
  payment_method: string;
  description: string;
  created_at: string;
}

export interface SubscriptionData {
  subscription: UserSubscription | null;
  plan: SubscriptionPlan | null;
  limits: {
    max_exercises_per_day: number;
    max_ai_analysis: number;
  };
}