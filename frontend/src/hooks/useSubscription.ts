# 구독 상태 확인 Hook
# frontend/src/hooks/useSubscription.ts
import { useState, useEffect } from 'react';
import { paymentApi } from '../services/paymentApi';
import { SubscriptionData } from '../types/payment';

export const useSubscription = () => {
  const [subscriptionData, setSubscriptionData] = useState<SubscriptionData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSubscription();
  }, []);

  const loadSubscription = async () => {
    try {
      setIsLoading(true);
      const data = await paymentApi.getUserSubscription();
      setSubscriptionData(data);
      setError(null);
    } catch (err) {
      setError('구독 정보를 불러오는 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  const hasActiveSubscription = () => {
    return subscriptionData?.subscription?.status === 'active';
  };

  const hasFeature = (feature: string) => {
    if (!hasActiveSubscription()) return false;
    
    const plan = subscriptionData?.plan;
    if (!plan) return false;

    switch (feature) {
      case 'premium_content':
        return plan.premium_content_access;
      case 'priority_support':
        return plan.priority_support;
      case 'unlimited_exercises':
        return plan.max_exercises_per_day === -1;
      case 'advanced_ai':
        return plan.max_ai_analysis > 10;
      default:
        return false;
    }
  };

  const getCurrentPlan = () => {
    return subscriptionData?.plan?.name || 'Free';
  };

  const getUsageLimits = () => {
    return subscriptionData?.limits || {
      max_exercises_per_day: 3,
      max_ai_analysis: 1
    };
  };

  return {
    subscriptionData,
    isLoading,
    error,
    hasActiveSubscription,
    hasFeature,
    getCurrentPlan,
    getUsageLimits,
    refetch: loadSubscription
  };
};