// 파일: ~/HealthcareAI/frontend/src/pages/admin/Billing.tsx
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { SubscriptionPlans } from '@/components/Subscription/SubscriptionPlans';
import { paymentAPI } from '@/lib/payment';

export const BillingPage: React.FC = () => {
  const { data: subscriptionStatus } = useQuery({
    queryKey: ['subscription-status'],
    queryFn: () => paymentAPI.getSubscriptionStatus('current_user'),
  });

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Billing & Subscription</h1>
        <p className="text-gray-600">Manage your subscription and billing information</p>
      </div>

      {/* Current Subscription Status */}
      {subscriptionStatus && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Current Subscription
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <span className="text-sm text-gray-500">Plan</span>
              <p className="font-medium">{subscriptionStatus.plan_name}</p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Status</span>
              <p className={`font-medium ${
                subscriptionStatus.status === 'active' 
                  ? 'text-green-600' 
                  : 'text-red-600'
              }`}>
                {subscriptionStatus.status}
              </p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Next Billing</span>
              <p className="font-medium">{subscriptionStatus.next_billing_date}</p>
            </div>
          </div>
        </div>
      )}

      {/* Subscription Plans */}
      <SubscriptionPlans />
    </div>
  );
};