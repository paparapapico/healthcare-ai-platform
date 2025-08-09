// 파일: ~/HealthcareAI/frontend/src/components/Subscription/SubscriptionPlans.tsx
import React, { useState } from 'react';
import { PaymentForm } from '@/components/Layout/Payment/PaymentForm';
import { CheckIcon } from '@heroicons/react/24/outline';

interface Plan {
  id: string;
  name: string;
  price: number;
  interval: 'month' | 'year';
  features: string[];
  popular?: boolean;
}

const plans: Plan[] = [
  {
    id: 'basic',
    name: 'Basic',
    price: 9.99,
    interval: 'month',
    features: [
      '5 AI workout sessions per month',
      'Basic pose analysis',
      'Health tracking',
      'Email support'
    ]
  },
  {
    id: 'pro',
    name: 'Pro',
    price: 19.99,
    interval: 'month',
    features: [
      'Unlimited AI workout sessions',
      'Advanced pose analysis',
      'Detailed health insights',
      'Social challenges',
      'Priority support',
      'Custom workout plans'
    ],
    popular: true
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: 49.99,
    interval: 'month',
    features: [
      'Everything in Pro',
      'Team management',
      'Custom integrations',
      'Advanced analytics',
      'Dedicated support',
      'White-label options'
    ]
  }
];

export const SubscriptionPlans: React.FC = () => {
  const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);
  const [showPayment, setShowPayment] = useState(false);

  const handleSelectPlan = (plan: Plan) => {
    setSelectedPlan(plan);
    setShowPayment(true);
  };

  const handlePaymentSuccess = () => {
    setShowPayment(false);
    setSelectedPlan(null);
    // Redirect to success page or update UI
  };

  const handlePaymentCancel = () => {
    setShowPayment(false);
    setSelectedPlan(null);
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900">Choose Your Plan</h2>
        <p className="mt-2 text-gray-600">
          Select the perfect plan for your fitness journey
        </p>
      </div>

      {!showPayment ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {plans.map((plan) => (
            <div
              key={plan.id}
              className={`card p-6 relative ${
                plan.popular ? 'ring-2 ring-primary-500' : ''
              }`}
            >
              {plan.popular && (
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <span className="bg-primary-500 text-white px-3 py-1 text-sm rounded-full">
                    Most Popular
                  </span>
                </div>
              )}

              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">{plan.name}</h3>
                <div className="mt-2">
                  <span className="text-3xl font-bold">${plan.price}</span>
                  <span className="text-gray-500">/{plan.interval}</span>
                </div>
              </div>

              <ul className="space-y-3 mb-6">
                {plan.features.map((feature, index) => (
                  <li key={index} className="flex items-center">
                    <CheckIcon className="w-5 h-5 text-green-500 mr-3" />
                    <span className="text-sm text-gray-700">{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => handleSelectPlan(plan)}
                className={`w-full py-2 px-4 rounded-md font-medium ${
                  plan.popular
                    ? 'bg-primary-600 text-white hover:bg-primary-700'
                    : 'bg-gray-200 text-gray-900 hover:bg-gray-300'
                }`}
              >
                Select Plan
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="max-w-md mx-auto">
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4">
              Complete Your Subscription
            </h3>
            <div className="mb-4 p-4 bg-gray-50 rounded-md">
              <div className="flex justify-between">
                <span>Plan: {selectedPlan?.name}</span>
                <span>${selectedPlan?.price}/{selectedPlan?.interval}</span>
              </div>
            </div>
            <PaymentForm
              amount={selectedPlan?.price || 0}
              onSuccess={handlePaymentSuccess}
              onCancel={handlePaymentCancel}
            />
          </div>
        </div>
      )}
    </div>
  );
};
