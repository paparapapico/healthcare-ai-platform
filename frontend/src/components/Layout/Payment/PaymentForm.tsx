// 파일: ~/HealthcareAI/frontend/src/components/Payment/PaymentForm.tsx
import React, { useState } from 'react';
import {
  CardElement,
  Elements,
  useStripe,
  useElements,
} from '@stripe/react-stripe-js';
import getStripe, { paymentAPI } from '@/lib/payment';
import toast from 'react-hot-toast';

interface PaymentFormProps {
  amount: number;
  onSuccess: () => void;
  onCancel: () => void;
}

const CheckoutForm: React.FC<PaymentFormProps> = ({ amount, onSuccess, onCancel }) => {
  const stripe = useStripe();
  const elements = useElements();
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!stripe || !elements) {
      return;
    }

    setLoading(true);

    try {
      // Create payment intent
      const { client_secret } = await paymentAPI.createPaymentIntent({
        amount: amount * 100, // Convert to cents
        currency: 'usd',
        metadata: {
          source: 'admin_dashboard'
        }
      });

      const cardElement = elements.getElement(CardElement);
      if (!cardElement) {
        throw new Error('Card element not found');
      }

      const { error, paymentIntent } = await stripe.confirmCardPayment(client_secret, {
        payment_method: {
          card: cardElement,
        },
      });

      if (error) {
        toast.error(error.message || 'Payment failed');
      } else if (paymentIntent.status === 'succeeded') {
        toast.success('Payment successful!');
        onSuccess();
      }
    } catch (error) {
      toast.error('Payment processing failed');
      console.error('Payment error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="p-4 border rounded-md">
        <CardElement
          options={{
            style: {
              base: {
                fontSize: '16px',
                color: '#424770',
                '::placeholder': {
                  color: '#aab7c4',
                },
              },
            },
          }}
        />
      </div>

      <div className="flex space-x-3">
        <button
          type="button"
          onClick={onCancel}
          className="btn btn-secondary flex-1"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!stripe || loading}
          className="btn btn-primary flex-1"
        >
          {loading ? 'Processing...' : `Pay $${amount}`}
        </button>
      </div>
    </form>
  );
};

export const PaymentForm: React.FC<PaymentFormProps> = (props) => {
  const stripePromise = getStripe();

  return (
    <Elements stripe={stripePromise}>
      <CheckoutForm {...props} />
    </Elements>
  );
};