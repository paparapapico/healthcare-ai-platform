// 파일: ~/HealthcareAI/frontend/src/pages/auth/Login.tsx
import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { Navigate } from 'react-router-dom';
import { authAPI } from '@/lib/api';
import toast from 'react-hot-toast';

interface LoginFormData {
  email: string;
  password: string;
}

export const LoginPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(
    !!localStorage.getItem('access_token')
  );

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>();

  if (isAuthenticated) {
    return <Navigate to="/admin" replace />;
  }

  const onSubmit = async (data: LoginFormData) => {
    setIsLoading(true);
    try {
      const response = await authAPI.login(data.email, data.password);
      localStorage.setItem('access_token', response.access_token);
      setIsAuthenticated(true);
      toast.success('Login successful!');
    } catch (error) {
      toast.error('Invalid credentials');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Healthcare AI Admin
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Sign in to your admin account
          </p>
        </div>
        
        <form className="mt-8 space-y-6" onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                Email address
              </label>
              <input
                {...register('email', {
                  required: 'Email is required',
                  pattern: {
                    value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                    message: 'Invalid email address',
                  },
                })}
                type="email"
                className="form-input mt-1"
                placeholder="admin@healthcare.ai"
              />
              {errors.email && (
                <p className="mt-1 text-sm text-red-600">{errors.email.message}</p>
              )}
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <input
                {...register('password', {
                  required: 'Password is required',
                  minLength: {
                    value: 6,
                    message: 'Password must be at least 6 characters',
                  },
                })}
                type="password"
                className="form-input mt-1"
                placeholder="••••••••"
              />
              {errors.password && (
                <p className="mt-1 text-sm text-red-600">{errors.password.message}</p>
              )}
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full btn btn-primary py-3 disabled:opacity-50"
            >
              {isLoading ? 'Signing in...' : 'Sign in'}
            </button>
          </div>
        </form>

        <div className="text-center text-sm text-gray-600">
          <p>Demo credentials:</p>
          <p>Email: admin@healthcare.ai</p>
          <p>Password: admin123</p>
        </div>
      </div>
    </div>
  );
};