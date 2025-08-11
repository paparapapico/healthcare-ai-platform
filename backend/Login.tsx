import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { Navigate } from 'react-router-dom';

interface LoginFormData {
  email: string;
  password: string;
}

export const LoginPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [debugInfo, setDebugInfo] = useState('');
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
    setError('');
    setDebugInfo('');
    
    try {
      console.log(' 로그인 시도:', data);
      
      // FormData 생성
      const formData = new FormData();
      formData.append('username', data.email);
      formData.append('password', data.password);
      
      console.log(' 전송 데이터:', {
        username: data.email,
        password: data.password
      });
      
      setDebugInfo(`전송 중: ${data.email}`);
      
      const response = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        body: formData,
        // Content-Type 헤더 제거 (브라우저가 자동 설정)
      });
      
      console.log(' 응답 상태:', response.status);
      console.log(' 응답 헤더:', response.headers);
      
      const responseText = await response.text();
      console.log(' 응답 내용 (텍스트):', responseText);
      
      if (response.ok) {
        const result = JSON.parse(responseText);
        console.log(' 로그인 성공:', result);
        
        localStorage.setItem('access_token', result.access_token);
        setIsAuthenticated(true);
        setDebugInfo('로그인 성공!');
      } else {
        console.error(' 로그인 실패:', response.status, responseText);
        setError(`로그인 실패: ${response.status} - ${responseText}`);
        setDebugInfo(`실패: ${response.status}`);
      }
    } catch (error) {
      console.error(' 네트워크 오류:', error);
      setError(`네트워크 오류: ${error.message}`);
      setDebugInfo(`오류: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4">
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
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-3">
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          )}
          
          {debugInfo && (
            <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
              <p className="text-blue-600 text-sm">디버그: {debugInfo}</p>
            </div>
          )}
          
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
                defaultValue="admin@healthcare.ai"
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
                })}
                type="password"
                className="form-input mt-1"
                placeholder="admin123"
                defaultValue="admin123"
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
          <p> 계정 정보:</p>
          <p>Email: admin@healthcare.ai</p>
          <p>Password: admin123</p>
        </div>
        
        {/* 디버그 테스트 버튼 */}
        <div className="text-center">
          <button
            onClick={async () => {
              try {
                const response = await fetch('http://localhost:8000/health');
                const data = await response.json();
                setDebugInfo(`백엔드 상태: ${data.status}`);
              } catch (error) {
                setDebugInfo(`백엔드 연결 실패: ${error.message}`);
              }
            }}
            className="text-blue-600 text-sm underline"
          >
             백엔드 연결 테스트
          </button>
        </div>
      </div>
    </div>
  );
};
