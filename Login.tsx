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
    setDebugInfo(` 로그인 시도: ${data.email}`);
    
    try {
      console.log(' LOGIN ATTEMPT:', data);
      
      // FormData 생성 (올바른 방식)
      const formData = new FormData();
      formData.append('username', data.email);
      formData.append('password', data.password);
      
      console.log(' SENDING:', {
        username: data.email,
        password: data.password,
        url: 'http://localhost:8000/api/auth/login'
      });
      
      const response = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        body: formData,
        // Content-Type을 명시적으로 설정하지 않음 (FormData 자동 설정)
      });
      
      console.log(' RESPONSE STATUS:', response.status);
      
      const responseText = await response.text();
      console.log(' RESPONSE TEXT:', responseText);
      
      if (response.ok) {
        const result = JSON.parse(responseText);
        console.log(' LOGIN SUCCESS:', result);
        
        localStorage.setItem('access_token', result.access_token);
        setIsAuthenticated(true);
        setDebugInfo(' 로그인 성공!');
      } else {
        console.error(' LOGIN FAILED:', response.status, responseText);
        setError(` 로그인 실패 (${response.status}): ${responseText}`);
        setDebugInfo(` 실패: ${response.status}`);
      }
    } catch (error) {
      console.error(' NETWORK ERROR:', error);
      setError(` 네트워크 오류: ${error.message}`);
      setDebugInfo(` 오류: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 직접 API 테스트 함수
  const testAPI = async () => {
    try {
      setDebugInfo(' API 테스트 중...');
      
      const formData = new FormData();
      formData.append('username', 'admin@healthcare.ai');
      formData.append('password', 'admin123');
      
      const response = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        body: formData,
      });
      
      const text = await response.text();
      setDebugInfo(` 테스트 결과: ${response.status} - ${text}`);
      
    } catch (error) {
      setDebugInfo(` 테스트 실패: ${error.message}`);
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
            디버깅 모드 - 로그인 테스트
          </p>
        </div>
        
        {debugInfo && (
          <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
            <p className="text-blue-600 text-sm font-mono">{debugInfo}</p>
          </div>
        )}
        
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-3">
            <p className="text-red-600 text-sm font-mono">{error}</p>
          </div>
        )}
        
        <form className="mt-8 space-y-6" onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Email
              </label>
              <input
                {...register('email', { required: true })}
                type="email"
                className="form-input mt-1"
                defaultValue="admin@healthcare.ai"
                placeholder="admin@healthcare.ai"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <input
                {...register('password', { required: true })}
                type="password"
                className="form-input mt-1"
                defaultValue="admin123"
                placeholder="admin123"
              />
            </div>
          </div>

          <div className="space-y-3">
            <button
              type="submit"
              disabled={isLoading}
              className="w-full btn btn-primary py-3"
            >
              {isLoading ? '로그인 중...' : ' 로그인'}
            </button>
            
            <button
              type="button"
              onClick={testAPI}
              className="w-full btn btn-secondary py-2"
            >
               API 직접 테스트
            </button>
          </div>
        </form>

        <div className="text-center text-sm text-gray-600 bg-gray-100 p-3 rounded">
          <p><strong> 테스트 계정:</strong></p>
          <p> admin@healthcare.ai</p>
          <p> admin123</p>
        </div>
      </div>
    </div>
  );
};
