// 파일: ~/HealthcareAI/frontend/src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { AdminLayout } from '@/components/Layout/AdminLayout';
import { AdminDashboard } from '@/pages/admin/Dashboard';
import { UsersPage } from '@/pages/admin/Users';
import { UserDetailPage } from '@/pages/admin/UserDetail';
import { AnalyticsPage } from '@/pages/admin/Analytics';
import { ChallengesPage } from '@/pages/admin/Challenges';
import { SettingsPage } from '@/pages/admin/Settings';
import { LoginPage } from '@/pages/auth/Login';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import './main.css';

// React Query 클라이언트 설정
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5분
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="App">
          <Routes>
            {/* Public Routes */}
            <Route path="/login" element={<LoginPage />} />
            
            {/* Protected Admin Routes */}
            <Route
              path="/admin"
              element={
                <ProtectedRoute>
                  <AdminLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<AdminDashboard />} />
              <Route path="users" element={<UsersPage />} />
              <Route path="users/:userId" element={<UserDetailPage />} />
              <Route path="analytics" element={<AnalyticsPage />} />
              <Route path="challenges" element={<ChallengesPage />} />
              <Route path="settings" element={<SettingsPage />} />
            </Route>

            {/* Redirect root to admin */}
            <Route path="/" element={<Navigate to="/admin" replace />} />
          </Routes>
          
          {/* Toast notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
            }}
          />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;

// 파일: ~/HealthcareAI/frontend/src/components/auth/ProtectedRoute.tsx
import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { authAPI } from '@/lib/api';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('access_token');
      
      if (!token) {
        setIsAuthenticated(false);
        setIsLoading(false);
        return;
      }

      try {
        await authAPI.getCurrentUser();
        setIsAuthenticated(true);
      } catch (error) {
        localStorage.removeItem('access_token');
        setIsAuthenticated(false);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};