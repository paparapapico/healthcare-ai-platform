// 파일: ~/HealthcareAI/frontend/src/App.tsx
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
import { LoginPage } from '@/pages/admin/auth/Login';
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