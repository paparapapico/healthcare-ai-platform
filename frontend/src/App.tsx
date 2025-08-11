// frontend/src/App.tsx 완전 수정
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './contexts/AuthContext';
import { ProtectedRoute } from './components/common/ProtectedRoute';
import { Layout } from './components/common/Layout';

// Auth Pages
import { LoginPage } from './pages/auth/LoginPage';

// Dashboard Pages
import { DashboardPage } from './pages/dashboard/DashboardPage';

// Admin Pages
import { UsersPage } from './pages/admin/UsersPage';
import { ChallengesPage } from './pages/admin/ChallengesPage';
import { NotificationsPage } from './pages/admin/NotificationsPage';

// Other Pages
import { MonitoringPage } from './pages/monitoring/MonitoringPage';
import { AnalyticsPage } from './pages/analytics/AnalyticsPage';

// Payment Pages
import { SubscriptionPage } from './pages/subscription/SubscriptionPage';
import { PaymentSuccessPage } from './pages/payment/PaymentSuccessPage';
import { PaymentCancelPage } from './pages/payment/PaymentCancelPage';
import { PaymentHistoryPage } from './pages/payment/PaymentHistoryPage';

// MUI 테마 설정
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          borderRadius: 8,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 6,
        },
      },
    },
  },
});

// React Query 클라이언트
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <AuthProvider>
          <Router>
            <Routes>
              {/* 로그인 페이지 */}
              <Route path="/login" element={<LoginPage />} />
              
              {/* 보호된 라우트들 */}
              <Route
                path="/*"
                element={
                  <ProtectedRoute>
                    <Layout>
                      <Routes>
                        <Route path="/" element={<Navigate to="/dashboard" replace />} />
                        <Route path="/dashboard" element={<DashboardPage />} />
                        
                        {/* Admin Routes */}
                        <Route path="/users" element={<UsersPage />} />
                        <Route path="/challenges" element={<ChallengesPage />} />
                        <Route path="/notifications" element={<NotificationsPage />} />
                        
                        {/* Analytics & Monitoring */}
                        <Route path="/monitoring" element={<MonitoringPage />} />
                        <Route path="/analytics" element={<AnalyticsPage />} />
                        
                        {/* Payment & Subscription Routes */}
                        <Route path="/subscription" element={<SubscriptionPage />} />
                        <Route path="/payment/success" element={<PaymentSuccessPage />} />
                        <Route path="/payment/cancel" element={<PaymentCancelPage />} />
                        <Route path="/payment/history" element={<PaymentHistoryPage />} />
                      </Routes>
                    </Layout>
                  </ProtectedRoute>
                }
              />
            </Routes>
          </Router>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;