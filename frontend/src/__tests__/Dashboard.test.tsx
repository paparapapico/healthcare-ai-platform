// 파일: ~/HealthcareAI/frontend/src/__tests__/Dashboard.test.tsx
import { describe, it, expect, beforeEach, vi, type MockedFunction } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { AdminDashboard } from '@/pages/admin/Dashboard';
import { dashboardAPI } from '@/lib/api';

// Mock the API
vi.mock('@/lib/api');
const mockDashboardAPI = dashboardAPI as typeof dashboardAPI & {
  getStats: MockedFunction<typeof dashboardAPI.getStats>;
};

// Mock WebSocket service
vi.mock('@/lib/websocket', () => ({
  wsService: {
    connect: vi.fn(() => ({
      on: vi.fn(),
      emit: vi.fn(),
    })),
    disconnect: vi.fn(),
  },
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('AdminDashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDashboardAPI.getStats.mockResolvedValue({
      total_users: 150,
      active_users_today: 42,
      total_workouts: 1250,
      total_challenges: 8,
      avg_session_duration: 32,
      top_exercises: [
        { exercise_type: 'push_ups', count: 45 },
        { exercise_type: 'squats', count: 38 },
      ],
    });
  });

  it('renders dashboard with stats', async () => {
    render(<AdminDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('150')).toBeInTheDocument();
      expect(screen.getByText('42')).toBeInTheDocument();
      expect(screen.getByText('1250')).toBeInTheDocument();
    });
  });

  it('shows loading spinner initially', () => {
    mockDashboardAPI.getStats.mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve({
        total_users: 150,
        active_users_today: 42,
        total_workouts: 1250,
        total_challenges: 8,
        avg_session_duration: 32,
        top_exercises: [],
      }), 1000))
    );
    
    render(<AdminDashboard />, { wrapper: createWrapper() });
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('displays popular exercises', async () => {
    render(<AdminDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
        expect(screen.getByText('push ups')).toBeInTheDocument();
        expect(screen.getByText('squats')).toBeInTheDocument();
    });
  });
});