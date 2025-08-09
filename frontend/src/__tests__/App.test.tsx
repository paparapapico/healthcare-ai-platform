// 파일: ~/HealthcareAI/frontend/src/__tests__/App.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import App from '../App';

// Mock the auth API
vi.mock('@/lib/api', () => ({
  authAPI: {
    getCurrentUser: vi.fn(),
    login: vi.fn(),
    logout: vi.fn(),
  },
  usersAPI: {
    getUsers: vi.fn(),
    getUserById: vi.fn(),
    updateUser: vi.fn(),
    deleteUser: vi.fn(),
  },
  dashboardAPI: {
    getStats: vi.fn(),
    getRealtimeData: vi.fn(),
  },
  workoutsAPI: {
    getWorkouts: vi.fn(),
    getUserWorkouts: vi.fn(),
    getWorkoutStats: vi.fn(),
  },
  healthAPI: {
    getHealthData: vi.fn(),
    addHealthData: vi.fn(),
  },
  challengesAPI: {
    getChallenges: vi.fn(),
    createChallenge: vi.fn(),
    joinChallenge: vi.fn(),
  },
}));

// Mock WebSocket service
vi.mock('@/lib/websocket', () => ({
  wsService: {
    connect: vi.fn(),
    disconnect: vi.fn(),
  },
}));

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />);
    // The app should render the router
    expect(document.querySelector('.App')).toBeInTheDocument();
  });

  it('redirects to login when not authenticated', () => {
    // Mock localStorage to return null (not authenticated)
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: vi.fn(() => null),
        setItem: vi.fn(),
        removeItem: vi.fn(),
      },
      writable: true,
    });

    render(<App />);
    
    // Should redirect to login page
    expect(window.location.pathname).toBe('/');
  });
});