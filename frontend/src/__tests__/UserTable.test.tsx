// 파일: ~/HealthcareAI/frontend/src/__tests__/UserTable.test.tsx
import { describe, it, expect, beforeEach, vi, type MockedFunction } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { UserTable } from '@/components/Layout/Users/UserTable';
import { usersAPI } from '@/lib/api';

vi.mock('@/lib/api');
const mockUsersAPI = usersAPI as typeof usersAPI & {
  getUsers: MockedFunction<typeof usersAPI.getUsers>;
};

const mockUsers = [
  {
    id: '1',
    email: 'john@example.com',
    full_name: 'John Doe',
    is_active: true,
    is_superuser: false,
    created_at: '2024-01-01T00:00:00Z',
  },
  {
    id: '2',
    email: 'admin@example.com',
    full_name: 'Admin User',
    is_active: true,
    is_superuser: true,
    created_at: '2024-01-02T00:00:00Z',
  },
];

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { 
      queries: { retry: false },
      mutations: { retry: false }
    },
  });
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('UserTable', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUsersAPI.getUsers.mockResolvedValue(mockUsers);
  });

  it('renders user list', async () => {
    render(<UserTable />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.getByText('Admin User')).toBeInTheDocument();
    });
  });

  it('filters users based on search', async () => {
    render(<UserTable />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search users...');
    fireEvent.change(searchInput, { target: { value: 'admin' } });

    await waitFor(() => {
      expect(screen.getByText('Admin User')).toBeInTheDocument();
      expect(screen.queryByText('John Doe')).not.toBeInTheDocument();
    });
  });

  it('shows loading spinner initially', () => {
    mockUsersAPI.getUsers.mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve(mockUsers), 1000))
    );
    
    render(<UserTable />, { wrapper: createWrapper() });
    expect(document.querySelector('.animate-spin')).toBeInTheDocument();
  });
});