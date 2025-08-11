// 타입 정의
interface User {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

interface DashboardStats {
  total_users: number;
  active_users_today: number;
  total_workouts: number;
  total_challenges: number;
  avg_session_duration: number;
  top_exercises: Array<{
    exercise_type: string;
    count: number;
  }>;
}

interface WorkoutSession {
  id: string;
  user_id: string;
  exercise_type: string;
  duration: number;
  calories_burned: number;
  pose_accuracy: number;
  created_at: string;
}

interface Challenge {
  id: string;
  title: string;
  description: string;
  target_value: number;
  current_value: number;
  start_date: string;
  end_date: string;
  participants_count: number;
  is_active: boolean;
}

interface HealthData {
  id: string;
  user_id: string;
  data_type: string;
  value: number;
  unit: string;
  recorded_at: string;
}

// Auth API
export const authAPI = {
  login: async (email: string, password: string) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    
    const response = await fetch('http://localhost:8000/api/auth/login', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`Login failed: ${response.status}`);
    }
    
    return response.json();
  },
  
  getCurrentUser: async (): Promise<User> => {
    const response = await fetch('http://localhost:8000/api/auth/me', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    });
    return response.json();
  },
  
  logout: () => {
    localStorage.removeItem('access_token');
  }
};

// Users API
export const usersAPI = {
  getUsers: async (skip = 0, limit = 100): Promise<User[]> => {
    const response = await fetch(`http://localhost:8000/api/users/?skip=${skip}&limit=${limit}`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    });
    return response.json();
  },
  
  getUserById: async (userId: string): Promise<User> => {
    const response = await fetch(`http://localhost:8000/api/users/${userId}`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    });
    return response.json();
  }
};

// Workouts API
export const workoutsAPI = {
  getWorkouts: async (): Promise<WorkoutSession[]> => {
    const response = await fetch('http://localhost:8000/api/workouts/', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    });
    return response.json();
  },
  
  getUserWorkouts: async (userId: string): Promise<WorkoutSession[]> => {
    // Mock data for now
    return [
      {
        id: "1",
        user_id: userId,
        exercise_type: "push_ups",
        duration: 300,
        calories_burned: 50,
        pose_accuracy: 85,
        created_at: "2024-01-01T10:00:00Z"
      }
    ];
  },
  
  getWorkoutStats: async () => {
    // Mock data
    return {
      total_workouts: 1250,
      avg_duration: 32
    };
  }
};

// Health API
export const healthAPI = {
  getHealthData: async (userId: string): Promise<HealthData[]> => {
    // Mock data for now
    return [
      {
        id: "1",
        user_id: userId,
        data_type: "nutrition",
        value: 2000,
        unit: "calories",
        recorded_at: "2024-01-01T12:00:00Z"
      }
    ];
  }
};

// Challenges API
export const challengesAPI = {
  getChallenges: async (): Promise<Challenge[]> => {
    const response = await fetch('http://localhost:8000/api/challenges/', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    });
    return response.json();
  },
  
  createChallenge: async (challenge: any): Promise<Challenge> => {
    // Mock implementation
    return {
      id: "new",
      title: challenge.title,
      description: challenge.description,
      target_value: challenge.target_value,
      current_value: 0,
      start_date: challenge.start_date,
      end_date: challenge.end_date,
      participants_count: 0,
      is_active: true
    };
  }
};

// Dashboard API
export const dashboardAPI = {
  getStats: async (): Promise<DashboardStats> => {
    const response = await fetch('http://localhost:8000/api/dashboard/stats', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    });
    return response.json();
  }
};
