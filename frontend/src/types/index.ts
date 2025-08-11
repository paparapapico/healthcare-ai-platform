export interface User {
  id: string;
  email: string;
  name: string;  // full_name 대신 name
  role: 'admin' | 'user';
  profile?: {
    age?: number;
    height?: number;
    weight?: number;
    fitness_goal?: string;
  };
  created_at: string;
  last_active?: string;
}

export interface Exercise {
  id: string;
  user_id: string;
  type: string;
  duration: number;
  calories_burned: number;
  pose_accuracy?: number;
  created_at: string;
}

export interface HealthData {
  id: string;
  user_id: string;
  date: string;
  steps?: number;
  calories?: number;
  sleep_hours?: number;
  water_intake?: number;
  weight?: number;
}

export interface Challenge {
  id: string;
  title: string;
  description: string;
  type: 'steps' | 'exercise' | 'nutrition';
  target: number;
  start_date: string;
  end_date: string;
  participants: number;
  is_active: boolean;
}

export interface SystemMetrics {
  active_users: number;
  total_exercises: number;
  server_status: 'healthy' | 'warning' | 'error';
  response_time: number;
  memory_usage: number;
  cpu_usage: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  error?: string;
}

export interface DashboardStats {
  totalUsers: number;
  activeUsers: number;
  totalExercises: number;
  todayExercises: number;
  avgSessionDuration: number;
  userGrowth: ChartData[];
  exerciseTypes: ChartData[];
  dailyActivity: ChartData[];
}

export interface ChartData {
  name: string;
  value: number;
  date?: string;
}