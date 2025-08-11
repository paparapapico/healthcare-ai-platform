export interface User {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  profile?: UserProfile;
}

export interface UserProfile {
  height: number;
  weight: number;
  age: number;
  gender: 'male' | 'female' | 'other';
  activity_level: 'sedentary' | 'lightly_active' | 'moderately_active' | 'very_active';
}

export interface WorkoutSession {
  id: string;
  user_id: string;
  exercise_type: string;
  duration: number;
  calories_burned: number;
  pose_accuracy: number;
  created_at: string;
  ai_feedback?: string;
}

export interface DashboardStats {
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

export interface Challenge {
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

export interface HealthData {
  id: string;
  user_id: string;
  data_type: 'nutrition' | 'sleep' | 'hydration';
  value: number;
  unit: string;
  recorded_at: string;
}
