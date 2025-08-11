export interface User {
  id: string;
  email: string;
  name: string;
  full_name?: string; // 호환성을 위해 추가
  role: 'admin' | 'user';
  is_active?: boolean;
  is_superuser?: boolean;
  profile?: {
    age?: number;
    height?: number;
    weight?: number;
    fitness_goal?: string;
  };
  created_at: string;
  last_active?: string;
}

// 나머지 타입들은 기존과 동일