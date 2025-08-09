// 파일: ~/HealthcareAI/frontend/src/pages/admin/Analytics.tsx
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { workoutsAPI } from '@/lib/api';
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell,
  Legend 
} from 'recharts';

// 타입 정의
interface WeeklyData {
  day: string;
  workouts: number;
  calories: number;
}

interface ExerciseData {
  name: string;
  value: number;
}

interface WorkoutStats {
  weekly_data?: WeeklyData[];
  exercise_distribution?: ExerciseData[];
  avg_duration?: number;
  avg_accuracy?: number;
  retention_rate?: number;
  daily_active_users?: number;
}

interface TooltipProps {
  active?: boolean;
  payload?: Array<{
    color: string;
    dataKey: string;
    value: number;
  }>;
  label?: string;
}

export const AnalyticsPage: React.FC = () => {
  const { data: stats, isLoading, error } = useQuery<WorkoutStats>({
    queryKey: ['workout-stats'],
    queryFn: () => workoutsAPI.getWorkoutStats('30d'),
    refetchInterval: 5 * 60 * 1000, // 5분마다 자동 새로고침
  });

  // 로딩 상태
  if (isLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
          <p className="text-gray-600">Detailed insights and performance metrics</p>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      </div>
    );
  }

  // 에러 상태
  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
          <p className="text-gray-600">Detailed insights and performance metrics</p>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="text-red-500 text-lg font-medium mb-2">
              Failed to load analytics data
            </div>
            <p className="text-gray-500">Please try refreshing the page</p>
          </div>
        </div>
      </div>
    );
  }

  // 기본 데이터와 실제 데이터 병합
  const weeklyData: WeeklyData[] = stats?.weekly_data || [
    { day: 'Mon', workouts: 45, calories: 1200 },
    { day: 'Tue', workouts: 52, calories: 1400 },
    { day: 'Wed', workouts: 38, calories: 1100 },
    { day: 'Thu', workouts: 61, calories: 1600 },
    { day: 'Fri', workouts: 55, calories: 1500 },
    { day: 'Sat', workouts: 67, calories: 1800 },
    { day: 'Sun', workouts: 43, calories: 1200 },
  ];

  const exerciseData: ExerciseData[] = stats?.exercise_distribution || [
    { name: 'Push-ups', value: 35 },
    { name: 'Squats', value: 28 },
    { name: 'Planks', value: 20 },
    { name: 'Jumping Jacks', value: 17 },
  ];

  // 키 메트릭 데이터
  const keyMetrics = {
    avgDuration: stats?.avg_duration || 32,
    avgAccuracy: stats?.avg_accuracy || 87,
    retentionRate: stats?.retention_rate || 74,
    dailyActiveUsers: stats?.daily_active_users || 243,
  };
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  // 커스텀 툴팁 컴포넌트
  const CustomTooltip: React.FC<TooltipProps> = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-medium text-gray-900">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {`${entry.dataKey}: ${entry.value}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-600">Detailed insights and performance metrics</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Workouts</p>
              <p className="text-2xl font-bold text-gray-900">
                {weeklyData.reduce((sum, item) => sum + item.workouts, 0)}
              </p>
            </div>
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
              <span className="text-blue-600 text-lg">💪</span>
            </div>
          </div>
        </div>
        
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Calories</p>
              <p className="text-2xl font-bold text-gray-900">
                {weeklyData.reduce((sum, item) => sum + item.calories, 0).toLocaleString()}
              </p>
            </div>
            <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
              <span className="text-green-600 text-lg">🔥</span>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg. Accuracy</p>
              <p className="text-2xl font-bold text-gray-900">{keyMetrics.avgAccuracy}%</p>
            </div>
            <div className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center">
              <span className="text-yellow-600 text-lg">🎯</span>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Users</p>
              <p className="text-2xl font-bold text-gray-900">{keyMetrics.dailyActiveUsers}</p>
            </div>
            <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
              <span className="text-purple-600 text-lg">👥</span>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weekly Workouts */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Weekly Workout Trends
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={weeklyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="day" 
                  tick={{ fontSize: 12 }}
                  axisLine={{ stroke: '#e0e0e0' }}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  axisLine={{ stroke: '#e0e0e0' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="workouts" 
                  stroke="#3b82f6" 
                  strokeWidth={3}
                  dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, stroke: '#3b82f6', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Exercise Distribution */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Exercise Distribution
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={exerciseData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => 
                    `${name} ${percent ? (percent * 100).toFixed(0) : 0}%`
                  }
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {exerciseData.map((_entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={COLORS[index % COLORS.length]} 
                    />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend 
                  verticalAlign="bottom" 
                  height={36}
                  iconType="circle"
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Calories Burned */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Daily Calories Burned
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weeklyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="day" 
                  tick={{ fontSize: 12 }}
                  axisLine={{ stroke: '#e0e0e0' }}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  axisLine={{ stroke: '#e0e0e0' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar 
                  dataKey="calories" 
                  fill="#10b981" 
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Key Performance Metrics
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-600">
                Avg. Workout Duration
              </span>
              <span className="text-lg font-semibold text-gray-900">
                {keyMetrics.avgDuration} min
              </span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-600">
                Avg. Pose Accuracy
              </span>
              <span className="text-lg font-semibold text-gray-900">
                {keyMetrics.avgAccuracy}%
              </span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-600">
                User Retention (7d)
              </span>
              <span className="text-lg font-semibold text-gray-900">
                {keyMetrics.retentionRate}%
              </span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-600">
                Daily Active Users
              </span>
              <span className="text-lg font-semibold text-gray-900">
                {keyMetrics.dailyActiveUsers.toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};