// 파일: ~/HealthcareAI/frontend/src/pages/admin/UserDetail.tsx
import React from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { usersAPI, workoutsAPI, healthAPI } from '@/lib/api';
import { format } from 'date-fns';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const UserDetailPage: React.FC = () => {
  const { userId } = useParams<{ userId: string }>();

  const { data: user, isLoading: userLoading } = useQuery({
    queryKey: ['user', userId],
    queryFn: () => usersAPI.getUserById(userId!),
    enabled: !!userId,
  });

  const { data: workouts = [] } = useQuery({
    queryKey: ['user-workouts', userId],
    queryFn: () => workoutsAPI.getUserWorkouts(userId!),
    enabled: !!userId,
  });

  const { data: healthData = [] } = useQuery({
    queryKey: ['user-health', userId],
    queryFn: () => healthAPI.getHealthData(userId!),
    enabled: !!userId,
  });

  if (userLoading || !user) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  // 운동 데이터 가공
  const workoutsByType = workouts.reduce((acc, workout) => {
    acc[workout.exercise_type] = (acc[workout.exercise_type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const chartData = Object.entries(workoutsByType).map(([type, count]) => ({
    exercise: type.replace('_', ' '),
    count,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">{user.full_name}</h1>
        <p className="text-gray-600">{user.email}</p>
      </div>

      {/* User Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Profile</h3>
          <div className="space-y-2">
            <div>
              <span className="text-sm text-gray-500">Status:</span>
              <span className={`ml-2 px-2 py-1 text-xs rounded ${
                user.is_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {user.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
            <div>
              <span className="text-sm text-gray-500">Role:</span>
              <span className="ml-2 text-sm font-medium">
                {user.is_superuser ? 'Admin' : 'User'}
              </span>
            </div>
            <div>
              <span className="text-sm text-gray-500">Joined:</span>
              <span className="ml-2 text-sm">
                {format(new Date(user.created_at), 'MMM dd, yyyy')}
              </span>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Workout Stats</h3>
          <div className="space-y-2">
            <div>
              <span className="text-sm text-gray-500">Total Sessions:</span>
              <span className="ml-2 text-lg font-bold">{workouts.length}</span>
            </div>
            <div>
              <span className="text-sm text-gray-500">Total Calories:</span>
              <span className="ml-2 text-lg font-bold">
                {workouts.reduce((sum, w) => sum + w.calories_burned, 0)}
              </span>
            </div>
            <div>
              <span className="text-sm text-gray-500">Avg Accuracy:</span>
              <span className="ml-2 text-lg font-bold">
                {workouts.length > 0 
                  ? Math.round(workouts.reduce((sum, w) => sum + w.pose_accuracy, 0) / workouts.length)
                  : 0}%
              </span>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Health Data</h3>
          <div className="space-y-2">
            <div>
              <span className="text-sm text-gray-500">Records:</span>
              <span className="ml-2 text-lg font-bold">{healthData.length}</span>
            </div>
            <div>
              <span className="text-sm text-gray-500">Last Update:</span>
              <span className="ml-2 text-sm">
                {healthData.length > 0
                  ? format(new Date(healthData[0].recorded_at), 'MMM dd')
                  : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Workout Chart */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Exercise Distribution</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="exercise" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Workouts */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Workouts</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Exercise
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Duration
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Calories
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Date
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {workouts.slice(0, 10).map((workout) => (
                <tr key={workout.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {workout.exercise_type.replace('_', ' ')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {Math.round(workout.duration / 60)}m
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {workout.calories_burned}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {Math.round(workout.pose_accuracy)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {format(new Date(workout.created_at), 'MMM dd, HH:mm')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};