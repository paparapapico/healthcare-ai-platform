// 파일: ~/HealthcareAI/frontend/src/pages/admin/Analytics.tsx
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { workoutsAPI } from '@/lib/api';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export const AnalyticsPage: React.FC = () => {
  const { data: stats } = useQuery({
    queryKey: ['workout-stats'],
    queryFn: () => workoutsAPI.getWorkoutStats('30d'),
  });

  // Mock data for demonstration
  const weeklyData = [
    { day: 'Mon', workouts: 45, calories: 1200 },
    { day: 'Tue', workouts: 52, calories: 1400 },
    { day: 'Wed', workouts: 38, calories: 1100 },
    { day: 'Thu', workouts: 61, calories: 1600 },
    { day: 'Fri', workouts: 55, calories: 1500 },
    { day: 'Sat', workouts: 67, calories: 1800 },
    { day: 'Sun', workouts: 43, calories: 1200 },
  ];

  const exerciseData = [
    { name: 'Push-ups', value: 35 },
    { name: 'Squats', value: 28 },
    { name: 'Planks', value: 20 },
    { name: 'Jumping Jacks', value: 17 },
  ];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-600">Detailed insights and performance metrics</p>
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
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="workouts" stroke="#3b82f6" strokeWidth={2} />
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
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {exerciseData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
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
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="calories" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Key Metrics
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Avg. Workout Duration</span>
              <span className="text-lg font-semibold">32 min</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Avg. Pose Accuracy</span>
              <span className="text-lg font-semibold">87%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">User Retention (7d)</span>
              <span className="text-lg font-semibold">74%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Daily Active Users</span>
              <span className="text-lg font-semibold">243</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};