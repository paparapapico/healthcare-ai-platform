// 파일: ~/HealthcareAI/frontend/src/pages/admin/Dashboard.tsx
import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { StatsCard } from '@/components/Dashboard/StatsCard';
import { RealtimeChart } from '@/components/Dashboard/RealtimeChart';
import { dashboardAPI } from '@/lib/api';
import {
  UsersIcon,
  ChartBarIcon,
  ClockIcon,
  FireIcon,
} from '@heroicons/react/24/outline';

export const AdminDashboard: React.FC = () => {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: dashboardAPI.getStats,
    refetchInterval: 30000, // 30초마다 새로고침
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600">Healthcare AI Platform Overview</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Users"
          value={stats?.total_users || 0}
          icon={<UsersIcon />}
          color="blue"
        />
        <StatsCard
          title="Active Today"
          value={stats?.active_users_today || 0}
          icon={<ChartBarIcon />}
          color="green"
        />
        <StatsCard
          title="Total Workouts"
          value={stats?.total_workouts || 0}
          icon={<FireIcon />}
          color="yellow"
        />
        <StatsCard
          title="Avg Session"
          value={`${Math.round(stats?.avg_session_duration || 0)}m`}
          icon={<ClockIcon />}
          color="red"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RealtimeChart />
        
        {/* Top Exercises */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Popular Exercises
          </h3>
          <div className="space-y-3">
            {stats?.top_exercises?.map((exercise, index) => (
              <div key={exercise.exercise_type} className="flex items-center justify-between">
                <div className="flex items-center">
                  <span className="text-sm font-medium text-gray-600 w-8">
                    #{index + 1}
                  </span>
                  <span className="text-sm font-medium text-gray-900 capitalize">
                    {exercise.exercise_type.replace('_', ' ')}
                  </span>
                </div>
                <span className="text-sm text-gray-500">
                  {exercise.count} sessions
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};