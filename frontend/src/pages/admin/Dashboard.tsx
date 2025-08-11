import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { dashboardAPI } from '@/lib/api';

export const AdminDashboard: React.FC = () => {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: dashboardAPI.getStats,
  });

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '300px'
      }}>
        <div style={{
          width: '48px',
          height: '48px',
          border: '4px solid #e5e7eb',
          borderTop: '4px solid #2563eb',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }}></div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1200px' }}>
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{
          fontSize: '28px',
          fontWeight: 'bold',
          color: '#111827',
          marginBottom: '8px'
        }}>
          Dashboard
        </h1>
        <p style={{ color: '#6b7280' }}>
          Healthcare AI Platform Overview
        </p>
      </div>

      {/* Stats Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: '24px',
        marginBottom: '32px'
      }}>
        <div style={{
          backgroundColor: 'white',
          padding: '24px',
          borderRadius: '8px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <p style={{ fontSize: '14px', color: '#6b7280', margin: '0 0 4px 0' }}>
                Total Users
              </p>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#111827', margin: 0 }}>
                {stats?.total_users || 0}
              </p>
            </div>
            <div style={{
              backgroundColor: '#2563eb',
              padding: '12px',
              borderRadius: '50%',
              fontSize: '20px'
            }}>
              
            </div>
          </div>
        </div>

        <div style={{
          backgroundColor: 'white',
          padding: '24px',
          borderRadius: '8px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <p style={{ fontSize: '14px', color: '#6b7280', margin: '0 0 4px 0' }}>
                Active Today
              </p>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#111827', margin: 0 }}>
                {stats?.active_users_today || 0}
              </p>
            </div>
            <div style={{
              backgroundColor: '#10b981',
              padding: '12px',
              borderRadius: '50%',
              fontSize: '20px'
            }}>
              
            </div>
          </div>
        </div>

        <div style={{
          backgroundColor: 'white',
          padding: '24px',
          borderRadius: '8px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <p style={{ fontSize: '14px', color: '#6b7280', margin: '0 0 4px 0' }}>
                Total Workouts
              </p>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#111827', margin: 0 }}>
                {stats?.total_workouts || 0}
              </p>
            </div>
            <div style={{
              backgroundColor: '#f59e0b',
              padding: '12px',
              borderRadius: '50%',
              fontSize: '20px'
            }}>
              
            </div>
          </div>
        </div>

        <div style={{
          backgroundColor: 'white',
          padding: '24px',
          borderRadius: '8px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <p style={{ fontSize: '14px', color: '#6b7280', margin: '0 0 4px 0' }}>
                Avg Session
              </p>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#111827', margin: 0 }}>
                {Math.round(stats?.avg_session_duration || 0)}m
              </p>
            </div>
            <div style={{
              backgroundColor: '#ef4444',
              padding: '12px',
              borderRadius: '50%',
              fontSize: '20px'
            }}>
              
            </div>
          </div>
        </div>
      </div>

      {/* Popular Exercises */}
      <div style={{
        backgroundColor: 'white',
        padding: '24px',
        borderRadius: '8px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        border: '1px solid #e5e7eb'
      }}>
        <h3 style={{
          fontSize: '18px',
          fontWeight: '600',
          color: '#111827',
          marginBottom: '16px'
        }}>
          Popular Exercises
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {stats?.top_exercises?.map((exercise, index) => (
            <div key={exercise.exercise_type} style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px 0',
              borderBottom: index < stats.top_exercises.length - 1 ? '1px solid #f3f4f6' : 'none'
            }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <span style={{
                  fontSize: '14px',
                  fontWeight: '500',
                  color: '#6b7280',
                  width: '32px'
                }}>
                  #{index + 1}
                </span>
                <span style={{
                  fontSize: '14px',
                  fontWeight: '500',
                  color: '#111827',
                  textTransform: 'capitalize'
                }}>
                  {exercise.exercise_type.replace('_', ' ')}
                </span>
              </div>
              <span style={{ fontSize: '14px', color: '#6b7280' }}>
                {exercise.count} sessions
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
