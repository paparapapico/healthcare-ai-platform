import React from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { AdminDashboard } from '../admin/Dashboard';
import { UserDashboard } from './UserDashboard';

export const DashboardPage: React.FC = () => {
  const { isAdmin } = useAuth();
  
  return isAdmin ? <AdminDashboard /> : <UserDashboard />;
};