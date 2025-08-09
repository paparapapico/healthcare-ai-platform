// 파일: ~/HealthcareAI/frontend/src/components/Layout/AdminLayout.tsx
import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  UsersIcon,
  ChartBarIcon,
  CogIcon,
  BellIcon,
  LogoutIcon,
} from '@heroicons/react/24/outline';
import { authAPI } from '@/lib/api';

const navigation = [
  { name: 'Dashboard', href: '/admin', icon: HomeIcon },
  { name: 'Users', href: '/admin/users', icon: UsersIcon },
  { name: 'Analytics', href: '/admin/analytics', icon: ChartBarIcon },
  { name: 'Challenges', href: '/admin/challenges', icon: BellIcon },
  { name: 'Settings', href: '/admin/settings', icon: CogIcon },
];

export const AdminLayout: React.FC = () => {
  const location = useLocation();

  const handleLogout = () => {
    authAPI.logout();
    window.location.href = '/login';
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg">
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center justify-center h-16 bg-primary-600">
            <h1 className="text-xl font-bold text-white">HealthcareAI</h1>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    isActive
                      ? 'bg-primary-100 text-primary-600'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  <item.icon className="w-5 h-5 mr-3" />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* Logout */}
          <div className="p-4 border-t">
            <button
              onClick={handleLogout}
              className="flex items-center w-full px-3 py-2 text-sm font-medium text-gray-600 rounded-md hover:bg-gray-100 hover:text-gray-900"
            >
              <LogoutIcon className="w-5 h-5 mr-3" />
              Logout
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 ml-64">
        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
};
