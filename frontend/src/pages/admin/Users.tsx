// 파일: ~/HealthcareAI/frontend/src/pages/admin/Users.tsx
import React from 'react';
import { UserTable } from '@/components/Users/UserTable';

export const UsersPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Users</h1>
        <p className="text-gray-600">Manage registered users and their permissions</p>
      </div>

      <UserTable />
    </div>
  );
};