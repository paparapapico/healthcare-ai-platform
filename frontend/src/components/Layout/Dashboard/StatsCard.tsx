// 파일: ~/HealthcareAI/frontend/src/components/Dashboard/StatsCard.tsx
import React from 'react';
import { motion } from 'framer-motion';

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: {
    value: number;
    type: 'increase' | 'decrease';
  };
  icon: React.ReactNode;
  color?: 'blue' | 'green' | 'yellow' | 'red';
}

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  change,
  icon,
  color = 'blue'
}) => {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card p-6"
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {change && (
            <div className={`flex items-center mt-2 text-sm ${
              change.type === 'increase' ? 'text-green-600' : 'text-red-600'
            }`}>
              <span>{change.type === 'increase' ? '↗' : '↘'}</span>
              <span className="ml-1">{Math.abs(change.value)}%</span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-full ${colorClasses[color]}`}>
          <div className="w-6 h-6 text-white">
            {icon}
          </div>
        </div>
      </div>
    </motion.div>
  );
};