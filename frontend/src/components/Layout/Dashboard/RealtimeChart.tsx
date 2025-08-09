// 파일: ~/HealthcareAI/frontend/src/components/Dashboard/RealtimeChart.tsx
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { wsService } from '@/lib/websocket';

interface RealtimeData {
  timestamp: string;
  activeUsers: number;
  workoutSessions: number;
}

export const RealtimeChart: React.FC = () => {
  const [data, setData] = useState<RealtimeData[]>([]);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      const socket = wsService.connect(token);
      
      // 실시간 데이터 구독
      socket.on('realtime_stats', (newData: RealtimeData) => {
        setData(prevData => {
          const updated = [...prevData, newData];
          // 최근 20개 데이터포인트만 유지
          return updated.slice(-20);
        });
      });

      // 초기 데이터 요청
      socket.emit('request_realtime_stats');
    }

    return () => {
      wsService.disconnect();
    };
  }, []);

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Real-time Activity
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp" 
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => new Date(value).toLocaleTimeString()}
            />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip 
              labelFormatter={(value) => new Date(value).toLocaleString()}
            />
            <Line
              type="monotone"
              dataKey="activeUsers"
              stroke="#3b82f6"
              strokeWidth={2}
              name="Active Users"
            />
            <Line
              type="monotone"
              dataKey="workoutSessions"
              stroke="#10b981"
              strokeWidth={2}
              name="Workout Sessions"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};