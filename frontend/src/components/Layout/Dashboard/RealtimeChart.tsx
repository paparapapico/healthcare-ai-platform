// 파일: HealthcareAI/frontend/src/components/Dashboard/RealtimeChart.tsx (수정된 버전)
import React, { useState, useEffect } from 'react';
import { wsService } from '@/lib/websocket';

interface RealtimeData {
  timestamp: string;
  activeUsers: number;
  workoutSessions: number;
  onlineUsers: number;
}

export const RealtimeChart: React.FC = () => {
  const [data, setData] = useState<RealtimeData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  useEffect(() => {
    // WebSocket 연결 시도
    const connectWebSocket = () => {
      try {
        const socket = wsService.connect();
        
        if (socket) {
          // 연결 상태 모니터링
          socket.on('connect', () => {
            setIsConnected(true);
            setConnectionError(null);
          });
          
          socket.on('disconnect', () => {
            setIsConnected(false);
          });
          
          socket.on('connect_error', (error) => {
            setConnectionError(`Connection failed: ${error.message}`);
            setIsConnected(false);
          });
          
          // 실시간 데이터 구독
          wsService.subscribeToRealtime((newData: RealtimeData) => {
            setData(prevData => {
              const updated = [...prevData, newData];
              // 최근 20개 데이터포인트만 유지
              return updated.slice(-20);
            });
          });
        }
      } catch (error) {
        console.error('WebSocket connection failed:', error);
        setConnectionError('WebSocket connection failed');
      }
    };

    connectWebSocket();

    return () => {
      wsService.disconnect();
    };
  }, []);

  // 모의 데이터로 fallback (WebSocket 연결 실패 시)
  useEffect(() => {
    if (!isConnected && !connectionError) {
      const interval = setInterval(() => {
        const mockData: RealtimeData = {
          timestamp: new Date().toISOString(),
          activeUsers: 40 + Math.floor(Math.random() * 10),
          workoutSessions: 10 + Math.floor(Math.random() * 10),
          onlineUsers: 5 + Math.floor(Math.random() * 5)
        };
        
        setData(prevData => {
          const updated = [...prevData, mockData];
          return updated.slice(-20);
        });
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [isConnected, connectionError]);

  return (
    <div className="card p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Real-time Activity
        </h3>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`}></div>
          <span className="text-sm text-gray-600">
            {isConnected ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>
      
      {connectionError && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
          <p className="text-sm text-yellow-800">
            ⚠️ WebSocket connection failed. Using mock data.
          </p>
        </div>
      )}
      
      <div className="h-64">
        {data.length > 0 ? (
          <div className="space-y-4">
            {/* 간단한 차트 대체 */}
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {data[data.length - 1]?.activeUsers || 0}
                </div>
                <div className="text-sm text-gray-600">Active Users</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {data[data.length - 1]?.workoutSessions || 0}
                </div>
                <div className="text-sm text-gray-600">Workout Sessions</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {data[data.length - 1]?.onlineUsers || 0}
                </div>
                <div className="text-sm text-gray-600">Online Users</div>
              </div>
            </div>
            
            {/* 간단한 트렌드 표시 */}
            <div className="mt-4">
              <div className="text-sm text-gray-600 mb-2">Recent Activity Trend:</div>
              <div className="flex space-x-1 h-20">
                {data.slice(-10).map((item, index) => (
                  <div
                    key={index}
                    className="flex-1 bg-blue-200 rounded-t"
                    style={{
                      height: `${(item.activeUsers / 50) * 100}%`,
                      minHeight: '10%'
                    }}
                    title={`${item.activeUsers} users at ${new Date(item.timestamp).toLocaleTimeString()}`}
                  ></div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-gray-500">
              {isConnected ? 'Waiting for data...' : 'Connecting...'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};