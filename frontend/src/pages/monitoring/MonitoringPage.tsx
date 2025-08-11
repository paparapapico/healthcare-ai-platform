import React from 'react';
import { Typography, Box } from '@mui/material';

export const MonitoringPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        실시간 모니터링
      </Typography>
      <Typography variant="body1" color="text.secondary">
        시스템 상태와 사용자 활동을 실시간으로 모니터링하세요
      </Typography>
    </Box>
  );
};