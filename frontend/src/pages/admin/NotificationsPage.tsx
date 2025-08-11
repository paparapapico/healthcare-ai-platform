import React from 'react';
import { Typography, Box } from '@mui/material';

export const NotificationsPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        알림 관리
      </Typography>
      <Typography variant="body1" color="text.secondary">
        사용자 알림을 관리하고 발송하세요
      </Typography>
    </Box>
  );
};