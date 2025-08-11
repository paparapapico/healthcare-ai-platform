import React from 'react';
import { Typography, Box } from '@mui/material';

export const AnalyticsPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        분석
      </Typography>
      <Typography variant="body1" color="text.secondary">
        상세한 인사이트와 성능 지표를 확인하세요
      </Typography>
    </Box>
  );
};