import React from 'react';
import { Typography, Box } from '@mui/material';

export const ChallengesPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        챌린지 관리
      </Typography>
      <Typography variant="body1" color="text.secondary">
        피트니스 챌린지를 관리하고 경쟁을 운영하세요
      </Typography>
    </Box>
  );
};