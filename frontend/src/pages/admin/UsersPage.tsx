import React from 'react';
import { UserTable } from '../../components/Users/UserTable';
import { Typography, Box } from '@mui/material';

export const UsersPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        사용자 관리
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        등록된 사용자들을 관리하고 권한을 설정하세요
      </Typography>
      <UserTable />
    </Box>
  );
};