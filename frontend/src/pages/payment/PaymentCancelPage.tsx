// # frontend/src/pages/payment/PaymentCancelPage.tsx
import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
} from '@mui/material';
import { Cancel } from '@mui/icons-material';

export const PaymentCancelPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box 
      display="flex" 
      justifyContent="center" 
      alignItems="center" 
      minHeight="400px"
      p={2}
    >
      <Card sx={{ maxWidth: 500, width: '100%' }}>
        <CardContent sx={{ textAlign: 'center', p: 4 }}>
          <Cancel 
            color="warning" 
            sx={{ fontSize: 80, mb: 2 }} 
          />
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            결제 취소
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            결제가 취소되었습니다. 언제든지 다시 구독하실 수 있습니다.
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
            <Button 
              variant="contained" 
              onClick={() => navigate('/subscription')}
              size="large"
            >
              구독 플랜 보기
            </Button>
            <Button 
              variant="outlined" 
              onClick={() => navigate('/dashboard')}
              size="large"
            >
              대시보드로 이동
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};