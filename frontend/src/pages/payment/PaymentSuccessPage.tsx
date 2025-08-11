// frontend/src/pages/payment/PaymentSuccessPage.tsx
import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import { CheckCircle, Error } from '@mui/icons-material';
import { paymentApi } from '../../services/paymentApi';

export const PaymentSuccessPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [subscriptionInfo, setSubscriptionInfo] = useState<any>(null);

  useEffect(() => {
    const sessionId = searchParams.get('session_id');
    
    if (!sessionId) {
      setError('결제 세션 ID가 없습니다.');
      setIsLoading(false);
      return;
    }

    handlePaymentSuccess(sessionId);
  }, [searchParams]);

  const handlePaymentSuccess = async (sessionId: string) => {
    try {
      const result = await paymentApi.handlePaymentSuccess(sessionId);
      setSuccess(true);
      setSubscriptionInfo(result.data);
    } catch (err) {
      setError('결제 처리 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <Box 
        display="flex" 
        flexDirection="column" 
        alignItems="center" 
        justifyContent="center" 
        minHeight="400px"
      >
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          결제를 처리하고 있습니다...
        </Typography>
      </Box>
    );
  }

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
          {success ? (
            <>
              <CheckCircle 
                color="success" 
                sx={{ fontSize: 80, mb: 2 }} 
              />
              <Typography variant="h4" fontWeight="bold" gutterBottom>
                결제 완료!
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                구독이 성공적으로 활성화되었습니다.
              </Typography>
              
              {subscriptionInfo && (
                <Alert severity="success" sx={{ mb: 3 }}>
                  <Typography variant="body2">
                    <strong>{subscriptionInfo.plan_name}</strong> 플랜이 활성화되었습니다.
                    <br />
                    다음 결제일: {new Date(subscriptionInfo.current_period_end).toLocaleDateString()}
                  </Typography>
                </Alert>
              )}

              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                <Button 
                  variant="contained" 
                  onClick={() => navigate('/dashboard')}
                  size="large"
                >
                  대시보드로 이동
                </Button>
                <Button 
                  variant="outlined" 
                  onClick={() => navigate('/subscription')}
                  size="large"
                >
                  구독 관리
                </Button>
              </Box>
            </>
          ) : (
            <>
              <Error 
                color="error" 
                sx={{ fontSize: 80, mb: 2 }} 
              />
              <Typography variant="h4" fontWeight="bold" gutterBottom>
                결제 실패
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                {error || '결제 처리 중 문제가 발생했습니다.'}
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                <Button 
                  variant="contained" 
                  onClick={() => navigate('/subscription')}
                  size="large"
                >
                  다시 시도
                </Button>
                <Button 
                  variant="outlined" 
                  onClick={() => navigate('/dashboard')}
                  size="large"
                >
                  대시보드로 이동
                </Button>
              </Box>
            </>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};