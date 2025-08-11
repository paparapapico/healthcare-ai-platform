
// # 프리미엄 기능 보호 컴포넌트
// # frontend/src/components/common/PremiumGuard.tsx
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Alert,
} from '@mui/material';
import { Crown, Lock } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSubscription } from '../../hooks/useSubscription';

interface PremiumGuardProps {
  feature: string;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const PremiumGuard: React.FC<PremiumGuardProps> = ({ 
  feature, 
  children, 
  fallback 
}) => {
  const navigate = useNavigate();
  const { hasFeature, getCurrentPlan } = useSubscription();

  if (hasFeature(feature)) {
    return <>{children}</>;
  }

  if (fallback) {
    return <>{fallback}</>;
  }

  return (
    <Card sx={{ textAlign: 'center', p: 3 }}>
      <CardContent>
        <Crown color="warning" sx={{ fontSize: 64, mb: 2 }} />
        <Typography variant="h5" fontWeight="bold" gutterBottom>
          프리미엄 기능
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          이 기능은 프리미엄 구독자만 이용할 수 있습니다.
          <br />
          현재 플랜: <strong>{getCurrentPlan()}</strong>
        </Typography>
        <Button
          variant="contained"
          color="warning"
          startIcon={<Crown />}
          onClick={() => navigate('/subscription')}
          size="large"
        >
          프리미엄으로 업그레이드
        </Button>
      </CardContent>
    </Card>
  );
};