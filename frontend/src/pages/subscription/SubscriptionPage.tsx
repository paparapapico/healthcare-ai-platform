// frontend/src/pages/subscription/SubscriptionPage.tsx
import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  CheckCircle,
  Star,
  Diamond,
  EmojiEvents,
  Cancel,
  Payment,
} from '@mui/icons-material';
import { PageHeader } from '../../components/common/PageHeader';
import { useAuth } from '../../contexts/AuthContext';
import { paymentApi } from '../../services/paymentApi';
import { SubscriptionPlan, SubscriptionData } from '../../types/payment';

const PlanCard: React.FC<{
  plan: SubscriptionPlan;
  isCurrentPlan: boolean;
  onSubscribe: (planId: string) => void;
  isLoading: boolean;
}> = ({ plan, isCurrentPlan, onSubscribe, isLoading }) => {
  const getIcon = () => {
  switch (plan.name) {
    case 'Basic': return <Star color="primary" />;
    case 'Premium': return <Diamond color="secondary" />;
    case 'Pro': return <EmojiEvents sx={{ color: 'gold' }} />; // Crown 대신 EmojiEvents 사용
    default: return <CheckCircle />;
  }
};

  const getColor = () => {
    switch (plan.name) {
      case 'Basic': return 'primary';
      case 'Premium': return 'secondary';
      case 'Pro': return 'warning';
      default: return 'default';
    }
  };

  const features = JSON.parse(plan.features);

  return (
    <Card 
      sx={{ 
        height: '100%',
        border: isCurrentPlan ? 2 : 1,
        borderColor: isCurrentPlan ? 'primary.main' : 'grey.300',
        position: 'relative',
        '&:hover': { boxShadow: 6 }
      }}
    >
      {isCurrentPlan && (
        <Chip
          label="현재 플랜"
          color="primary"
          size="small"
          sx={{ position: 'absolute', top: 16, right: 16 }}
        />
      )}
      
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {getIcon()}
          <Typography variant="h5" fontWeight="bold" sx={{ ml: 1 }}>
            {plan.name}
          </Typography>
        </Box>

        <Box sx={{ mb: 3 }}>
          <Typography variant="h3" fontWeight="bold" color={`${getColor()}.main`}>
            ${plan.price}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            / {plan.interval === 'month' ? '월' : '년'}
          </Typography>
        </Box>

        <List dense>
          {features.map((feature: string, index: number) => (
            <ListItem key={index} sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <CheckCircle color="success" fontSize="small" />
              </ListItemIcon>
              <ListItemText 
                primary={feature}
                primaryTypographyProps={{ variant: 'body2' }}
              />
            </ListItem>
          ))}
        </List>

        <Button
          fullWidth
          variant={isCurrentPlan ? "outlined" : "contained"}
          color={getColor() as any}
          size="large"
          disabled={isCurrentPlan || isLoading}
          onClick={() => onSubscribe(plan.id)}
          sx={{ mt: 2 }}
        >
          {isLoading ? (
            <CircularProgress size={24} />
          ) : isCurrentPlan ? (
            '현재 플랜'
          ) : (
            '구독하기'
          )}
        </Button>
      </CardContent>
    </Card>
  );
};

export const SubscriptionPage: React.FC = () => {
  const { user } = useAuth();
  const [plans, setPlans] = useState<SubscriptionPlan[]>([]);
  const [subscriptionData, setSubscriptionData] = useState<SubscriptionData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [subscribeLoading, setSubscribeLoading] = useState<string | null>(null);
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [alert, setAlert] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setIsLoading(true);
      const [plansData, subData] = await Promise.all([
        paymentApi.getSubscriptionPlans(),
        paymentApi.getUserSubscription(),
      ]);
      setPlans(plansData);
      setSubscriptionData(subData);
    } catch (error) {
      setAlert({ type: 'error', message: '데이터를 불러오는 중 오류가 발생했습니다.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubscribe = async (planId: string) => {
    try {
      setSubscribeLoading(planId);
      const { checkout_url } = await paymentApi.createCheckoutSession(planId);
      window.location.href = checkout_url;
    } catch (error) {
      setAlert({ type: 'error', message: '구독 처리 중 오류가 발생했습니다.' });
      setSubscribeLoading(null);
    }
  };

  const handleCancelSubscription = async () => {
    try {
      await paymentApi.cancelSubscription(false);
      setAlert({ type: 'success', message: '구독이 기간 종료 시 취소됩니다.' });
      setCancelDialogOpen(false);
      loadData();
    } catch (error) {
      setAlert({ type: 'error', message: '구독 취소 중 오류가 발생했습니다.' });
    }
  };

  const getCurrentPlanId = () => {
    return subscriptionData?.plan?.id || null;
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <>
      <PageHeader
        title="구독 관리"
        subtitle="원하는 플랜을 선택하여 더 많은 기능을 이용하세요"
        breadcrumbs={[
          { label: '대시보드', path: '/dashboard' },
          { label: '구독 관리' },
        ]}
      />

      {alert && (
        <Alert 
          severity={alert.type} 
          onClose={() => setAlert(null)}
          sx={{ mb: 2 }}
        >
          {alert.message}
        </Alert>
      )}

      {/* 현재 구독 상태 */}
      {subscriptionData?.subscription && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              현재 구독 상태
            </Typography>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6}>
                <Box>
                  <Typography variant="body1">
                    플랜: <strong>{subscriptionData.plan?.name}</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    상태: <Chip 
                      label={subscriptionData.subscription.status} 
                      color={subscriptionData.subscription.status === 'active' ? 'success' : 'warning'}
                      size="small" 
                    />
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    다음 결제일: {new Date(subscriptionData.subscription.current_period_end).toLocaleDateString()}
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Box sx={{ textAlign: { sm: 'right' } }}>
                  {subscriptionData.subscription.cancel_at_period_end ? (
                    <Typography variant="body2" color="warning.main">
                      구독이 {new Date(subscriptionData.subscription.current_period_end).toLocaleDateString()}에 취소됩니다
                    </Typography>
                  ) : (
                    <Button
                      variant="outlined"
                      color="error"
                      startIcon={<Cancel />}
                      onClick={() => setCancelDialogOpen(true)}
                    >
                      구독 취소
                    </Button>
                  )}
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* 구독 플랜 카드들 */}
      <Grid container spacing={3}>
        {plans.map((plan) => (
          <Grid item xs={12} md={4} key={plan.id}>
            <PlanCard
              plan={plan}
              isCurrentPlan={getCurrentPlanId() === plan.id}
              onSubscribe={handleSubscribe}
              isLoading={subscribeLoading === plan.id}
            />
          </Grid>
        ))}
      </Grid>

      {/* 현재 사용량 표시 */}
      {subscriptionData && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              현재 사용 한도
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  일일 운동 횟수
                </Typography>
                <Typography variant="h6">
                  {subscriptionData.limits.max_exercises_per_day === -1 
                    ? '무제한' 
                    : subscriptionData.limits.max_exercises_per_day
                  }
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  AI 분석 횟수
                </Typography>
                <Typography variant="h6">
                  {subscriptionData.limits.max_ai_analysis === -1 
                    ? '무제한' 
                    : subscriptionData.limits.max_ai_analysis
                  }
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* 구독 취소 확인 다이얼로그 */}
      <Dialog
        open={cancelDialogOpen}
        onClose={() => setCancelDialogOpen(false)}
      >
        <DialogTitle>구독 취소</DialogTitle>
        <DialogContent>
          <Typography>
            정말로 구독을 취소하시겠습니까? 현재 결제 기간이 끝날 때까지 서비스를 계속 이용할 수 있습니다.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCancelDialogOpen(false)}>
            취소
          </Button>
          <Button onClick={handleCancelSubscription} color="error">
            구독 취소
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};