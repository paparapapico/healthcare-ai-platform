// # 사용량 제한 표시 컴포넌트
// # frontend/src/components/common/UsageLimitBar.tsx
import React from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Chip,
} from '@mui/material';
import { useSubscription } from '../../hooks/useSubscription';

interface UsageLimitBarProps {
  type: 'exercises' | 'ai_analysis';
  currentUsage: number;
  title: string;
}

export const UsageLimitBar: React.FC<UsageLimitBarProps> = ({
  type,
  currentUsage,
  title
}) => {
  const { getUsageLimits } = useSubscription();
  const limits = getUsageLimits();
  
  const maxLimit = type === 'exercises' 
    ? limits.max_exercises_per_day 
    : limits.max_ai_analysis;

  if (maxLimit === -1) {
    return (
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2">{title}</Typography>
          <Chip label="무제한" color="success" size="small" />
        </Box>
      </Box>
    );
  }

  const percentage = Math.min((currentUsage / maxLimit) * 100, 100);
  const isOverLimit = currentUsage >= maxLimit;

  return (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="body2">{title}</Typography>
        <Typography variant="body2" color={isOverLimit ? 'error.main' : 'text.secondary'}>
          {currentUsage} / {maxLimit}
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={percentage}
        color={isOverLimit ? 'error' : percentage > 80 ? 'warning' : 'primary'}
        sx={{ height: 8, borderRadius: 4 }}
      />
      {isOverLimit && (
        <Typography variant="caption" color="error.main">
          사용 한도를 초과했습니다. 프리미엄으로 업그레이드하세요.
        </Typography>
      )}
    </Box>
  );
};