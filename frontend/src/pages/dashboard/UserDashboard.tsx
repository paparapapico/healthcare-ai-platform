import React from 'react';
import { Grid, Card, CardContent, Typography } from '@mui/material';

export const UserDashboard: React.FC = () => {
  return (
    <div>
      <Typography variant="h4" gutterBottom>
        내 대시보드
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">오늘의 운동</Typography>
              <Typography variant="h3">0</Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">이번 주 활동</Typography>
              <Typography variant="h3">0</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </div>
  );
};