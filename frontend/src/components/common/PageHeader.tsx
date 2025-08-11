import React from 'react';
import { Typography, Box, Breadcrumbs, Link } from '@mui/material';
import { useNavigate } from 'react-router-dom';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  breadcrumbs?: Array<{ label: string; path?: string }>;
  actions?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({ 
  title, 
  subtitle, 
  breadcrumbs, 
  actions 
}) => {
  const navigate = useNavigate();

  return (
    <Box sx={{ mb: 3 }}>
      {breadcrumbs && (
        <Breadcrumbs sx={{ mb: 1 }}>
          {breadcrumbs.map((crumb, index) => (
            crumb.path ? (
              <Link
                key={index}
                color="inherit"
                onClick={() => navigate(crumb.path!)}
                sx={{ cursor: 'pointer' }}
              >
                {crumb.label}
              </Link>
            ) : (
              <Typography key={index} color="text.primary">
                {crumb.label}
              </Typography>
            )
          ))}
        </Breadcrumbs>
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="body1" color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </Box>
        {actions && (
          <Box>
            {actions}
          </Box>
        )}
      </Box>
    </Box>
  );
};