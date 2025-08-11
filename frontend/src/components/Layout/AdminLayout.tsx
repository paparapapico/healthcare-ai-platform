import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
// # import 추가:
import { Payment } from '@mui/icons-material';

const navigation = [
  { name: 'Dashboard', href: '/admin', icon: '' },
  { name: 'Users', href: '/admin/users', icon: '' },
  { name: 'Analytics', href: '/admin/analytics', icon: '' },
  { name: 'Challenges', href: '/admin/challenges', icon: '' },
  { name: 'Settings', href: '/admin/settings', icon: '' },
];

export const AdminLayout: React.FC = () => {
  const location = useLocation();

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    window.location.href = '/login';
  };

  return (
    <div style={{
      display: 'flex',
      height: '100vh',
      backgroundColor: '#f3f4f6'
    }}>
      {/* Sidebar */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '256px',
        height: '100vh',
        backgroundColor: 'white',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        zIndex: 50,
        display: 'flex',
        flexDirection: 'column'
      }}>
        {/* Logo */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '64px',
          backgroundColor: '#2563eb',
          color: 'white'
        }}>
          <h1 style={{
            fontSize: '20px',
            fontWeight: 'bold',
            margin: 0
          }}>
            HealthcareAI
          </h1>
        </div>

        {/* Navigation */}
        <nav style={{
          flex: 1,
          padding: '24px 16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px'
        }}>
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '8px 12px',
                  fontSize: '14px',
                  fontWeight: '500',
                  borderRadius: '6px',
                  textDecoration: 'none',
                  backgroundColor: isActive ? '#dbeafe' : 'transparent',
                  color: isActive ? '#2563eb' : '#4b5563',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.backgroundColor = '#f3f4f6';
                  }
                }}
                onMouseOut={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }
                }}
              >
                <span style={{ marginRight: '12px', fontSize: '16px' }}>
                  {item.icon}
                </span>
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Logout */}
        <div style={{
          padding: '16px',
          borderTop: '1px solid #e5e7eb'
        }}>
          <button
            onClick={handleLogout}
            style={{
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              padding: '8px 12px',
              fontSize: '14px',
              fontWeight: '500',
              color: '#4b5563',
              backgroundColor: 'transparent',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              transition: 'background-color 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = '#f3f4f6';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            <span style={{ marginRight: '12px' }}></span>
            Logout
          </button>
        </div>
      </div>

      {/* Main content */}
      <div style={{
        flex: 1,
        marginLeft: '256px',
        minHeight: '100vh'
      }}>
        <main style={{
          padding: '24px',
          maxWidth: '100%',
          overflowX: 'auto'
        }}>
          <Outlet />
        </main>
      </div>
    </div>
  );
};
