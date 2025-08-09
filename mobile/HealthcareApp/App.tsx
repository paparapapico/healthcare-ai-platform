/**
 * AI Healthcare Platform - Mobile App
 */

import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import AppNavigator from './src/navigation/AppNavigator';
import { createContext, useContext } from 'react';
import notificationService from './src/services/NotificationService';


// API 설정
const API_BASE_URL = 'http://192.168.45.73/api/v1';

// Auth Context
interface AuthContextType {
  user: any | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  login: async () => {},
  logout: async () => {},
  isLoading: true,
});

export const useAuth = () => useContext(AuthContext);

// Auth Provider Component
const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // 알림 서비스 초기화
    notificationService.initialize();
    checkAuthStatus();
    
    return () => {
      // 정리
      notificationService.cleanup();
    };
  }, []);

  const checkAuthStatus = async () => {
    try {
      const token = await AsyncStorage.getItem('authToken');
      const userData = await AsyncStorage.getItem('userData');
      
      if (token && userData) {
        setUser(JSON.parse(userData));
      }
    } catch (error) {
      console.error('Failed to check auth status:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (email: string, password: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();
      
      // Save auth data
      await AsyncStorage.setItem('authToken', data.access_token);
      await AsyncStorage.setItem('userData', JSON.stringify(data.user));
      
      setUser(data.user);
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const logout = async () => {
    try {
      await AsyncStorage.removeItem('authToken');
      await AsyncStorage.removeItem('userData');
      setUser(null);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
};

// Main App Component
const App: React.FC = () => {
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    // Initialize app
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      // Load any necessary resources
      // Initialize services (analytics, crash reporting, etc.)
      
      // Simulate loading time
      await new Promise(resolve => setTimeout(() => resolve(undefined), 1000));
      
      setIsReady(true);
    } catch (error) {
      console.error('Failed to initialize app:', error);
      setIsReady(true); // Continue anyway
    }
  };

  if (!isReady) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#667eea" />
        <Text style={styles.loadingText}>AI 헬스케어</Text>
      </View>
    );
  }

  return (
    <AuthProvider>
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor="#fff" />
        <AppContent />
      </SafeAreaView>
    </AuthProvider>
  );
};

// App Content (handles auth state)
const AppContent: React.FC = () => {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#667eea" />
      </View>
    );
  }

  // Show login screen if not authenticated
  // For now, skip login and show main app directly
  // return user ? <AppNavigator /> : <LoginScreen />;
  
  return <AppNavigator />;
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  loadingText: {
    marginTop: 20,
    fontSize: 18,
    color: '#667eea',
    fontWeight: '600',
  },
});

export default App;