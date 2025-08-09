/**
 * App Navigator - 앱 네비게이션 설정
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { View, Text, StyleSheet } from 'react-native';

// Import Screens
import HomeScreen from '../screens/HomeScreen';
import WorkoutScreen from '../screens/WorkoutScreen';
import ReportScreen from '../screens/ReportScreen';
import ProfileScreen from '../screens/ProfileScreen';

// Navigation Types
export type RootStackParamList = {
  Main: undefined;
  Workout: undefined;
  Report: undefined;
  Profile: undefined;
};

export type TabParamList = {
  Home: undefined;
  Workout: undefined;
  Report: undefined;
  Profile: undefined;
};

const Tab = createBottomTabNavigator<TabParamList>();
const Stack = createStackNavigator<RootStackParamList>();

// Tab Icon Component
const TabIcon: React.FC<{ icon: string; focused: boolean }> = ({ icon, focused }) => (
  <View style={styles.tabIconContainer}>
    <Text style={[styles.tabIcon, focused && styles.tabIconFocused]}>{icon}</Text>
  </View>
);

// Tab Navigator
const TabNavigator = () => {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarStyle: styles.tabBar,
        tabBarActiveTintColor: '#667eea',
        tabBarInactiveTintColor: '#999',
        tabBarLabelStyle: styles.tabLabel,
        headerShown: false,
      }}>
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarLabel: '홈',
          tabBarIcon: ({ focused }) => <TabIcon icon="🏠" focused={focused} />,
        }}
      />
      <Tab.Screen
        name="Workout"
        component={WorkoutScreen}
        options={{
          tabBarLabel: '운동',
          tabBarIcon: ({ focused }) => <TabIcon icon="💪" focused={focused} />,
        }}
      />
      <Tab.Screen
        name="Report"
        component={ReportScreen}
        options={{
          tabBarLabel: '리포트',
          tabBarIcon: ({ focused }) => <TabIcon icon="📊" focused={focused} />,
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          tabBarLabel: '프로필',
          tabBarIcon: ({ focused }) => <TabIcon icon="👤" focused={focused} />,
        }}
      />
    </Tab.Navigator>
  );
};

// Main App Navigator
const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
        }}>
        <Stack.Screen name="Main" component={TabNavigator} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const styles = StyleSheet.create({
  tabBar: {
    height: 60,
    paddingBottom: 8,
    paddingTop: 8,
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  tabLabel: {
    fontSize: 12,
    marginBottom: 0,
  },
  tabIconContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabIcon: {
    fontSize: 24,
  },
  tabIconFocused: {
    transform: [{ scale: 1.1 }],
  },
});

export default AppNavigator;