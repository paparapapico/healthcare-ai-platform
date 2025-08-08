/**
 * AI Healthcare Platform - Mobile App
 */

import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Dimensions,
} from 'react-native';

// API 설정
const API_BASE_URL = 'http://192.168.45.73/api/v1';

// 화면 크기
const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// ========================
// Types
// ========================

interface User {
  id: number;
  email: string;
  name: string;
  health_score: number;
}

interface DashboardData {
  health_score: number;
  today_stats: {
    steps: number;
    calories: number;
    exercise_minutes: number;
    water_glasses: number;
  };
  recommendations: Array<{
    type: string;
    title: string;
    description: string;
    icon: string;
  }>;
}

// ========================
// Components
// ========================

// 헬스 스코어 카드
const HealthScoreCard: React.FC<{ score: number }> = ({ score }) => {
  const getScoreColor = () => {
    if (score >= 80) return '#4CAF50';
    if (score >= 60) return '#FFC107';
    return '#F44336';
  };

  return (
    <View style={styles.healthScoreCard}>
      <Text style={styles.scoreTitle}>오늘의 건강 점수</Text>
      <View style={styles.scoreCircle}>
        <Text style={[styles.scoreValue, { color: getScoreColor() }]}>
          {score}
        </Text>
        <Text style={styles.scoreLabel}>/ 100</Text>
      </View>
      <View style={styles.scoreBar}>
        <View
          style={[
            styles.scoreBarFill,
            { width: `${score}%`, backgroundColor: getScoreColor() },
          ]}
        />
      </View>
    </View>
  );
};

// 통계 카드
const StatCard: React.FC<{
  icon: string;
  value: number | string;
  label: string;
  color: string;
}> = ({ icon, value, label, color }) => (
  <View style={[styles.statCard, { borderLeftColor: color }]}>
    <Text style={styles.statIcon}>{icon}</Text>
    <Text style={styles.statValue}>{value}</Text>
    <Text style={styles.statLabel}>{label}</Text>
  </View>
);

// 추천 카드
const RecommendationCard: React.FC<{
  recommendation: {
    icon: string;
    title: string;
    description: string;
  };
}> = ({ recommendation }) => (
  <TouchableOpacity style={styles.recommendationCard}>
    <Text style={styles.recommendationIcon}>{recommendation.icon}</Text>
    <View style={styles.recommendationContent}>
      <Text style={styles.recommendationTitle}>{recommendation.title}</Text>
      <Text style={styles.recommendationDesc}>{recommendation.description}</Text>
    </View>
  </TouchableOpacity>
);

// 액션 버튼
const ActionButton: React.FC<{
  icon: string;
  label: string;
  onPress: () => void;
  color: string;
}> = ({ icon, label, onPress, color }) => (
  <TouchableOpacity
    style={[styles.actionButton, { backgroundColor: color }]}
    onPress={onPress}>
    <Text style={styles.actionIcon}>{icon}</Text>
    <Text style={styles.actionLabel}>{label}</Text>
  </TouchableOpacity>
);

// ========================
// Main App Component
// ========================

const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  // 대시보드 데이터 로드
  const loadDashboardData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard`);
      const data = await response.json();
      setDashboardData(data);
    } catch (error) {
      console.error('Failed to load dashboard:', error);
      Alert.alert('오류', '데이터를 불러올 수 없습니다');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // 더미 사용자 설정 (실제로는 로그인 후)
    setUser({
      id: 1,
      email: 'user@example.com',
      name: '김건강',
      health_score: 85,
    });
    loadDashboardData();
  }, []);

  // 운동 시작
  const startWorkout = (type: string) => {
    Alert.alert(
      '운동 시작',
      `${type} 운동을 시작하시겠습니까?`,
      [
        { text: '취소', style: 'cancel' },
        {
          text: '시작',
          onPress: () => {
            // TODO: 운동 화면으로 이동
            console.log(`Starting ${type} workout`);
          },
        },
      ],
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#667eea" />
        <Text style={styles.loadingText}>로딩 중...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#667eea" />
      
      {/* 헤더 */}
      <View style={styles.header}>
        <View>
          <Text style={styles.greeting}>안녕하세요, {user?.name}님 👋</Text>
          <Text style={styles.date}>{new Date().toLocaleDateString('ko-KR')}</Text>
        </View>
        <TouchableOpacity style={styles.notificationBtn}>
          <Text>🔔</Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        showsVerticalScrollIndicator={false}>
        
        {/* 건강 점수 */}
        <HealthScoreCard score={dashboardData?.health_score || 0} />

        {/* 오늘의 통계 */}
        <Text style={styles.sectionTitle}>오늘의 활동</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.statsContainer}>
          <StatCard
            icon="🚶"
            value={dashboardData?.today_stats.steps.toLocaleString() || '0'}
            label="걸음"
            color="#4CAF50"
          />
          <StatCard
            icon="🔥"
            value={dashboardData?.today_stats.calories || '0'}
            label="칼로리"
            color="#FF9800"
          />
          <StatCard
            icon="⏱️"
            value={`${dashboardData?.today_stats.exercise_minutes || 0}분`}
            label="운동"
            color="#2196F3"
          />
          <StatCard
            icon="💧"
            value={`${dashboardData?.today_stats.water_glasses || 0}잔`}
            label="물"
            color="#00BCD4"
          />
        </ScrollView>

        {/* 빠른 시작 */}
        <Text style={styles.sectionTitle}>빠른 시작</Text>
        <View style={styles.actionGrid}>
          <ActionButton
            icon="🏃"
            label="AI 운동"
            onPress={() => startWorkout('AI')}
            color="#667eea"
          />
          <ActionButton
            icon="🧘"
            label="명상"
            onPress={() => startWorkout('명상')}
            color="#764ba2"
          />
          <ActionButton
            icon="😴"
            label="수면 분석"
            onPress={() => Alert.alert('수면 분석', '준비 중입니다')}
            color="#f093fb"
          />
          <ActionButton
            icon="🍎"
            label="식단 기록"
            onPress={() => Alert.alert('식단 기록', '준비 중입니다')}
            color="#4facfe"
          />
        </View>

        {/* 추천 사항 */}
        <Text style={styles.sectionTitle}>오늘의 추천</Text>
        {dashboardData?.recommendations.map((rec, index) => (
          <RecommendationCard key={index} recommendation={rec} />
        ))}

        <View style={{ height: 100 }} />
      </ScrollView>

      {/* 하단 네비게이션 */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem}>
          <Text style={styles.navIcon}>🏠</Text>
          <Text style={styles.navLabel}>홈</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Text style={styles.navIcon}>💪</Text>
          <Text style={styles.navLabel}>운동</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Text style={styles.navIcon}>📊</Text>
          <Text style={styles.navLabel}>리포트</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Text style={styles.navIcon}>👤</Text>
          <Text style={styles.navLabel}>프로필</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

// ========================
// Styles
// ========================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  greeting: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  date: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  notificationBtn: {
    padding: 10,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginHorizontal: 20,
    marginTop: 25,
    marginBottom: 15,
  },
  healthScoreCard: {
    backgroundColor: 'white',
    marginHorizontal: 20,
    marginTop: 20,
    padding: 25,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  scoreTitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 15,
  },
  scoreCircle: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 20,
  },
  scoreValue: {
    fontSize: 60,
    fontWeight: 'bold',
  },
  scoreLabel: {
    fontSize: 20,
    color: '#999',
    marginLeft: 5,
  },
  scoreBar: {
    height: 10,
    backgroundColor: '#e0e0e0',
    borderRadius: 5,
    overflow: 'hidden',
  },
  scoreBarFill: {
    height: '100%',
    borderRadius: 5,
  },
  statsContainer: {
    paddingHorizontal: 20,
  },
  statCard: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 15,
    marginRight: 15,
    minWidth: 120,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
    borderLeftWidth: 4,
  },
  statIcon: {
    fontSize: 30,
    marginBottom: 10,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  statLabel: {
    fontSize: 14,
    color: '#666',
  },
  actionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 15,
    justifyContent: 'space-between',
  },
  actionButton: {
    width: (screenWidth - 50) / 2,
    padding: 25,
    borderRadius: 20,
    alignItems: 'center',
    marginHorizontal: 5,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  actionIcon: {
    fontSize: 40,
    marginBottom: 10,
  },
  actionLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: 'white',
  },
  recommendationCard: {
    backgroundColor: 'white',
    marginHorizontal: 20,
    marginBottom: 15,
    padding: 20,
    borderRadius: 15,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  recommendationIcon: {
    fontSize: 35,
    marginRight: 15,
  },
  recommendationContent: {
    flex: 1,
  },
  recommendationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 5,
  },
  recommendationDesc: {
    fontSize: 14,
    color: '#666',
  },
  bottomNav: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: 'white',
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
  },
  navItem: {
    alignItems: 'center',
    padding: 10,
  },
  navIcon: {
    fontSize: 24,
    marginBottom: 5,
  },
  navLabel: {
    fontSize: 12,
    color: '#666',
  },
});

export default App;