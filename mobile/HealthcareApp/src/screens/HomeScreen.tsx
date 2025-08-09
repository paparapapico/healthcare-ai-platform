/**
 * Home Screen - 메인 대시보드
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Dimensions,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const { width: screenWidth } = Dimensions.get('window');

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

const HomeScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [userName, setUserName] = useState('사용자');
  const [dashboardData, setDashboardData] = useState<DashboardData>({
    health_score: 85,
    today_stats: {
      steps: 8432,
      calories: 2150,
      exercise_minutes: 45,
      water_glasses: 6,
    },
    recommendations: [
      {
        type: 'exercise',
        title: '오늘의 추천 운동',
        description: '스쿼트 3세트 (각 15회)',
        icon: '🏃',
      },
      {
        type: 'nutrition',
        title: '영양 팁',
        description: '단백질 섭취를 늘려보세요',
        icon: '🥗',
      },
    ],
  });
  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // API 호출 시뮬레이션
    setTimeout(() => {
      setRefreshing(false);
    }, 2000);
  }, []);

  const handleQuickAction = (action: string) => {
    switch (action) {
      case 'workout':
        navigation.navigate('Workout');
        break;
      case 'meditation':
        Alert.alert('명상', '명상 기능은 준비 중입니다');
        break;
      case 'sleep':
        Alert.alert('수면 분석', '수면 분석 기능은 준비 중입니다');
        break;
      case 'diet':
        Alert.alert('식단 기록', '식단 기록 기능은 준비 중입니다');
        break;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>안녕하세요, {userName}님 👋</Text>
            <Text style={styles.date}>
              {new Date().toLocaleDateString('ko-KR', {
                month: 'long',
                day: 'numeric',
                weekday: 'long',
              })}
            </Text>
          </View>
          <TouchableOpacity style={styles.notificationBtn}>
            <Text style={styles.notificationIcon}>🔔</Text>
          </TouchableOpacity>
        </View>

        {/* Health Score Card */}
        <View style={styles.healthScoreCard}>
          <Text style={styles.scoreTitle}>오늘의 건강 점수</Text>
          <View style={styles.scoreContainer}>
            <Text style={styles.scoreValue}>{dashboardData.health_score}</Text>
            <Text style={styles.scoreMax}>/ 100</Text>
          </View>
          <View style={styles.scoreBar}>
            <View
              style={[
                styles.scoreBarFill,
                { width: `${dashboardData.health_score}%` },
              ]}
            />
          </View>
        </View>

        {/* Today's Stats */}
        <Text style={styles.sectionTitle}>오늘의 활동</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.statsContainer}>
          <StatCard
            icon="🚶"
            value={dashboardData.today_stats.steps.toLocaleString()}
            label="걸음"
            color="#4CAF50"
          />
          <StatCard
            icon="🔥"
            value={dashboardData.today_stats.calories.toString()}
            label="칼로리"
            color="#FF9800"
          />
          <StatCard
            icon="⏱️"
            value={`${dashboardData.today_stats.exercise_minutes}분`}
            label="운동"
            color="#2196F3"
          />
          <StatCard
            icon="💧"
            value={`${dashboardData.today_stats.water_glasses}잔`}
            label="물"
            color="#00BCD4"
          />
        </ScrollView>

        {/* Quick Actions */}
        <Text style={styles.sectionTitle}>빠른 시작</Text>
        <View style={styles.actionGrid}>
          <ActionButton
            icon="🏃"
            label="AI 운동"
            onPress={() => handleQuickAction('workout')}
            color="#667eea"
          />
          <ActionButton
            icon="🧘"
            label="명상"
            onPress={() => handleQuickAction('meditation')}
            color="#764ba2"
          />
          <ActionButton
            icon="😴"
            label="수면 분석"
            onPress={() => handleQuickAction('sleep')}
            color="#f093fb"
          />
          <ActionButton
            icon="🍎"
            label="식단 기록"
            onPress={() => handleQuickAction('diet')}
            color="#4facfe"
          />
        </View>

        {/* Recommendations */}
        <Text style={styles.sectionTitle}>오늘의 추천</Text>
        {dashboardData.recommendations.map((rec, index) => (
          <RecommendationCard key={index} recommendation={rec} />
        ))}

        <View style={{ height: 100 }} />
      </ScrollView>
    </SafeAreaView>
  );
};

// Sub Components
const StatCard: React.FC<{
  icon: string;
  value: string;
  label: string;
  color: string;
}> = ({ icon, value, label, color }) => (
  <View style={[styles.statCard, { borderLeftColor: color }]}>
    <Text style={styles.statIcon}>{icon}</Text>
    <Text style={styles.statValue}>{value}</Text>
    <Text style={styles.statLabel}>{label}</Text>
  </View>
);

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
      <Text style={styles.recommendationDesc}>
        {recommendation.description}
      </Text>
    </View>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: 'white',
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
  notificationIcon: {
    fontSize: 24,
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
  scoreContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 20,
  },
  scoreValue: {
    fontSize: 60,
    fontWeight: 'bold',
    color: '#667eea',
  },
  scoreMax: {
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
    backgroundColor: '#667eea',
    borderRadius: 5,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginHorizontal: 20,
    marginTop: 25,
    marginBottom: 15,
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
});

export default HomeScreen;