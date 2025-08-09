/**
 * Report Screen - 운동 기록 및 통계
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const { width: screenWidth } = Dimensions.get('window');

interface WorkoutHistory {
  date: string;
  exercise_type: string;
  duration: number;
  reps: number;
  calories: number;
  form_score: number;
}

interface WeeklyStats {
  totalWorkouts: number;
  totalMinutes: number;
  totalCalories: number;
  avgFormScore: number;
}

const ReportScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [selectedPeriod, setSelectedPeriod] = useState<'week' | 'month' | 'year'>('week');
  const [weeklyStats, setWeeklyStats] = useState<WeeklyStats>({
    totalWorkouts: 5,
    totalMinutes: 150,
    totalCalories: 1250,
    avgFormScore: 88.5,
  });
  
  const [workoutHistory, setWorkoutHistory] = useState<WorkoutHistory[]>([
    {
      date: '2025-01-07',
      exercise_type: 'squat',
      duration: 30,
      reps: 45,
      calories: 250,
      form_score: 92,
    },
    {
      date: '2025-01-06',
      exercise_type: 'pushup',
      duration: 25,
      reps: 30,
      calories: 200,
      form_score: 85,
    },
    {
      date: '2025-01-05',
      exercise_type: 'plank',
      duration: 15,
      reps: 3,
      calories: 100,
      form_score: 90,
    },
  ]);

  // 모의 차트 데이터
  const chartData = [
    { day: '월', value: 85 },
    { day: '화', value: 90 },
    { day: '수', value: 88 },
    { day: '목', value: 92 },
    { day: '금', value: 87 },
    { day: '토', value: 95 },
    { day: '일', value: 89 },
  ];

  const getExerciseIcon = (type: string) => {
    switch (type) {
      case 'squat': return '🏋️';
      case 'pushup': return '💪';
      case 'plank': return '🧘';
      case 'lunge': return '🦵';
      default: return '🏃';
    }
  };

  const getFormScoreColor = (score: number) => {
    if (score >= 90) return '#4CAF50';
    if (score >= 80) return '#8BC34A';
    if (score >= 70) return '#FFC107';
    return '#FF9800';
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>운동 리포트</Text>
          <View style={styles.periodSelector}>
            {(['week', 'month', 'year'] as const).map((period) => (
              <TouchableOpacity
                key={period}
                style={[
                  styles.periodButton,
                  selectedPeriod === period && styles.periodButtonActive,
                ]}
                onPress={() => setSelectedPeriod(period)}>
                <Text
                  style={[
                    styles.periodButtonText,
                    selectedPeriod === period && styles.periodButtonTextActive,
                  ]}>
                  {period === 'week' ? '주간' : period === 'month' ? '월간' : '연간'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Summary Stats */}
        <View style={styles.summaryContainer}>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryValue}>{weeklyStats.totalWorkouts}</Text>
            <Text style={styles.summaryLabel}>운동 횟수</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryValue}>{weeklyStats.totalMinutes}</Text>
            <Text style={styles.summaryLabel}>총 운동시간(분)</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryValue}>{weeklyStats.totalCalories}</Text>
            <Text style={styles.summaryLabel}>소모 칼로리</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryValue}>{weeklyStats.avgFormScore}%</Text>
            <Text style={styles.summaryLabel}>평균 자세점수</Text>
          </View>
        </View>

        {/* Chart Section */}
        <View style={styles.chartContainer}>
          <Text style={styles.sectionTitle}>주간 자세 점수</Text>
          <View style={styles.chart}>
            {chartData.map((data, index) => (
              <View key={index} style={styles.chartBar}>
                <View style={styles.barContainer}>
                  <View
                    style={[
                      styles.bar,
                      {
                        height: `${data.value}%`,
                        backgroundColor: getFormScoreColor(data.value),
                      },
                    ]}
                  />
                </View>
                <Text style={styles.barLabel}>{data.day}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Workout History */}
        <View style={styles.historyContainer}>
          <Text style={styles.sectionTitle}>최근 운동 기록</Text>
          {workoutHistory.map((workout, index) => (
            <TouchableOpacity key={index} style={styles.historyCard}>
              <View style={styles.historyLeft}>
                <Text style={styles.historyIcon}>
                  {getExerciseIcon(workout.exercise_type)}
                </Text>
                <View>
                  <Text style={styles.historyExercise}>
                    {workout.exercise_type.toUpperCase()}
                  </Text>
                  <Text style={styles.historyDate}>
                    {new Date(workout.date).toLocaleDateString('ko-KR')}
                  </Text>
                </View>
              </View>
              <View style={styles.historyRight}>
                <View style={styles.historyStats}>
                  <Text style={styles.historyStatValue}>{workout.reps}회</Text>
                  <Text style={styles.historyStatValue}>{workout.duration}분</Text>
                  <Text style={styles.historyStatValue}>{workout.calories}kcal</Text>
                </View>
                <View
                  style={[
                    styles.formScoreBadge,
                    { backgroundColor: getFormScoreColor(workout.form_score) },
                  ]}>
                  <Text style={styles.formScoreText}>{workout.form_score}%</Text>
                </View>
              </View>
            </TouchableOpacity>
          ))}
        </View>

        {/* Achievement Section */}
        <View style={styles.achievementContainer}>
          <Text style={styles.sectionTitle}>이번 주 달성</Text>
          <View style={styles.achievementGrid}>
            <AchievementBadge
              icon="🔥"
              title="3일 연속"
              achieved={true}
            />
            <AchievementBadge
              icon="💯"
              title="완벽한 자세"
              achieved={true}
            />
            <AchievementBadge
              icon="⚡"
              title="1000kcal"
              achieved={false}
            />
            <AchievementBadge
              icon="🏆"
              title="주간 목표"
              achieved={false}
            />
          </View>
        </View>

        <View style={{ height: 100 }} />
      </ScrollView>
    </SafeAreaView>
  );
};

// Sub Components
const AchievementBadge: React.FC<{
  icon: string;
  title: string;
  achieved: boolean;
}> = ({ icon, title, achieved }) => (
  <View style={[styles.achievementBadge, !achieved && styles.achievementBadgeInactive]}>
    <Text style={styles.achievementIcon}>{icon}</Text>
    <Text style={[styles.achievementTitle, !achieved && styles.achievementTitleInactive]}>
      {title}
    </Text>
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: 'white',
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  periodSelector: {
    flexDirection: 'row',
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
    padding: 4,
  },
  periodButton: {
    flex: 1,
    paddingVertical: 8,
    alignItems: 'center',
    borderRadius: 8,
  },
  periodButtonActive: {
    backgroundColor: 'white',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  periodButtonText: {
    fontSize: 14,
    color: '#666',
  },
  periodButtonTextActive: {
    color: '#667eea',
    fontWeight: '600',
  },
  summaryContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 15,
    marginTop: 20,
  },
  summaryCard: {
    width: (screenWidth - 40) / 2,
    backgroundColor: 'white',
    padding: 20,
    margin: 5,
    borderRadius: 15,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  summaryValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#667eea',
    marginBottom: 5,
  },
  summaryLabel: {
    fontSize: 12,
    color: '#666',
  },
  chartContainer: {
    backgroundColor: 'white',
    margin: 20,
    padding: 20,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  chart: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'flex-end',
    height: 150,
  },
  chartBar: {
    flex: 1,
    alignItems: 'center',
  },
  barContainer: {
    height: 120,
    width: 30,
    justifyContent: 'flex-end',
  },
  bar: {
    width: '100%',
    borderRadius: 5,
  },
  barLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
  },
  historyContainer: {
    paddingHorizontal: 20,
  },
  historyCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  historyLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  historyIcon: {
    fontSize: 30,
    marginRight: 15,
  },
  historyExercise: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  historyDate: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  historyRight: {
    alignItems: 'flex-end',
  },
  historyStats: {
    flexDirection: 'row',
    marginBottom: 5,
  },
  historyStatValue: {
    fontSize: 12,
    color: '#666',
    marginLeft: 10,
  },
  formScoreBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  formScoreText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  achievementContainer: {
    paddingHorizontal: 20,
    marginTop: 20,
  },
  achievementGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  achievementBadge: {
    width: (screenWidth - 50) / 4,
    aspectRatio: 1,
    backgroundColor: 'white',
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  achievementBadgeInactive: {
    backgroundColor: '#f0f0f0',
    opacity: 0.5,
  },
  achievementIcon: {
    fontSize: 30,
    marginBottom: 5,
  },
  achievementTitle: {
    fontSize: 10,
    color: '#333',
    fontWeight: '600',
  },
  achievementTitleInactive: {
    color: '#999',
  },
});

export default ReportScreen;