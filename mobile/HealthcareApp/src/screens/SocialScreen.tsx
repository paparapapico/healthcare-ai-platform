/**
 * Social & Health Screens
 * 소셜 기능과 건강 데이터 화면
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  FlatList,
  Modal,
  TextInput,
  Alert,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import {
  LineChart,
  BarChart,
  ProgressChart,
} from 'react-native-chart-kit';

const { width: screenWidth } = Dimensions.get('window');

// ========================
// Social Screen
// ========================

interface Friend {
  id: number;
  friend_name: string;
  friend_health_score: number;
  status: string;
}

interface Challenge {
  id: number;
  title: string;
  description: string;
  participant_count: number;
  end_date: string;
  my_progress?: {
    current_reps: number;
    current_calories: number;
    rank: number;
  };
}

interface LeaderboardEntry {
  rank: number;
  user_name: string;
  score: number;
  workout_count: number;
  total_calories: number;
}

export const SocialScreen: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'friends' | 'challenges' | 'leaderboard'>('friends');
  const [friends, setFriends] = useState<Friend[]>([]);
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [showAddFriend, setShowAddFriend] = useState(false);
  const [friendEmail, setFriendEmail] = useState('');

  // Load data based on active tab
  useEffect(() => {
    loadData();
  }, [activeTab]);

  const loadData = async () => {
    setLoading(true);
    try {
      const baseUrl = 'http://localhost:8000/api/v1/social';
      
      switch (activeTab) {
        case 'friends':
          const friendsRes = await fetch(`${baseUrl}/friends`);
          const friendsData = await friendsRes.json();
          setFriends(friendsData);
          break;
          
        case 'challenges':
          const challengesRes = await fetch(`${baseUrl}/challenges`);
          const challengesData = await challengesRes.json();
          setChallenges(challengesData);
          break;
          
        case 'leaderboard':
          const leaderboardRes = await fetch(`${baseUrl}/leaderboard?period=week`);
          const leaderboardData = await leaderboardRes.json();
          setLeaderboard(leaderboardData);
          break;
      }
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  const sendFriendRequest = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social/friends/request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ friend_email: friendEmail }),
      });
      
      const data = await response.json();
      if (response.ok) {
        Alert.alert('성공', '친구 요청을 보냈습니다');
        setShowAddFriend(false);
        setFriendEmail('');
        loadData();
      } else {
        Alert.alert('오류', data.detail);
      }
    } catch (error) {
      Alert.alert('오류', '친구 요청 실패');
    }
  };

  const joinChallenge = async (challengeId: number) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/social/challenges/${challengeId}/join`,
        { method: 'POST' }
      );
      
      if (response.ok) {
        Alert.alert('성공', '챌린지에 참가했습니다');
        loadData();
      }
    } catch (error) {
      Alert.alert('오류', '챌린지 참가 실패');
    }
  };

  // Friend Card Component
  const FriendCard: React.FC<{ friend: Friend }> = ({ friend }) => (
    <View style={styles.friendCard}>
      <View style={styles.friendAvatar}>
        <Text style={styles.avatarText}>{friend.friend_name[0]}</Text>
      </View>
      <View style={styles.friendInfo}>
        <Text style={styles.friendName}>{friend.friend_name}</Text>
        <Text style={styles.friendScore}>건강 점수: {friend.friend_health_score}</Text>
      </View>
      <TouchableOpacity style={styles.friendAction}>
        <Text>💬</Text>
      </TouchableOpacity>
    </View>
  );

  // Challenge Card Component
  const ChallengeCard: React.FC<{ challenge: Challenge }> = ({ challenge }) => (
    <TouchableOpacity 
      style={styles.challengeCard}
      onPress={() => joinChallenge(challenge.id)}
    >
      <Text style={styles.challengeTitle}>{challenge.title}</Text>
      <Text style={styles.challengeDesc}>{challenge.description}</Text>
      <View style={styles.challengeStats}>
        <Text style={styles.challengeStat}>👥 {challenge.participant_count}명</Text>
        <Text style={styles.challengeStat}>📅 {new Date(challenge.end_date).toLocaleDateString()}</Text>
      </View>
      {challenge.my_progress && (
        <View style={styles.myProgress}>
          <Text style={styles.progressText}>
            🏅 {challenge.my_progress.rank}위 | 
            💪 {challenge.my_progress.current_reps}회 | 
            🔥 {challenge.my_progress.current_calories}kcal
          </Text>
        </View>
      )}
    </TouchableOpacity>
  );

  // Leaderboard Entry Component
  const LeaderboardItem: React.FC<{ entry: LeaderboardEntry }> = ({ entry }) => (
    <View style={styles.leaderboardItem}>
      <View style={styles.rankBadge}>
        <Text style={styles.rankText}>
          {entry.rank === 1 ? '🥇' : entry.rank === 2 ? '🥈' : entry.rank === 3 ? '🥉' : entry.rank}
        </Text>
      </View>
      <View style={styles.leaderboardInfo}>
        <Text style={styles.leaderboardName}>{entry.user_name}</Text>
        <Text style={styles.leaderboardStats}>
          운동 {entry.workout_count}회 | {entry.total_calories.toFixed(0)}kcal
        </Text>
      </View>
      <Text style={styles.leaderboardScore}>{entry.score.toFixed(0)}점</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Tab Navigation */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'friends' && styles.activeTab]}
          onPress={() => setActiveTab('friends')}
        >
          <Text style={[styles.tabText, activeTab === 'friends' && styles.activeTabText]}>
            친구
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'challenges' && styles.activeTab]}
          onPress={() => setActiveTab('challenges')}
        >
          <Text style={[styles.tabText, activeTab === 'challenges' && styles.activeTabText]}>
            챌린지
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'leaderboard' && styles.activeTab]}
          onPress={() => setActiveTab('leaderboard')}
        >
          <Text style={[styles.tabText, activeTab === 'leaderboard' && styles.activeTabText]}>
            랭킹
          </Text>
        </TouchableOpacity>
      </View>

      {/* Content */}
      {loading ? (
        <ActivityIndicator size="large" color="#667eea" style={{ marginTop: 50 }} />
      ) : (
        <ScrollView style={styles.content}>
          {activeTab === 'friends' && (
            <>
              <TouchableOpacity
                style={styles.addButton}
                onPress={() => setShowAddFriend(true)}
              >
                <Text style={styles.addButtonText}>+ 친구 추가</Text>
              </TouchableOpacity>
              {friends.map((friend) => (
                <FriendCard key={friend.id} friend={friend} />
              ))}
            </>
          )}

          {activeTab === 'challenges' && (
            <>
              <TouchableOpacity style={styles.addButton}>
                <Text style={styles.addButtonText}>+ 챌린지 만들기</Text>
              </TouchableOpacity>
              {challenges.map((challenge) => (
                <ChallengeCard key={challenge.id} challenge={challenge} />
              ))}
            </>
          )}

          {activeTab === 'leaderboard' && (
            <>
              <View style={styles.periodSelector}>
                <TouchableOpacity style={styles.periodButton}>
                  <Text>일간</Text>
                </TouchableOpacity>
                <TouchableOpacity style={[styles.periodButton, styles.activePeriod]}>
                  <Text style={styles.activePeriodText}>주간</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.periodButton}>
                  <Text>월간</Text>
                </TouchableOpacity>
              </View>
              {leaderboard.map((entry) => (
                <LeaderboardItem key={entry.rank} entry={entry} />
              ))}
            </>
          )}
        </ScrollView>
      )}

      {/* Add Friend Modal */}
      <Modal
        visible={showAddFriend}
        transparent
        animationType="slide"
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>친구 추가</Text>
            <TextInput
              style={styles.modalInput}
              placeholder="친구의 이메일 주소"
              value={friendEmail}
              onChangeText={setFriendEmail}
              keyboardType="email-address"
              autoCapitalize="none"
            />
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.modalButton}
                onPress={() => setShowAddFriend(false)}
              >
                <Text>취소</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButton, styles.modalButtonPrimary]}
                onPress={sendFriendRequest}
              >
                <Text style={styles.modalButtonPrimaryText}>요청 보내기</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
};

// ========================
// Health Screen
// ========================

interface HealthData {
  health_score: number;
  weight_trend: string;
  sleep_quality: number;
  nutrition_score: number;
  activity_level: number;
  hydration_status: {
    percentage: number;
    status: string;
    glasses_drunk: number;
    glasses_remaining: number;
  };
  recommendations: string[];
}

export const HealthScreen: React.FC = () => {
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [weightHistory, setWeightHistory] = useState<any[]>([]);
  const [sleepHistory, setSleepHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHealthData();
  }, []);

  const loadHealthData = async () => {
    try {
      // Load dashboard data
      const dashboardRes = await fetch('http://localhost:8000/api/v1/health/dashboard');
      const dashboard = await dashboardRes.json();
      setHealthData(dashboard);

      // Load weight history
      const metricsRes = await fetch('http://localhost:8000/api/v1/health/metrics/history?days=7');
      const metrics = await metricsRes.json();
      setWeightHistory(metrics.weight || []);

      // Load sleep history
      const sleepRes = await fetch('http://localhost:8000/api/v1/health/sleep/analysis');
      const sleep = await sleepRes.json();
      setSleepHistory(sleep.sleep_history || []);
    } catch (error) {
      console.error('Failed to load health data:', error);
    } finally {
      setLoading(false);
    }
  };

  const logWater = async (amount: number) => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/health/water/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ amount_ml: amount }),
      });
      
      if (response.ok) {
        Alert.alert('성공', `${amount}ml 기록 완료`);
        loadHealthData();
      }
    } catch (error) {
      Alert.alert('오류', '물 섭취 기록 실패');
    }
  };

  if (loading || !healthData) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#667eea" />
      </View>
    );
  }

  // Chart data
  const chartData = {
    labels: weightHistory.slice(-7).map(w => w.date.slice(5)),
    datasets: [{
      data: weightHistory.slice(-7).map(w => w.value),
    }],
  };

  const progressData = {
    labels: ['수면', '영양', '활동'],
    data: [
      healthData.sleep_quality / 100,
      healthData.nutrition_score / 100,
      healthData.activity_level / 100,
    ],
  };

  return (
    <ScrollView style={styles.container}>
      {/* Health Score Card */}
      <View style={styles.healthScoreCard}>
        <Text style={styles.healthScoreTitle}>종합 건강 점수</Text>
        <View style={styles.scoreCircle}>
          <Text style={styles.scoreNumber}>{healthData.health_score.toFixed(0)}</Text>
          <Text style={styles.scoreMax}>/100</Text>
        </View>
        <View style={styles.trendIndicator}>
          <Text style={styles.trendText}>
            {healthData.weight_trend === 'up' ? '⬆️' : 
             healthData.weight_trend === 'down' ? '⬇️' : '➡️'} 체중 {healthData.weight_trend}
          </Text>
        </View>
      </View>

      {/* Progress Charts */}
      <View style={styles.chartCard}>
        <Text style={styles.chartTitle}>건강 지표</Text>
        <ProgressChart
          data={progressData}
          width={screenWidth - 40}
          height={200}
          strokeWidth={16}
          radius={32}
          chartConfig={{
            backgroundColor: '#fff',
            backgroundGradientFrom: '#fff',
            backgroundGradientTo: '#fff',
            color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
            labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
          }}
          hideLegend={false}
        />
      </View>

      {/* Water Intake */}
      <View style={styles.waterCard}>
        <Text style={styles.waterTitle}>💧 수분 섭취</Text>
        <View style={styles.waterProgress}>
          <View style={styles.waterBar}>
            <View 
              style={[
                styles.waterFill, 
                { width: `${healthData.hydration_status.percentage}%` }
              ]} 
            />
          </View>
          <Text style={styles.waterText}>
            {healthData.hydration_status.glasses_drunk} / 8잔
          </Text>
        </View>
        <Text style={styles.waterStatus}>{healthData.hydration_status.status}</Text>
        <View style={styles.waterButtons}>
          <TouchableOpacity
            style={styles.waterButton}
            onPress={() => logWater(250)}
          >
            <Text style={styles.waterButtonText}>+1잔</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.waterButton}
            onPress={() => logWater(500)}
          >
            <Text style={styles.waterButtonText}>+2잔</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Weight Chart */}
      {weightHistory.length > 0 && (
        <View style={styles.chartCard}>
          <Text style={styles.chartTitle}>체중 변화</Text>
          <LineChart
            data={chartData}
            width={screenWidth - 40}
            height={200}
            chartConfig={{
              backgroundColor: '#fff',
              backgroundGradientFrom: '#fff',
              backgroundGradientTo: '#fff',
              decimalPlaces: 1,
              color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
              labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
              style: {
                borderRadius: 16,
              },
            }}
            bezier
            style={{
              marginVertical: 8,
              borderRadius: 16,
            }}
          />
        </View>
      )}

      {/* Sleep Analysis */}
      <View style={styles.sleepCard}>
        <Text style={styles.sleepTitle}>😴 수면 분석</Text>
        {sleepHistory.map((sleep, index) => (
          <View key={index} style={styles.sleepItem}>
            <Text style={styles.sleepDate}>{sleep.date}</Text>
            <Text style={styles.sleepHours}>{sleep.hours}시간</Text>
            <View style={styles.sleepQuality}>
              <View 
                style={[
                  styles.sleepQualityBar,
                  { width: `${sleep.quality}%`, backgroundColor: sleep.quality > 70 ? '#4CAF50' : '#FFC107' }
                ]}
              />
            </View>
          </View>
        ))}
      </View>

      {/* Recommendations */}
      <View style={styles.recommendationsCard}>
        <Text style={styles.recommendationsTitle}>💡 건강 추천</Text>
        {healthData.recommendations.map((rec, index) => (
          <View key={index} style={styles.recommendationItem}>
            <Text style={styles.recommendationText}>• {rec}</Text>
          </View>
        ))}
      </View>

      <View style={{ height: 100 }} />
    </ScrollView>
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
  },
  
  // Tab Bar
  tabBar: {
    flexDirection: 'row',
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  tab: {
    flex: 1,
    paddingVertical: 15,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: '#667eea',
  },
  tabText: {
    fontSize: 16,
    color: '#666',
  },
  activeTabText: {
    color: '#667eea',
    fontWeight: '600',
  },
  
  // Content
  content: {
    flex: 1,
    padding: 20,
  },
  
  // Social Components
  addButton: {
    backgroundColor: '#667eea',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 20,
  },
  addButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  
  friendCard: {
    flexDirection: 'row',
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    alignItems: 'center',
  },
  friendAvatar: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#667eea',
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  friendInfo: {
    flex: 1,
    marginLeft: 15,
  },
  friendName: {
    fontSize: 16,
    fontWeight: '600',
  },
  friendScore: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  friendAction: {
    padding: 10,
  },
  
  challengeCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
  },
  challengeTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  challengeDesc: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10,
  },
  challengeStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  challengeStat: {
    fontSize: 14,
    color: '#999',
  },
  myProgress: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  progressText: {
    fontSize: 14,
    color: '#667eea',
  },
  
  leaderboardItem: {
    flexDirection: 'row',
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    alignItems: 'center',
  },
  rankBadge: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  rankText: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  leaderboardInfo: {
    flex: 1,
    marginLeft: 15,
  },
  leaderboardName: {
    fontSize: 16,
    fontWeight: '600',
  },
  leaderboardStats: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
  },
  leaderboardScore: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#667eea',
  },
  
  periodSelector: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 20,
  },
  periodButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    marginHorizontal: 5,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
  },
  activePeriod: {
    backgroundColor: '#667eea',
  },
  activePeriodText: {
    color: 'white',
  },
  
  // Modal
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    width: '80%',
    backgroundColor: 'white',
    borderRadius: 20,
    padding: 20,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  modalInput: {
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 10,
    padding: 15,
    fontSize: 16,
    marginBottom: 20,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  modalButton: {
    paddingHorizontal: 30,
    paddingVertical: 10,
    borderRadius: 10,
  },
  modalButtonPrimary: {
    backgroundColor: '#667eea',
  },
  modalButtonPrimaryText: {
    color: 'white',
  },
  
  // Health Components
  healthScoreCard: {
    backgroundColor: 'white',
    margin: 20,
    padding: 25,
    borderRadius: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  healthScoreTitle: {
    fontSize: 18,
    color: '#666',
    marginBottom: 15,
  },
  scoreCircle: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  scoreNumber: {
    fontSize: 60,
    fontWeight: 'bold',
    color: '#667eea',
  },
  scoreMax: {
    fontSize: 20,
    color: '#999',
    marginLeft: 5,
  },
  trendIndicator: {
    marginTop: 15,
  },
  trendText: {
    fontSize: 16,
    color: '#666',
  },
  
  chartCard: {
    backgroundColor: 'white',
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 20,
    borderRadius: 15,
  },
  chartTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  
  waterCard: {
    backgroundColor: 'white',
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 20,
    borderRadius: 15,
  },
  waterTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  waterProgress: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  waterBar: {
    flex: 1,
    height: 20,
    backgroundColor: '#e0e0e0',
    borderRadius: 10,
    overflow: 'hidden',
    marginRight: 10,
  },
  waterFill: {
    height: '100%',
    backgroundColor: '#00BCD4',
  },
  waterText: {
    fontSize: 14,
    color: '#666',
  },
  waterStatus: {
    fontSize: 14,
    color: '#999',
    marginBottom: 15,
  },
  waterButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  waterButton: {
    backgroundColor: '#00BCD4',
    paddingHorizontal: 30,
    paddingVertical: 10,
    borderRadius: 20,
  },
  waterButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  
  sleepCard: {
    backgroundColor: 'white',
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 20,
    borderRadius: 15,
  },
  sleepTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  sleepItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  sleepDate: {
    flex: 1,
    fontSize: 14,
    color: '#666',
  },
  sleepHours: {
    fontSize: 14,
    fontWeight: '600',
    marginRight: 10,
  },
  sleepQuality: {
    width: 60,
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  sleepQualityBar: {
    height: '100%',
    borderRadius: 4,
  },
  
  recommendationsCard: {
    backgroundColor: 'white',
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 20,
    borderRadius: 15,
  },
  recommendationsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  recommendationItem: {
    marginBottom: 10,
  },
  recommendationText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
});

export default { SocialScreen, HealthScreen };