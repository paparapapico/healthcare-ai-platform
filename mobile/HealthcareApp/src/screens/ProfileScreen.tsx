/**
 * Profile Screen - 사용자 프로필 및 설정
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Alert,
  Switch,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

interface UserProfile {
  name: string;
  email: string;
  age: number;
  height: number;
  weight: number;
  gender: string;
  memberSince: string;
  subscriptionType: string;
}

const ProfileScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [userProfile, setUserProfile] = useState<UserProfile>({
    name: '김건강',
    email: 'user@example.com',
    age: 30,
    height: 175,
    weight: 70,
    gender: '남성',
    memberSince: '2025-01-01',
    subscriptionType: 'Premium',
  });

  const [notifications, setNotifications] = useState(true);
  const [darkMode, setDarkMode] = useState(false);

  const handleEditProfile = () => {
    Alert.alert('프로필 수정', '프로필 수정 기능은 준비 중입니다');
  };

  const handleLogout = () => {
    Alert.alert(
      '로그아웃',
      '정말 로그아웃 하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        { 
          text: '로그아웃', 
          style: 'destructive',
          onPress: () => {
            // 로그아웃 로직
            console.log('Logged out');
          }
        },
      ]
    );
  };

  const menuItems = [
    { icon: '👤', title: '프로필 수정', onPress: handleEditProfile },
    { icon: '🏆', title: '내 업적', onPress: () => Alert.alert('업적', '준비 중입니다') },
    { icon: '👥', title: '친구 관리', onPress: () => Alert.alert('친구', '준비 중입니다') },
    { icon: '📊', title: '건강 데이터', onPress: () => navigation.navigate('Report') },
    { icon: '💳', title: '구독 관리', onPress: () => Alert.alert('구독', '준비 중입니다') },
    { icon: '❓', title: '도움말', onPress: () => Alert.alert('도움말', '준비 중입니다') },
    { icon: '📧', title: '문의하기', onPress: () => Alert.alert('문의', '준비 중입니다') },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>프로필</Text>
        </View>

        {/* Profile Card */}
        <View style={styles.profileCard}>
          <View style={styles.avatarContainer}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>
                {userProfile.name.charAt(0)}
              </Text>
            </View>
            <TouchableOpacity style={styles.editAvatarButton}>
              <Text style={styles.editAvatarIcon}>📷</Text>
            </TouchableOpacity>
          </View>
          
          <Text style={styles.userName}>{userProfile.name}</Text>
          <Text style={styles.userEmail}>{userProfile.email}</Text>
          
          <View style={styles.subscriptionBadge}>
            <Text style={styles.subscriptionText}>
              {userProfile.subscriptionType} 회원
            </Text>
          </View>

          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{userProfile.height}cm</Text>
              <Text style={styles.statLabel}>키</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{userProfile.weight}kg</Text>
              <Text style={styles.statLabel}>체중</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{userProfile.age}세</Text>
              <Text style={styles.statLabel}>나이</Text>
            </View>
          </View>
        </View>

        {/* Settings Section */}
        <View style={styles.settingsSection}>
          <Text style={styles.sectionTitle}>설정</Text>
          
          <View style={styles.settingItem}>
            <View style={styles.settingLeft}>
              <Text style={styles.settingIcon}>🔔</Text>
              <Text style={styles.settingTitle}>알림</Text>
            </View>
            <Switch
              value={notifications}
              onValueChange={setNotifications}
              trackColor={{ false: '#767577', true: '#667eea' }}
              thumbColor={notifications ? '#fff' : '#f4f3f4'}
            />
          </View>

          <View style={styles.settingItem}>
            <View style={styles.settingLeft}>
              <Text style={styles.settingIcon}>🌙</Text>
              <Text style={styles.settingTitle}>다크 모드</Text>
            </View>
            <Switch
              value={darkMode}
              onValueChange={setDarkMode}
              trackColor={{ false: '#767577', true: '#667eea' }}
              thumbColor={darkMode ? '#fff' : '#f4f3f4'}
            />
          </View>
        </View>

        {/* Menu Items */}
        <View style={styles.menuSection}>
          <Text style={styles.sectionTitle}>메뉴</Text>
          {menuItems.map((item, index) => (
            <TouchableOpacity
              key={index}
              style={styles.menuItem}
              onPress={item.onPress}>
              <View style={styles.menuLeft}>
                <Text style={styles.menuIcon}>{item.icon}</Text>
                <Text style={styles.menuTitle}>{item.title}</Text>
              </View>
              <Text style={styles.menuArrow}>›</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* App Info */}
        <View style={styles.appInfo}>
          <Text style={styles.appVersion}>앱 버전 1.0.0</Text>
          <Text style={styles.memberSince}>
            가입일: {new Date(userProfile.memberSince).toLocaleDateString('ko-KR')}
          </Text>
        </View>

        {/* Logout Button */}
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Text style={styles.logoutText}>로그아웃</Text>
        </TouchableOpacity>

        <View style={{ height: 100 }} />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: 'white',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
  profileCard: {
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
  avatarContainer: {
    position: 'relative',
    marginBottom: 15,
  },
  avatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: '#667eea',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    fontSize: 40,
    color: 'white',
    fontWeight: 'bold',
  },
  editAvatarButton: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 35,
    height: 35,
    borderRadius: 17.5,
    backgroundColor: 'white',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },
  editAvatarIcon: {
    fontSize: 20,
  },
  userName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  userEmail: {
    fontSize: 14,
    color: '#666',
    marginBottom: 15,
  },
  subscriptionBadge: {
    backgroundColor: '#667eea',
    paddingHorizontal: 15,
    paddingVertical: 5,
    borderRadius: 15,
    marginBottom: 20,
  },
  subscriptionText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '100%',
    justifyContent: 'space-around',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
  },
  statDivider: {
    width: 1,
    height: 30,
    backgroundColor: '#e0e0e0',
  },
  settingsSection: {
    backgroundColor: 'white',
    marginHorizontal: 20,
    padding: 20,
    borderRadius: 15,
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  settingLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  settingIcon: {
    fontSize: 20,
    marginRight: 12,
  },
  settingTitle: {
    fontSize: 16,
    color: '#333',
  },
  menuSection: {
    backgroundColor: 'white',
    marginHorizontal: 20,
    padding: 20,
    borderRadius: 15,
    marginBottom: 15,
  },
  menuItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  menuLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  menuIcon: {
    fontSize: 20,
    marginRight: 12,
  },
  menuTitle: {
    fontSize: 16,
    color: '#333',
  },
  menuArrow: {
    fontSize: 24,
    color: '#999',
  },
  appInfo: {
    alignItems: 'center',
    marginVertical: 20,
  },
  appVersion: {
    fontSize: 12,
    color: '#999',
    marginBottom: 5,
  },
  memberSince: {
    fontSize: 12,
    color: '#999',
  },
  logoutButton: {
    marginHorizontal: 20,
    marginBottom: 20,
    paddingVertical: 15,
    backgroundColor: '#ff4444',
    borderRadius: 12,
    alignItems: 'center',
  },
  logoutText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default ProfileScreen;