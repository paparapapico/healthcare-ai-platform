/**
 * Notification Service for React Native
 * FCM 푸시 알림 및 로컬 알림 관리
 * 파일 위치: mobile/HealthcareApp/src/services/NotificationService.ts
 */

import messaging, { FirebaseMessagingTypes } from '@react-native-firebase/messaging';
import notifee, { 
  AndroidImportance, 
  AndroidStyle,
  EventType,
  Notification,
  TriggerType
} from '@notifee/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform, Alert } from 'react-native';

// API 엔드포인트
const API_BASE_URL = 'http://localhost:8000/api/v1';

// ========================
// Types
// ========================

interface NotificationData {
  type: string;
  title: string;
  body: string;
  data?: any;
  imageUrl?: string;
}

interface NotificationPreferences {
  workoutReminder: boolean;
  friendActivity: boolean;
  waterReminder: boolean;
  sleepReminder: boolean;
  weeklyReport: boolean;
}

// ========================
// Notification Service Class
// ========================

class NotificationService {
  private fcmToken: string | null = null;
  private notificationListener: (() => void) | null = null;
  private backgroundMessageHandler: (() => void) | null = null;

  /**
   * 알림 서비스 초기화
   */
  async initialize(): Promise<void> {
    try {
      // 알림 권한 요청
      await this.requestPermission();

      // FCM 토큰 획득
      await this.getFCMToken();

      // 알림 채널 생성 (Android)
      await this.createNotificationChannels();

      // 알림 리스너 설정
      this.setupNotificationListeners();

      console.log('Notification service initialized');
    } catch (error) {
      console.error('Failed to initialize notification service:', error);
    }
  }

  /**
   * 알림 권한 요청
   */
  private async requestPermission(): Promise<boolean> {
    if (Platform.OS === 'ios') {
      const authStatus = await messaging().requestPermission();
      const enabled =
        authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
        authStatus === messaging.AuthorizationStatus.PROVISIONAL;

      if (enabled) {
        console.log('Notification permission granted');
        return true;
      }
    } else {
      // Android는 자동 승인
      return true;
    }
    
    return false;
  }

  /**
   * FCM 토큰 획득 및 서버 등록
   */
  private async getFCMToken(): Promise<void> {
    try {
      // 기존 토큰 확인
      let token = await AsyncStorage.getItem('fcmToken');
      
      if (!token) {
        // 새 토큰 생성
        token = await messaging().getToken();
        
        if (token) {
          // 로컬 저장
          await AsyncStorage.setItem('fcmToken', token);
          
          // 서버에 등록
          await this.registerTokenToServer(token);
        }
      }
      
      this.fcmToken = token;
      console.log('FCM Token:', token);
      
      // 토큰 갱신 리스너
      messaging().onTokenRefresh(async (newToken) => {
        console.log('FCM Token refreshed:', newToken);
        this.fcmToken = newToken;
        await AsyncStorage.setItem('fcmToken', newToken);
        await this.registerTokenToServer(newToken);
      });
      
    } catch (error) {
      console.error('Failed to get FCM token:', error);
    }
  }

  /**
   * 서버에 FCM 토큰 등록
   */
  private async registerTokenToServer(token: string): Promise<void> {
    try {
      const response = await fetch(`${API_BASE_URL}/notifications/register-device`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // TODO: Add auth token
        },
        body: JSON.stringify({
          fcm_token: token,
          device_type: Platform.OS,
          device_model: Platform.Version,
        }),
      });

      if (response.ok) {
        console.log('Token registered to server');
      }
    } catch (error) {
      console.error('Failed to register token:', error);
    }
  }

  /**
   * 알림 채널 생성 (Android)
   */
  private async createNotificationChannels(): Promise<void> {
    if (Platform.OS === 'android') {
      // 운동 알림 채널
      await notifee.createChannel({
        id: 'workout',
        name: '운동 알림',
        importance: AndroidImportance.HIGH,
        sound: 'default',
      });

      // 건강 알림 채널
      await notifee.createChannel({
        id: 'health',
        name: '건강 알림',
        importance: AndroidImportance.DEFAULT,
      });

      // 소셜 알림 채널
      await notifee.createChannel({
        id: 'social',
        name: '소셜 알림',
        importance: AndroidImportance.DEFAULT,
      });

      // 리마인더 채널
      await notifee.createChannel({
        id: 'reminder',
        name: '리마인더',
        importance: AndroidImportance.HIGH,
        sound: 'default',
      });
    }
  }

  /**
   * 알림 리스너 설정
   */
  private setupNotificationListeners(): void {
    // 포그라운드 알림 처리
    this.notificationListener = messaging().onMessage(async (remoteMessage) => {
      console.log('Foreground notification:', remoteMessage);
      
      // 로컬 알림으로 표시
      await this.displayLocalNotification(remoteMessage);
    });

    // 백그라운드 알림 처리
    messaging().setBackgroundMessageHandler(async (remoteMessage) => {
      console.log('Background notification:', remoteMessage);
      
      // 필요한 백그라운드 작업 수행
      await this.handleBackgroundNotification(remoteMessage);
    });

    // 알림 클릭 처리
    messaging().onNotificationOpenedApp((remoteMessage) => {
      console.log('Notification opened app:', remoteMessage);
      this.handleNotificationOpen(remoteMessage);
    });

    // 앱이 종료된 상태에서 알림으로 열린 경우
    messaging()
      .getInitialNotification()
      .then((remoteMessage) => {
        if (remoteMessage) {
          console.log('Initial notification:', remoteMessage);
          this.handleNotificationOpen(remoteMessage);
        }
      });

    // Notifee 이벤트 리스너
    notifee.onForegroundEvent(({ type, detail }) => {
      switch (type) {
        case EventType.DISMISSED:
          console.log('Notification dismissed:', detail.notification);
          break;
        case EventType.PRESS:
          console.log('Notification pressed:', detail.notification);
          this.handleLocalNotificationPress(detail.notification);
          break;
      }
    });
  }

  /**
   * 로컬 알림 표시
   */
  private async displayLocalNotification(
    remoteMessage: FirebaseMessagingTypes.RemoteMessage
  ): Promise<void> {
    const { notification, data } = remoteMessage;
    
    if (!notification) return;

    // 채널 결정
    const channelId = this.getChannelId(typeof data?.type === 'string' ? data.type : undefined);

    // 알림 옵션
    const notificationOptions: Notification = {
      title: notification.title,
      body: notification.body,
      android: {
        channelId,
        importance: AndroidImportance.HIGH,
        pressAction: {
          id: 'default',
        },
        style: notification.body && notification.body.length > 50 ? {
          type: AndroidStyle.BIGTEXT,
          text: notification.body,
        } : undefined,
        largeIcon: notification.android?.imageUrl,
      },
             ios: {
         foregroundPresentationOptions: {
           badge: true,
           sound: true,
           banner: true,
           list: true,
         },
       },
      data: data as any,
    };

    // 알림 표시
    await notifee.displayNotification(notificationOptions);
  }

  /**
   * 백그라운드 알림 처리
   */
  private async handleBackgroundNotification(
    remoteMessage: FirebaseMessagingTypes.RemoteMessage
  ): Promise<void> {
    const { data } = remoteMessage;
    
    // 알림 타입별 처리
    switch (data?.type) {
      case 'workout_reminder':
        // 운동 데이터 미리 로드
        break;
      case 'water_reminder':
        // 물 섭취 데이터 업데이트
        break;
      default:
        break;
    }
  }

  /**
   * 알림 오픈 처리
   */
  private handleNotificationOpen(
    remoteMessage: FirebaseMessagingTypes.RemoteMessage
  ): void {
    const { data } = remoteMessage;
    
    // 딥링크 처리
    if (data?.action_url && typeof data.action_url === 'string') {
      this.navigateToScreen(data.action_url);
    }
  }

  /**
   * 로컬 알림 클릭 처리
   */
  private handleLocalNotificationPress(notification?: Notification): void {
    if (notification?.data?.action_url) {
      this.navigateToScreen(notification.data.action_url as string);
    }
  }

  /**
   * 화면 네비게이션
   */
  private navigateToScreen(url: string): void {
    // React Navigation과 연동
    // 예: NavigationService.navigate(url);
    console.log('Navigate to:', url);
  }

  /**
   * 채널 ID 결정
   */
  private getChannelId(type?: string): string {
    switch (type) {
      case 'workout_reminder':
      case 'streak_reminder':
        return 'workout';
      case 'friend_request':
      case 'friend_activity':
      case 'challenge_invite':
        return 'social';
      case 'water_reminder':
      case 'sleep_reminder':
        return 'health';
      default:
        return 'reminder';
    }
  }

  /**
   * 로컬 알림 예약
   */
  async scheduleLocalNotification(
    title: string,
    body: string,
    timestamp: number,
    data?: any
  ): Promise<string> {
    const notificationId = await notifee.createTriggerNotification(
      {
        title,
        body,
        android: {
          channelId: 'reminder',
        },
        data,
      },
      {
        type: TriggerType.TIMESTAMP,
        timestamp,
      }
    );
    
    return notificationId;
  }

  /**
   * 예약된 알림 취소
   */
  async cancelScheduledNotification(notificationId: string): Promise<void> {
    await notifee.cancelNotification(notificationId);
  }

  /**
   * 모든 알림 취소
   */
  async cancelAllNotifications(): Promise<void> {
    await notifee.cancelAllNotifications();
  }

  /**
   * 배지 수 설정 (iOS)
   */
  async setBadgeCount(count: number): Promise<void> {
    if (Platform.OS === 'ios') {
      await notifee.setBadgeCount(count);
    }
  }

  /**
   * 알림 설정 가져오기
   */
  async getNotificationPreferences(): Promise<NotificationPreferences> {
    try {
      const response = await fetch(`${API_BASE_URL}/notifications/preferences`, {
        headers: {
          // TODO: Add auth token
        },
      });
      
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error('Failed to get preferences:', error);
    }
    
    // 기본값
    return {
      workoutReminder: true,
      friendActivity: true,
      waterReminder: true,
      sleepReminder: true,
      weeklyReport: true,
    };
  }

  /**
   * 알림 설정 업데이트
   */
  async updateNotificationPreferences(
    preferences: NotificationPreferences
  ): Promise<void> {
    try {
      await fetch(`${API_BASE_URL}/notifications/preferences`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          // TODO: Add auth token
        },
        body: JSON.stringify(preferences),
      });
    } catch (error) {
      console.error('Failed to update preferences:', error);
    }
  }

  /**
   * 서비스 정리
   */
  cleanup(): void {
    if (this.notificationListener) {
      this.notificationListener();
    }
  }
}

// Singleton instance
const notificationService = new NotificationService();
export default notificationService;

// ========================
// Helper Functions
// ========================

/**
 * 운동 리마인더 설정
 */
export async function setWorkoutReminder(hour: number, minute: number): Promise<void> {
  const now = new Date();
  const reminderTime = new Date();
  reminderTime.setHours(hour, minute, 0, 0);
  
  // 오늘 시간이 지났으면 내일로 설정
  if (reminderTime <= now) {
    reminderTime.setDate(reminderTime.getDate() + 1);
  }
  
  await notificationService.scheduleLocalNotification(
    '운동할 시간이에요! 💪',
    '오늘의 운동을 시작해볼까요?',
    reminderTime.getTime(),
    { type: 'workout_reminder' }
  );
}

/**
 * 물 섭취 리마인더 설정
 */
export async function setWaterReminders(): Promise<void> {
  const hours = [9, 12, 15, 18, 21];
  
  for (const hour of hours) {
    const reminderTime = new Date();
    reminderTime.setHours(hour, 0, 0, 0);
    
    if (reminderTime > new Date()) {
      await notificationService.scheduleLocalNotification(
        '💧 물 마실 시간',
        '수분 보충을 잊지 마세요!',
        reminderTime.getTime(),
        { type: 'water_reminder' }
      );
    }
  }
}