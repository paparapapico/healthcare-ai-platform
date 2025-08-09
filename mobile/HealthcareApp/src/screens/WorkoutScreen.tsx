/**
 * Real-time Workout Analysis Screen
 * 실시간 운동 분석 화면
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Dimensions,
  Platform,
  SafeAreaView,
  ScrollView,
} from 'react-native';
import { RNCamera } from 'react-native-camera';
import ViewShot from 'react-native-view-shot';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// WebSocket 연결 설정
const WS_URL = 'ws://localhost:8000/ws';

// ========================
// Types
// ========================

interface WorkoutStats {
  reps: number;
  formScore: number;
  calories: number;
  duration: number;
}

interface PoseAngles {
  knee?: number;
  elbow?: number;
  hip?: number;
  back?: number;
}

interface AnalysisResult {
  form_score: number;
  stage: string;
  rep_count: number;
  angles: PoseAngles;
  feedback: string[];
  corrections: string[];
  calories_burned: number;
  duration: number;
}

// ========================
// Components
// ========================

const WorkoutScreen: React.FC = () => {
  // State
  const [isWorkoutActive, setIsWorkoutActive] = useState(false);
  const [exerciseType, setExerciseType] = useState('squat');
  const [stats, setStats] = useState<WorkoutStats>({
    reps: 0,
    formScore: 0,
    calories: 0,
    duration: 0,
  });
  const [currentFeedback, setCurrentFeedback] = useState<string>('');
  const [corrections, setCorrections] = useState<string[]>([]);
  const [stage, setStage] = useState<string>('ready');
  const [angles, setAngles] = useState<PoseAngles>({});
  const [fps, setFps] = useState(0);
  
  // Refs
  const cameraRef = useRef<RNCamera>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  
  // WebSocket 연결
  const connectWebSocket = useCallback(() => {
    const clientId = `user_${Date.now()}`;
    const ws = new WebSocket(`${WS_URL}/${clientId}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      // 운동 시작 메시지
      ws.send(JSON.stringify({
        type: 'start_workout',
        exercise_type: exerciseType,
      }));
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      Alert.alert('연결 오류', '서버 연결에 실패했습니다');
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    wsRef.current = ws;
  }, [exerciseType]);
  
  // WebSocket 메시지 처리
  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'analysis_result':
        if (message.success && message.data) {
          const data: AnalysisResult = message.data;
          
          // 통계 업데이트
          setStats({
            reps: data.rep_count,
            formScore: data.form_score,
            calories: data.calories_burned,
            duration: data.duration,
          });
          
          // 피드백 업데이트
          if (data.feedback.length > 0) {
            setCurrentFeedback(data.feedback[0]);
          }
          setCorrections(data.corrections);
          
          // 상태 업데이트
          setStage(data.stage);
          setAngles(data.angles);
          
          // FPS 계산
          frameCountRef.current++;
          const elapsed = (Date.now() - startTimeRef.current) / 1000;
          if (elapsed > 1) {
            setFps(Math.round(frameCountRef.current / elapsed));
            frameCountRef.current = 0;
            startTimeRef.current = Date.now();
          }
        }
        break;
        
      case 'session_summary':
        handleWorkoutComplete(message.data);
        break;
        
      case 'error':
        console.error('Server error:', message.error);
        break;
    }
  };
  
  // 카메라 프레임 캡처 및 전송
  const captureAndSendFrame = async () => {
    if (!cameraRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    
    try {
      const options = { quality: 0.5, base64: true };
      const data = await cameraRef.current.takePictureAsync(options);
      
      if (data.base64) {
        wsRef.current.send(JSON.stringify({
          type: 'frame',
          frame: data.base64,
        }));
      }
    } catch (error) {
      console.error('Failed to capture frame:', error);
    }
  };
  
  // 운동 시작
  const startWorkout = () => {
    setIsWorkoutActive(true);
    startTimeRef.current = Date.now();
    frameCountRef.current = 0;
    
    // WebSocket 연결
    connectWebSocket();
    
    // 프레임 캡처 시작 (초당 10프레임)
    frameIntervalRef.current = setInterval(captureAndSendFrame, 100);
  };
  
  // 운동 종료
  const stopWorkout = () => {
    setIsWorkoutActive(false);
    
    // 프레임 캡처 중지
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    // WebSocket 종료 메시지
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop_workout' }));
      setTimeout(() => {
        wsRef.current?.close();
        wsRef.current = null;
      }, 1000);
    }
  };
  
  // 운동 완료 처리
  const handleWorkoutComplete = (summary: any) => {
    Alert.alert(
      '운동 완료! 🎉',
      `총 ${summary.total_reps}개\n` +
      `칼로리: ${summary.total_calories}kcal\n` +
      `평균 자세 점수: ${summary.avg_form_score}점\n` +
      `운동 시간: ${Math.floor(summary.duration_seconds / 60)}분 ${Math.round(summary.duration_seconds % 60)}초`,
      [
        {
          text: '확인',
          onPress: () => {
            // 초기화
            setStats({ reps: 0, formScore: 0, calories: 0, duration: 0 });
            setCurrentFeedback('');
            setCorrections([]);
          },
        },
      ]
    );
  };
  
  // Cleanup
  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);
  
  // Form Score 색상
  const getFormScoreColor = (score: number) => {
    if (score >= 80) return '#4CAF50';
    if (score >= 60) return '#FFC107';
    return '#F44336';
  };
  
  // Stage 아이콘
  const getStageIcon = (stage: string) => {
    switch (stage) {
      case 'up': return '⬆️';
      case 'down': return '⬇️';
      case 'hold': return '⏸️';
      default: return '🔄';
    }
  };
  
  return (
    <SafeAreaView style={styles.container}>
      {/* 카메라 뷰 */}
      <RNCamera
        ref={cameraRef}
        style={styles.camera}
        type={RNCamera.Constants.Type.front}
        captureAudio={false}
        androidCameraPermissionOptions={{
          title: '카메라 권한',
          message: '운동 분석을 위해 카메라 접근이 필요합니다',
          buttonPositive: '확인',
          buttonNegative: '취소',
        }}
      >
        {/* 오버레이 UI */}
        <View style={styles.overlay}>
          {/* 상단 정보 */}
          <View style={styles.topInfo}>
            <View style={styles.statsCard}>
              <Text style={styles.exerciseTitle}>{exerciseType.toUpperCase()}</Text>
              <View style={styles.statsRow}>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{stats.reps}</Text>
                  <Text style={styles.statLabel}>횟수</Text>
                </View>
                <View style={styles.statItem}>
                  <Text 
                    style={[
                      styles.statValue, 
                      { color: getFormScoreColor(stats.formScore) }
                    ]}
                  >
                    {Math.round(stats.formScore)}%
                  </Text>
                  <Text style={styles.statLabel}>자세</Text>
                </View>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{stats.calories.toFixed(1)}</Text>
                  <Text style={styles.statLabel}>kcal</Text>
                </View>
              </View>
            </View>
            
            {/* FPS 표시 */}
            <View style={styles.fpsIndicator}>
              <Text style={styles.fpsText}>FPS: {fps}</Text>
            </View>
          </View>
          
          {/* 각도 정보 */}
          {Object.keys(angles).length > 0 && (
            <View style={styles.anglesContainer}>
              {Object.entries(angles).map(([joint, angle]) => (
                <View key={joint} style={styles.angleItem}>
                  <Text style={styles.angleLabel}>{joint}</Text>
                  <Text style={styles.angleValue}>{Math.round(angle as number)}°</Text>
                </View>
              ))}
            </View>
          )}
          
          {/* 현재 단계 */}
          <View style={styles.stageIndicator}>
            <Text style={styles.stageIcon}>{getStageIcon(stage)}</Text>
            <Text style={styles.stageText}>{stage.toUpperCase()}</Text>
          </View>
          
          {/* 피드백 영역 */}
          <View style={styles.feedbackContainer}>
            {currentFeedback !== '' && (
              <View style={styles.feedbackCard}>
                <Text style={styles.feedbackText}>✅ {currentFeedback}</Text>
              </View>
            )}
            
            {corrections.length > 0 && (
              <ScrollView style={styles.correctionsScroll}>
                {corrections.map((correction, index) => (
                  <View key={index} style={styles.correctionCard}>
                    <Text style={styles.correctionText}>⚠️ {correction}</Text>
                  </View>
                ))}
              </ScrollView>
            )}
          </View>
          
          {/* 컨트롤 버튼 */}
          <View style={styles.controls}>
            {!isWorkoutActive ? (
              <>
                {/* 운동 선택 */}
                <View style={styles.exerciseSelector}>
                  {['squat', 'pushup', 'plank', 'lunge'].map((type) => (
                    <TouchableOpacity
                      key={type}
                      style={[
                        styles.exerciseOption,
                        exerciseType === type && styles.exerciseOptionActive,
                      ]}
                      onPress={() => setExerciseType(type)}
                    >
                      <Text 
                        style={[
                          styles.exerciseOptionText,
                          exerciseType === type && styles.exerciseOptionTextActive,
                        ]}
                      >
                        {type.toUpperCase()}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                
                <TouchableOpacity style={styles.startButton} onPress={startWorkout}>
                  <Text style={styles.startButtonText}>운동 시작</Text>
                </TouchableOpacity>
              </>
            ) : (
              <TouchableOpacity style={styles.stopButton} onPress={stopWorkout}>
                <Text style={styles.stopButtonText}>운동 종료</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>
      </RNCamera>
    </SafeAreaView>
  );
};

// ========================
// Styles
// ========================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    justifyContent: 'space-between',
  },
  topInfo: {
    padding: 20,
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
  },
  statsCard: {
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    borderRadius: 15,
    padding: 15,
    backdropFilter: 'blur(10px)',
  },
  exerciseTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 15,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
  },
  statLabel: {
    fontSize: 12,
    color: '#aaa',
    marginTop: 5,
  },
  fpsIndicator: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 50 : 20,
    right: 20,
    backgroundColor: 'rgba(0, 255, 0, 0.2)',
    padding: 5,
    borderRadius: 5,
  },
  fpsText: {
    color: '#0f0',
    fontSize: 12,
    fontWeight: 'bold',
  },
  anglesContainer: {
    position: 'absolute',
    left: 20,
    top: '40%',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: 10,
    borderRadius: 10,
  },
  angleItem: {
    flexDirection: 'row',
    marginBottom: 5,
  },
  angleLabel: {
    color: '#aaa',
    fontSize: 12,
    marginRight: 10,
  },
  angleValue: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  stageIndicator: {
    position: 'absolute',
    right: 20,
    top: '40%',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  stageIcon: {
    fontSize: 40,
    marginBottom: 5,
  },
  stageText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  feedbackContainer: {
    paddingHorizontal: 20,
    maxHeight: 150,
  },
  feedbackCard: {
    backgroundColor: 'rgba(76, 175, 80, 0.9)',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
  },
  feedbackText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  correctionsScroll: {
    maxHeight: 100,
  },
  correctionCard: {
    backgroundColor: 'rgba(255, 152, 0, 0.9)',
    padding: 10,
    borderRadius: 8,
    marginBottom: 5,
  },
  correctionText: {
    color: '#fff',
    fontSize: 14,
  },
  controls: {
    padding: 20,
    paddingBottom: Platform.OS === 'ios' ? 40 : 20,
  },
  exerciseSelector: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  exerciseOption: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
  },
  exerciseOptionActive: {
    backgroundColor: 'rgba(102, 126, 234, 0.8)',
  },
  exerciseOptionText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  exerciseOptionTextActive: {
    color: '#fff',
  },
  startButton: {
    backgroundColor: '#4CAF50',
    paddingVertical: 15,
    borderRadius: 30,
    alignItems: 'center',
  },
  startButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  stopButton: {
    backgroundColor: '#F44336',
    paddingVertical: 15,
    borderRadius: 30,
    alignItems: 'center',
  },
  stopButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default WorkoutScreen;