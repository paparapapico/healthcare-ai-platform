import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity, 
  SafeAreaView,
  Alert,
  StatusBar
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';

export default function App() {
  const [showCamera, setShowCamera] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraRef, setCameraRef] = useState<CameraView | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [facing, setFacing] = useState<CameraType>('front');

  const takePicture = async () => {
    if (cameraRef && !isAnalyzing) {
      setIsAnalyzing(true);
      try {
        const photo = await cameraRef.takePictureAsync({
          quality: 0.8,
          base64: true,
        });
        
        // 백엔드 API 호출 시뮬레이션 (나중에 실제 API로 교체)
        setTimeout(() => {
          Alert.alert(
            '🎉 자세 분석 완료!',
            '총점: 87/100\n\n📋 분석 결과:\n✅ 척추 정렬: 양호\n⚠️ 목 자세: 개선 필요\n✅ 어깨 균형: 양호\n\n💡 개선 제안:\n• 목을 뒤로 당기세요\n• 턱을 살짝 당기세요\n• 어깨를 자연스럽게 내리세요',
            [
              { 
                text: '확인', 
                onPress: () => setIsAnalyzing(false) 
              }
            ]
          );
        }, 3000);
        
      } catch (error) {
        Alert.alert('오류', '사진 촬영에 실패했습니다.');
        setIsAnalyzing(false);
      }
    }
  };

  // 홈 화면
  if (!showCamera) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#667eea" />
        
        {/* 헤더 */}
        <View style={styles.header}>
          <Text style={styles.title}>🏥 Healthcare AI</Text>
          <Text style={styles.subtitle}>AI 기반 자세 분석 플랫폼</Text>
        </View>

        {/* 메인 콘텐츠 */}
        <View style={styles.content}>
          <View style={styles.welcomeCard}>
            <Text style={styles.welcomeText}>안녕하세요! 👋</Text>
            <Text style={styles.descText}>
              AI 기술로 당신의 자세를 분석하고{'\n'}
              맞춤형 개선 방안을 제공합니다
            </Text>
          </View>

          {/* 기능 카드들 */}
          <View style={styles.featureGrid}>
            <TouchableOpacity 
              style={[styles.featureCard, styles.primaryCard]}
              onPress={() => setShowCamera(true)}
            >
              <Text style={styles.featureIcon}>📸</Text>
              <Text style={styles.featureTitle}>자세 분석</Text>
              <Text style={styles.featureDesc}>실시간 AI 자세 분석</Text>
            </TouchableOpacity>

            <TouchableOpacity style={[styles.featureCard, styles.secondaryCard]}>
              <Text style={styles.featureIcon}>💪</Text>
              <Text style={styles.featureTitle}>운동 추천</Text>
              <Text style={styles.featureDesc}>맞춤형 운동법</Text>
            </TouchableOpacity>
          </View>

          {/* 통계 */}
          <View style={styles.statsCard}>
            <Text style={styles.statsTitle}>📊 오늘의 건강 지수</Text>
            <View style={styles.statsRow}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>87</Text>
                <Text style={styles.statLabel}>자세 점수</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>5</Text>
                <Text style={styles.statLabel}>분석 횟수</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>3</Text>
                <Text style={styles.statLabel}>연속 일수</Text>
              </View>
            </View>
          </View>
        </View>

        {/* 빠른 분석 버튼 */}
        <TouchableOpacity 
          style={styles.quickAnalyzeBtn}
          onPress={() => setShowCamera(true)}
        >
          <Text style={styles.quickAnalyzeText}>🚀 빠른 자세 분석 시작</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  // 카메라 권한 체크
  if (!permission) {
    // Camera permissions are still loading
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.loadingText}>카메라 권한을 확인하는 중...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>📷 카메라 접근 권한이 필요합니다</Text>
        <TouchableOpacity 
          style={styles.permissionBtn}
          onPress={requestPermission}
        >
          <Text style={styles.permissionBtnText}>권한 허용</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.backBtn}
          onPress={() => setShowCamera(false)}
        >
          <Text style={styles.backBtnText}>뒤로 가기</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // 카메라 화면
  return (
    <SafeAreaView style={styles.cameraContainer}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      
      {/* 카메라 헤더 */}
      <View style={styles.cameraHeader}>
        <TouchableOpacity onPress={() => setShowCamera(false)}>
          <Text style={styles.backButton}>← 뒤로</Text>
        </TouchableOpacity>
        <Text style={styles.cameraTitle}>🔍 실시간 자세 분석</Text>
        <View style={styles.placeholder} />
      </View>

      {/* 카메라 뷰 */}
      <View style={styles.cameraWrapper}>
        <CameraView
          style={styles.camera}
          facing={facing}
          ref={setCameraRef}
        >
          {/* 가이드 오버레이 */}
          <View style={styles.cameraOverlay}>
            <View style={styles.guideBox}>
              <Text style={styles.guideText}>
                📋 자세 분석 가이드{'\n'}
                • 전신이 화면에 나오도록 조정{'\n'}
                • 편안한 자세로 서기{'\n'}
                • 조명이 밝은 곳에서 촬영
              </Text>
            </View>
          </View>
        </CameraView>
      </View>

      {/* 카메라 컨트롤 */}
      <View style={styles.cameraControls}>
        <TouchableOpacity 
          style={[
            styles.analyzeButton, 
            isAnalyzing && styles.analyzingButton
          ]}
          onPress={takePicture}
          disabled={isAnalyzing}
        >
          <Text style={styles.analyzeButtonText}>
            {isAnalyzing ? '🔄 AI 분석 중...' : '📸 자세 분석하기'}
          </Text>
        </TouchableOpacity>

        <Text style={styles.instructions}>
          {isAnalyzing 
            ? 'AI가 당신의 자세를 분석하고 있습니다...' 
            : '준비가 되면 분석 버튼을 눌러주세요'
          }
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    backgroundColor: '#667eea',
    paddingTop: 20,
    paddingBottom: 30,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: 'white',
    opacity: 0.9,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  welcomeCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  welcomeText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  descText: {
    fontSize: 16,
    color: '#666',
    lineHeight: 24,
  },
  featureGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  featureCard: {
    width: '48%',
    aspectRatio: 1,
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  primaryCard: {
    backgroundColor: '#ff6b6b',
  },
  secondaryCard: {
    backgroundColor: '#4ecdc4',
  },
  featureIcon: {
    fontSize: 32,
    marginBottom: 10,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  featureDesc: {
    fontSize: 12,
    color: 'white',
    textAlign: 'center',
    opacity: 0.9,
  },
  statsCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#667eea',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  quickAnalyzeBtn: {
    backgroundColor: '#667eea',
    marginHorizontal: 20,
    marginBottom: 20,
    borderRadius: 16,
    paddingVertical: 18,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 4,
  },
  quickAnalyzeText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  // 카메라 스타일
  cameraContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  cameraHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: 'rgba(0,0,0,0.8)',
  },
  backButton: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  cameraTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  placeholder: {
    width: 50,
  },
  cameraWrapper: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'flex-start',
    alignItems: 'center',
    paddingTop: 50,
  },
  guideBox: {
    backgroundColor: 'rgba(0,0,0,0.8)',
    padding: 20,
    borderRadius: 12,
    marginHorizontal: 20,
  },
  guideText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 14,
    lineHeight: 20,
  },
  cameraControls: {
    backgroundColor: 'rgba(0,0,0,0.9)',
    padding: 30,
    alignItems: 'center',
  },
  analyzeButton: {
    backgroundColor: '#ff6b6b',
    paddingHorizontal: 40,
    paddingVertical: 15,
    borderRadius: 30,
    marginBottom: 15,
  },
  analyzingButton: {
    backgroundColor: '#666',
  },
  analyzeButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  instructions: {
    color: 'white',
    textAlign: 'center',
    fontSize: 14,
    opacity: 0.8,
  },
  // 공통 스타일
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
    padding: 20,
  },
  loadingText: {
    fontSize: 16,
    color: '#666',
  },
  errorText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 30,
  },
  permissionBtn: {
    backgroundColor: '#667eea',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 8,
    marginBottom: 15,
  },
  permissionBtnText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  backBtn: {
    backgroundColor: '#666',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 8,
  },
  backBtnText: {
    color: 'white',
    fontSize: 16,
  },
});