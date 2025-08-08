import cv2
import mediapipe as mp
import numpy as np
import time

class PoseDetector:
    """AI 자세 인식 클래스"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def calculate_angle(self, a, b, c):
        """세 점을 이용한 각도 계산"""
        a = np.array(a)  # 첫 번째 점
        b = np.array(b)  # 중간 점 (관절)
        c = np.array(c)  # 세 번째 점
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def detect_squat(self, landmarks):
        """스쿼트 자세 감지 및 분석"""
        # 주요 관절 포인트 추출
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # 무릎 각도 계산
        angle = self.calculate_angle(hip, knee, ankle)
        
        # 스쿼트 단계 판정
        squat_stage = None
        feedback = ""
        
        if angle > 160:
            squat_stage = "UP"
            feedback = "시작 자세 - 준비되었습니다"
        elif angle < 90:
            squat_stage = "DOWN"
            feedback = "좋습니다! 충분히 내려갔습니다"
        else:
            squat_stage = "MIDDLE"
            feedback = f"무릎 각도: {angle:.1f}° - 조금 더 내려가세요"
            
        return angle, squat_stage, feedback
    
    def run_webcam(self):
        """웹캠으로 실시간 자세 분석"""
        cap = cv2.VideoCapture(0)
        
        # 스쿼트 카운터
        counter = 0 
        stage = None
        
        # FPS 계산
        prev_time = 0
        
        print("🎥 웹캠 시작! 'q'를 누르면 종료됩니다.")
        print("📌 스쿼트 자세를 취해보세요!")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # BGR을 RGB로 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 포즈 감지
            results = self.pose.process(image)
            
            # RGB를 BGR로 다시 변환
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # 스쿼트 감지
                angle, squat_stage, feedback = self.detect_squat(landmarks)
                
                # 카운터 로직
                if squat_stage == "DOWN" and stage == "UP":
                    stage = "DOWN"
                elif squat_stage == "UP" and stage == "DOWN":
                    counter += 1
                    stage = "UP"
                    print(f"✅ 스쿼트 완료! 총 {counter}개")
                elif stage is None:
                    stage = squat_stage
                
                # 화면에 정보 표시
                # 상태 박스
                cv2.rectangle(image, (0, 0), (350, 120), (245, 117, 16), -1)
                
                # 카운터
                cv2.putText(image, 'REPS', (15, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (15, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 각도
                cv2.putText(image, 'ANGLE', (120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f'{int(angle)}', (120, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 상태
                cv2.putText(image, 'STAGE', (220, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage if stage else "", (220, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 피드백
                cv2.rectangle(image, (0, 420), (640, 480), (0, 255, 0), -1)
                cv2.putText(image, feedback, (10, 455),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                
                # FPS 표시
                cv2.putText(image, f'FPS: {int(fps)}', (500, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
            except:
                pass
            
            # 포즈 스켈레톤 그리기
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            # 화면 표시
            cv2.imshow('AI Pose Detection - Squat Counter', image)
            
            # 'q' 키로 종료
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        print(f"\n🏁 운동 종료!")
        print(f"📊 총 스쿼트 개수: {counter}개")
        
        cap.release()
        cv2.destroyAllWindows()

def test_basic_detection():
    """기본 포즈 감지 테스트"""
    print("=" * 50)
    print("🤖 AI 자세 인식 POC 테스트")
    print("=" * 50)
    
    mp_pose = mp.solutions.pose
    
    # 정적 이미지 테스트
    print("\n1. MediaPipe 설치 확인... ✅")
    print(f"   - MediaPipe 버전: {mp.__version__}")
    
    # OpenCV 테스트
    print("\n2. OpenCV 설치 확인... ✅")
    print(f"   - OpenCV 버전: {cv2.__version__}")
    
    # 카메라 연결 테스트
    print("\n3. 카메라 연결 테스트...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("   - 카메라 연결 성공! ✅")
        cap.release()
    else:
        print("   - 카메라 연결 실패 ❌")
        return False
    
    print("\n✨ 모든 테스트 통과! POC 실행 준비 완료")
    return True

if __name__ == "__main__":
    # 기본 테스트 실행
    if test_basic_detection():
        print("\n" + "=" * 50)
        print("🎬 실시간 스쿼트 자세 분석을 시작합니다!")
        print("=" * 50)
        
        # 포즈 감지기 실행
        detector = PoseDetector()
        detector.run_webcam()
    else:
        print("\n❌ 환경 설정을 확인해주세요.")
        print("필요한 패키지:")
        print("  pip install mediapipe opencv-python numpy")