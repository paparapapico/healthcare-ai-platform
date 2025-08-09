"""
Simple Locust Load Test for Healthcare AI
매우 간단한 부하 테스트 파일
"""

from locust import HttpUser, task, between
import random
import time

class SimpleUser(HttpUser):
    """간단한 사용자 테스트"""
    wait_time = between(1, 3)
    
    def on_start(self):
        """테스트 시작 시 사용자 등록"""
        timestamp = int(time.time())
        random_num = random.randint(1000, 9999)
        
        self.user_email = f"test{timestamp}{random_num}@example.com"
        self.user_password = "test123"
        
        # 회원가입
        user_data = {
            "email": self.user_email,
            "password": self.user_password,
            "name": f"Test User {random_num}",
            "birth_date": "1990-01-01",
            "gender": "male",
            "height": 175,
            "weight": 70.0
        }
        
        response = self.client.post("/api/v1/auth/register", json=user_data)
        if response.status_code == 200:
            print(f"✅ User registered: {self.user_email}")
            
            # 로그인
            login_data = {
                "username": self.user_email,
                "password": self.user_password
            }
            
            login_response = self.client.post(
                "/api/v1/auth/login",
                data=login_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if login_response.status_code == 200:
                token_data = login_response.json()
                self.access_token = token_data.get("access_token")
                self.auth_headers = {"Authorization": f"Bearer {self.access_token}"}
                print(f"✅ User logged in: {self.user_email}")
            else:
                self.auth_headers = {}
        else:
            self.auth_headers = {}
    
    @task(5)
    def visit_homepage(self):
        """홈페이지 방문"""
        self.client.get("/")
    
    @task(3)
    def check_health_dashboard(self):
        """건강 대시보드 확인"""
        self.client.get("/api/v1/health/dashboard", headers=self.auth_headers)
    
    @task(2)
    def view_workouts(self):
        """운동 목록 보기"""
        self.client.get("/api/v1/workouts", headers=self.auth_headers)
    
    @task(1)
    def create_workout(self):
        """운동 생성"""
        workout_data = {
            "exercise_type": "squat",
            "duration": 300,
            "reps": 20,
            "calories_burned": 100.0,
            "form_score": 85.0
        }
        self.client.post("/api/v1/workouts", json=workout_data, headers=self.auth_headers)
    
    @task(1)
    def log_water(self):
        """물 섭취 기록"""
        self.client.post("/api/v1/health/water/log?amount_ml=250", headers=self.auth_headers)

class QuickUser(HttpUser):
    """빠른 테스트용 사용자"""
    wait_time = between(0.5, 1.0)
    
    @task
    def quick_homepage_check(self):
        """빠른 홈페이지 체크"""
        self.client.get("/")
    
    @task  
    def quick_health_check(self):
        """빠른 헬스체크"""
        self.client.get("/health")