from fastapi.testclient import TestClient
from app.main import app
from datetime import datetime, date
import pytest
import time
import random
import json


client = TestClient(app)

def generate_unique_email(prefix="test"):
    """고유한 이메일 생성"""
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)
    return f"{prefix}{timestamp}{random_num}@example.com"

def debug_response(response, test_name):
    """응답을 자세히 분석하는 함수"""
    print(f"\n=== {test_name} 디버그 정보 ===")
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    try:
        response_json = response.json()
        print(f"Response JSON: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
        
        # 에러 세부사항 분석
        if "detail" in response_json:
            detail = response_json["detail"]
            print(f"\n상세 에러: {detail}")
            
            # Pydantic 검증 에러인지 확인
            if "validation errors" in detail:
                print("🔍 Pydantic 검증 에러 감지!")
                print("이는 API 응답 모델에 필수 필드가 누락된 것을 의미합니다.")
                
                # input_value 파싱해서 실제로 어떤 데이터가 반환되는지 확인
                if "input_value" in detail:
                    import re
                    input_matches = re.findall(r"input_value=({[^}]+})", detail)
                    for i, match in enumerate(input_matches):
                        print(f"실제 반환된 데이터 {i+1}: {match}")
                        
    except json.JSONDecodeError:
        print(f"Response Text (JSON 파싱 실패): {response.text}")
    except Exception as e:
        print(f"응답 분석 중 에러: {e}")
        print(f"Raw Response: {response.text}")

def test_register_clean():
    """깨끗한 회원가입 테스트"""
    print("\n" + "="*50)
    print("깨끗한 회원가입 테스트")
    print("="*50)
    
    clean_user = {
        "email": generate_unique_email("clean"),
        "password": "testpass123",
        "name": "Clean User", 
        "birth_date": "1990-01-01",
        "gender": "male",
        "height": 175,
        "weight": 70.0
    }
    
    print(f"📤 요청 데이터: {json.dumps(clean_user, indent=2, ensure_ascii=False)}")
    
    try:
        response = client.post("/api/users", json=clean_user)
        debug_response(response, "깨끗한 회원가입")
        
        if response.status_code in [200, 201]:
            print("✅ 회원가입 성공!")
            return True
        else:
            print(f"❌ 회원가입 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"💥 예외 발생: {e}")
        return False

def test_register_with_basic():
    """BASIC subscription_tier로 테스트"""
    print("\n" + "="*50)
    print("BASIC subscription_tier 테스트")
    print("="*50)
    
    user_with_basic = {
        "email": generate_unique_email("basic"),
        "password": "testpass123",
        "name": "Basic User",
        "birth_date": "1990-01-01",
        "gender": "male",
        "height": 175,
        "weight": 70.0,
        "subscription_tier": "BASIC"
    }
    
    print(f"📤 요청 데이터: {json.dumps(user_with_basic, indent=2, ensure_ascii=False)}")
    
    try:
        response = client.post("/api/v1/auth/register", json=user_with_basic)
        debug_response(response, "BASIC subscription_tier")
        
        if response.status_code in [200, 201]:
            print("✅ 회원가입 성공!")
            return True
        else:
            print(f"❌ 회원가입 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"💥 예외 발생: {e}")
        return False

def test_register_multiple():
    """여러 번 회원가입 테스트"""
    print("\n" + "="*50)
    print("여러 번 회원가입 테스트")
    print("="*50)
    
    success_count = 0
    total_tests = 3
    
    for i in range(total_tests):
        user = {
            "email": generate_unique_email(f"user{i}"),
            "password": "testpass123",
            "name": f"Test User {i+1}",
            "birth_date": "1990-01-01",
            "gender": "male",
            "height": 175,
            "weight": 70.0
        }
        
        print(f"\n--- 테스트 {i+1}/{total_tests} ---")
        print(f"📧 이메일: {user['email']}")
        
        try:
            response = client.post("/api/v1/auth/register", json=user)
            
            if response.status_code in [200, 201]:
                print(f"✅ 회원가입 {i+1} 성공!")
                success_count += 1
            else:
                print(f"❌ 회원가입 {i+1} 실패: {response.status_code}")
                debug_response(response, f"회원가입 {i+1}")
                
        except Exception as e:
            print(f"💥 예외 발생: {e}")
    
    print(f"\n📊 결과 요약: {success_count}/{total_tests} 성공")
    return success_count == total_tests

def test_api_health():
    """API 기본 상태 확인"""
    print("\n" + "="*50)
    print("API 상태 확인")
    print("="*50)
    
    try:
        # 기본 헬스체크 (있다면)
        health_endpoints = ["/", "/health", "/api/v1/health"]
        
        for endpoint in health_endpoints:
            try:
                response = client.get(endpoint)
                print(f"📡 {endpoint}: {response.status_code}")
                if response.status_code == 200:
                    return True
            except:
                continue
                
        # 회원가입 엔드포인트가 존재하는지 확인 (잘못된 요청으로)
        response = client.post("/api/v1/auth/register", json={})
        print(f"📡 /api/v1/auth/register (빈 요청): {response.status_code}")
        
        # 422는 validation error, 즉 엔드포인트는 존재함
        if response.status_code == 422:
            print("✅ 회원가입 엔드포인트 존재 확인")
            return True
        elif response.status_code == 404:
            print("❌ 회원가입 엔드포인트를 찾을 수 없음")
            return False
        else:
            print(f"🤔 예상치 못한 응답: {response.status_code}")
            debug_response(response, "API 상태 확인")
            return True
            
    except Exception as e:
        print(f"💥 API 상태 확인 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 HealthcareAI 백엔드 테스트 시작")
    print("=" * 60)
    
    # API 상태 확인부터
    if not test_api_health():
        print("❌ API 상태 확인 실패. 서버가 실행 중인지 확인하세요.")
        return
    
    # 각 테스트 실행
    tests = [
        ("깨끗한 회원가입", test_register_clean),
        ("BASIC 구독", test_register_with_basic), 
        ("다중 회원가입", test_register_multiple)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
    
    # 최종 결과 요약
    print("\n" + "="*60)
    print("🏁 최종 테스트 결과")
    print("="*60)
    
    success_count = 0
    for test_name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{test_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n📊 전체 결과: {success_count}/{len(results)} 테스트 통과")
    
    if success_count == 0:
        print("\n💡 문제 해결 힌트:")
        print("1. 백엔드 서버가 실행 중인지 확인")
        print("2. UserResponse 모델에 필수 필드들이 포함되었는지 확인")
        print("3. 데이터베이스 연결 상태 확인")
        print("4. 로그에서 상세한 에러 메시지 확인")

if __name__ == "__main__":
    main()