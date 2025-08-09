"""
API Documentation Configuration
Swagger/OpenAPI 문서 설정
파일 위치: backend/app/api/docs.py
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import HTMLResponse

def custom_openapi(app: FastAPI):
    """커스텀 OpenAPI 스키마"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="AI Healthcare Platform API",
        version="1.0.0",
        description="""
        ## 🏥 AI 헬스케어 플랫폼 API
        
        AI 기반 통합 헬스케어 플랫폼의 RESTful API 문서입니다.
        
        ### 주요 기능
        
        * **인증** - JWT 기반 사용자 인증
        * **운동 분석** - AI 실시간 자세 분석
        * **건강 추적** - 종합 건강 데이터 관리
        * **소셜** - 친구, 챌린지, 리더보드
        * **알림** - 푸시 알림 관리
        
        ### 인증 방법
        
        1. `/api/v1/auth/register`로 회원가입
        2. `/api/v1/auth/login`으로 로그인하여 토큰 획득
        3. `Authorization: Bearer {token}` 헤더 추가
        
        ### Rate Limiting
        
        - 일반 API: 60 requests/minute
        - 인증 API: 5 requests/minute
        
        ### 응답 형식
        
        모든 응답은 JSON 형식이며, 다음 구조를 따릅니다:
        
        ```json
        {
            "success": true,
            "data": {...},
            "message": "Success",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        ```
        
        ### 에러 코드
        
        | 코드 | 설명 |
        |------|------|
        | 400 | Bad Request - 잘못된 요청 |
        | 401 | Unauthorized - 인증 필요 |
        | 403 | Forbidden - 권한 없음 |
        | 404 | Not Found - 리소스 없음 |
        | 429 | Too Many Requests - Rate Limit 초과 |
        | 500 | Internal Server Error - 서버 오류 |
        
        ### 문의
        
        API 관련 문의: api@healthcare-app.com
        """,
        routes=app.routes,
        tags=[
            {
                "name": "auth",
                "description": "인증 관련 API",
            },
            {
                "name": "users",
                "description": "사용자 관리 API",
            },
            {
                "name": "workouts",
                "description": "운동 관련 API",
            },
            {
                "name": "health",
                "description": "건강 데이터 API",
            },
            {
                "name": "social",
                "description": "소셜 기능 API",
            },
            {
                "name": "notifications",
                "description": "알림 관리 API",
            },
        ],
        servers=[
            {"url": "http://localhost:8000", "description": "Development server"},
            {"url": "https://api.healthcare-app.com", "description": "Production server"},
        ],
    )
    
    # Security Schemes
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def setup_docs(app: FastAPI):
    """문서 엔드포인트 설정"""
    
    # Custom OpenAPI schema
    app.openapi = lambda: custom_openapi(app)
    
    # Custom Swagger UI
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        )
    
    # Custom ReDoc
    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " - ReDoc",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js",
        )
    
    # API Specification download
    @app.get("/api/openapi.json", include_in_schema=False)
    async def get_open_api_endpoint():
        return custom_openapi(app)
    
    # API Status Page
    @app.get("/api/status", include_in_schema=False, response_class=HTMLResponse)
    async def api_status():
        return """
        <html>
            <head>
                <title>API Status</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 50px auto;
                        padding: 20px;
                    }
                    .status { 
                        color: #4CAF50; 
                        font-weight: bold;
                    }
                    .endpoint {
                        background: #f5f5f5;
                        padding: 10px;
                        margin: 10px 0;
                        border-radius: 5px;
                    }
                </style>
            </head>
            <body>
                <h1>🏥 Healthcare API Status</h1>
                <p>Status: <span class="status">OPERATIONAL</span></p>
                <h2>Available Endpoints:</h2>
                <div class="endpoint">📚 Documentation: <a href="/docs">/docs</a></div>
                <div class="endpoint">📖 ReDoc: <a href="/redoc">/redoc</a></div>
                <div class="endpoint">📊 Metrics: <a href="/metrics">/metrics</a></div>
                <div class="endpoint">❤️ Health Check: <a href="/health">/health</a></div>
            </body>
        </html>
        """