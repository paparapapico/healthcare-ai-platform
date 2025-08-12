"""
올림픽 AI 전체 파이프라인 실행 스크립트
Complete Olympic AI Pipeline Runner
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from datetime import datetime
from typing import Dict, List
import subprocess
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OlympicAIPipeline:
    """올림픽 AI 전체 파이프라인 관리자"""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: 파이프라인 설정 파일 경로
        """
        self.config = self.load_config(config_path)
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def load_config(self, config_path: str) -> Dict:
        """설정 로드"""
        default_config = {
            'data_collection': {
                'youtube': {
                    'enabled': True,
                    'max_videos': 100,
                    'output_dir': 'olympic_dataset'
                },
                'social_media': {
                    'enabled': True,
                    'instagram_posts': 50,
                    'tiktok_videos': 50,
                    'output_dir': 'social_media_dataset'
                },
                'official_sites': {
                    'enabled': True,
                    'output_dir': 'official_sports_dataset'
                },
                'crowdsourcing': {
                    'enabled': True,
                    'api_port': 8001
                }
            },
            'data_processing': {
                'min_quality_score': 60,
                'target_fps': 30,
                'image_size': (640, 480),
                'augmentation': True
            },
            'model_training': {
                'architecture': 'transformer',
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'exercises': ['squat', 'push_up', 'deadlift']
            },
            'optimization': {
                'target_platforms': ['android', 'ios'],
                'quantization': 'int8',
                'target_size_mb': 10
            },
            'deployment': {
                'backend_integration': True,
                'mobile_integration': True,
                'api_endpoints': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                # 재귀적으로 설정 병합
                self.merge_configs(default_config, custom_config)
        
        return default_config
    
    def merge_configs(self, default: Dict, custom: Dict):
        """설정 병합"""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self.merge_configs(default[key], value)
            else:
                default[key] = value
    
    async def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        self.start_time = datetime.now()
        logger.info("="*50)
        logger.info("🚀 올림픽 AI 파이프라인 시작!")
        logger.info("="*50)
        
        try:
            # 1단계: 데이터 수집
            if any(self.config['data_collection'].values()):
                await self.collect_data()
            
            # 2단계: 데이터 전처리
            await self.process_data()
            
            # 3단계: 모델 학습
            await self.train_models()
            
            # 4단계: 모델 최적화
            await self.optimize_models()
            
            # 5단계: 배포
            await self.deploy_models()
            
            # 6단계: 테스트
            await self.test_system()
            
            self.end_time = datetime.now()
            
            # 최종 보고서
            self.generate_final_report()
            
            logger.info("="*50)
            logger.info("✅ 파이프라인 완료!")
            logger.info(f"⏱️ 총 소요 시간: {self.end_time - self.start_time}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류: {e}")
            raise
    
    async def collect_data(self):
        """데이터 수집 단계"""
        logger.info("\n📊 1단계: 데이터 수집 시작...")
        
        collection_tasks = []
        
        # YouTube 올림픽 데이터
        if self.config['data_collection']['youtube']['enabled']:
            logger.info("  - YouTube 올림픽 영상 수집...")
            task = self.run_youtube_scraper()
            collection_tasks.append(task)
        
        # 소셜 미디어 데이터
        if self.config['data_collection']['social_media']['enabled']:
            logger.info("  - Instagram/TikTok 데이터 수집...")
            task = self.run_social_media_scraper()
            collection_tasks.append(task)
        
        # 공식 스포츠 사이트
        if self.config['data_collection']['official_sites']['enabled']:
            logger.info("  - 공식 스포츠 사이트 데이터 수집...")
            task = self.run_official_sites_scraper()
            collection_tasks.append(task)
        
        # 크라우드소싱 플랫폼 시작
        if self.config['data_collection']['crowdsourcing']['enabled']:
            logger.info("  - 크라우드소싱 플랫폼 시작...")
            self.start_crowdsourcing_platform()
        
        # 병렬 실행
        if collection_tasks:
            results = await asyncio.gather(*collection_tasks, return_exceptions=True)
            self.results['data_collection'] = results
        
        logger.info("✅ 데이터 수집 완료!")
        
        # 수집 통계
        self.print_collection_stats()
    
    async def run_youtube_scraper(self):
        """YouTube 스크래퍼 실행"""
        try:
            cmd = [
                'python',
                'ai_models/data_collection/youtube_olympic_scraper.py',
                '--max_videos', str(self.config['data_collection']['youtube']['max_videos']),
                '--output_dir', self.config['data_collection']['youtube']['output_dir']
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("    ✓ YouTube 수집 성공")
                return {'status': 'success', 'source': 'youtube'}
            else:
                logger.error(f"    ✗ YouTube 수집 실패: {stderr.decode()}")
                return {'status': 'failed', 'source': 'youtube', 'error': stderr.decode()}
                
        except Exception as e:
            logger.error(f"YouTube 스크래퍼 오류: {e}")
            return {'status': 'error', 'source': 'youtube', 'error': str(e)}
    
    async def run_social_media_scraper(self):
        """소셜 미디어 스크래퍼 실행"""
        try:
            # social_media_scraper.py의 main 함수 직접 임포트
            sys.path.append('ai_models/data_collection')
            from social_media_scraper import SocialMediaScraper
            
            scraper = SocialMediaScraper(
                output_dir=self.config['data_collection']['social_media']['output_dir']
            )
            
            # Instagram Reels
            instagram_posts = await scraper.scrape_instagram_reels(
                max_posts=self.config['data_collection']['social_media']['instagram_posts']
            )
            
            # TikTok
            tiktok_videos = await scraper.scrape_tiktok_videos(
                max_videos=self.config['data_collection']['social_media']['tiktok_videos']
            )
            
            # 포즈 추출
            await scraper.process_all_videos()
            
            # 보고서 생성
            scraper.generate_report()
            
            logger.info("    ✓ 소셜 미디어 수집 성공")
            return {
                'status': 'success',
                'source': 'social_media',
                'instagram': len(instagram_posts),
                'tiktok': len(tiktok_videos)
            }
            
        except Exception as e:
            logger.error(f"소셜 미디어 스크래퍼 오류: {e}")
            return {'status': 'error', 'source': 'social_media', 'error': str(e)}
    
    async def run_official_sites_scraper(self):
        """공식 사이트 스크래퍼 실행"""
        try:
            sys.path.append('ai_models/data_collection')
            from sports_official_scraper import OfficialSportsScraper
            
            scraper = OfficialSportsScraper(
                output_dir=self.config['data_collection']['official_sites']['output_dir']
            )
            
            # 각 사이트 스크래핑
            await scraper.scrape_olympic_website()
            await scraper.scrape_iwf_website()
            await scraper.scrape_espn()
            await scraper.scrape_crossfit_games()
            
            # 데이터 저장
            scraper.save_collected_data()
            scraper.generate_report()
            
            logger.info("    ✓ 공식 사이트 수집 성공")
            return {
                'status': 'success',
                'source': 'official_sites',
                'athletes': len(scraper.collected_data['athletes']),
                'competitions': len(scraper.collected_data['competitions'])
            }
            
        except Exception as e:
            logger.error(f"공식 사이트 스크래퍼 오류: {e}")
            return {'status': 'error', 'source': 'official_sites', 'error': str(e)}
    
    def start_crowdsourcing_platform(self):
        """크라우드소싱 플랫폼 시작"""
        try:
            # 백그라운드 프로세스로 실행
            cmd = [
                'python',
                'ai_models/data_collection/crowdsource_platform.py'
            ]
            
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"    ✓ 크라우드소싱 플랫폼 시작 (포트: {self.config['data_collection']['crowdsourcing']['api_port']})")
            
        except Exception as e:
            logger.error(f"크라우드소싱 플랫폼 시작 실패: {e}")
    
    async def process_data(self):
        """데이터 전처리 단계"""
        logger.info("\n🔧 2단계: 데이터 전처리 시작...")
        
        # 모든 수집된 데이터 통합
        all_datasets = []
        
        # YouTube 데이터
        youtube_dir = self.config['data_collection']['youtube']['output_dir']
        if os.path.exists(f"{youtube_dir}/training_dataset.json"):
            with open(f"{youtube_dir}/training_dataset.json", 'r') as f:
                youtube_data = json.load(f)
                all_datasets.append(('youtube', youtube_data))
        
        # 소셜 미디어 데이터
        social_dir = self.config['data_collection']['social_media']['output_dir']
        if os.path.exists(social_dir):
            # 포즈 데이터 파일들 수집
            pose_files = [f for f in os.listdir(f"{social_dir}/metadata") if f.endswith('_poses.json')]
            social_data = []
            for pose_file in pose_files:
                with open(f"{social_dir}/metadata/{pose_file}", 'r') as f:
                    social_data.extend(json.load(f))
            all_datasets.append(('social_media', social_data))
        
        # 데이터 정제 및 통합
        processed_data = await self.clean_and_merge_data(all_datasets)
        
        # 데이터 증강
        if self.config['data_processing']['augmentation']:
            processed_data = await self.augment_data(processed_data)
        
        # 학습/검증/테스트 분할
        train_data, val_data, test_data = self.split_data(processed_data)
        
        # 저장
        os.makedirs('ai_models/processed_data', exist_ok=True)
        
        with open('ai_models/processed_data/train_data.json', 'w') as f:
            json.dump(train_data, f)
        
        with open('ai_models/processed_data/val_data.json', 'w') as f:
            json.dump(val_data, f)
        
        with open('ai_models/processed_data/test_data.json', 'w') as f:
            json.dump(test_data, f)
        
        self.results['data_processing'] = {
            'total_samples': len(processed_data),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data)
        }
        
        logger.info(f"  - 총 샘플: {len(processed_data)}")
        logger.info(f"  - 학습: {len(train_data)}, 검증: {len(val_data)}, 테스트: {len(test_data)}")
        logger.info("✅ 데이터 전처리 완료!")
    
    async def clean_and_merge_data(self, datasets: List) -> List:
        """데이터 정제 및 병합"""
        merged_data = []
        
        for source, data in datasets:
            logger.info(f"  - {source} 데이터 정제 중...")
            
            if isinstance(data, dict):
                # 올림픽 티어 데이터 우선
                if 'olympic_tier' in data:
                    merged_data.extend(data['olympic_tier'])
                if 'professional_tier' in data:
                    merged_data.extend(data['professional_tier'])
            elif isinstance(data, list):
                # 품질 필터링
                for item in data:
                    if self.validate_data_quality(item):
                        merged_data.append(item)
        
        return merged_data
    
    def validate_data_quality(self, data_item: Dict) -> bool:
        """데이터 품질 검증"""
        # 최소 품질 점수
        if 'quality_score' in data_item:
            return data_item['quality_score'] >= self.config['data_processing']['min_quality_score']
        
        # 랜드마크 존재 여부
        if 'landmarks' in data_item:
            return data_item['landmarks'] is not None and len(data_item['landmarks']) > 0
        
        return True
    
    async def augment_data(self, data: List) -> List:
        """데이터 증강"""
        logger.info("  - 데이터 증강 중...")
        augmented = []
        
        for item in tqdm(data, desc="Augmenting"):
            augmented.append(item)  # 원본
            
            # 시간 워핑
            if 'landmarks' in item:
                warped = item.copy()
                warped['augmentation'] = 'time_warp'
                augmented.append(warped)
            
            # 노이즈 추가
            noisy = item.copy()
            noisy['augmentation'] = 'noise'
            augmented.append(noisy)
        
        return augmented
    
    def split_data(self, data: List) -> tuple:
        """데이터 분할"""
        np.random.shuffle(data)
        
        total = len(data)
        train_size = int(total * 0.7)
        val_size = int(total * 0.15)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    async def train_models(self):
        """모델 학습 단계"""
        logger.info("\n🧠 3단계: AI 모델 학습 시작...")
        
        training_results = []
        
        for exercise in self.config['model_training']['exercises']:
            logger.info(f"  - {exercise} 모델 학습 중...")
            
            cmd = [
                'python',
                'ai_models/training/train_model.py',
                '--exercise', exercise,
                '--data', 'ai_models/processed_data/train_data.json',
                '--epochs', str(self.config['model_training']['epochs']),
                '--batch_size', str(self.config['model_training']['batch_size']),
                '--architecture', self.config['model_training']['architecture'],
                '--export', 'tflite'
            ]
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"    ✓ {exercise} 모델 학습 완료")
                    training_results.append({
                        'exercise': exercise,
                        'status': 'success'
                    })
                else:
                    logger.error(f"    ✗ {exercise} 모델 학습 실패")
                    training_results.append({
                        'exercise': exercise,
                        'status': 'failed',
                        'error': stderr.decode()
                    })
                    
            except Exception as e:
                logger.error(f"모델 학습 오류: {e}")
                training_results.append({
                    'exercise': exercise,
                    'status': 'error',
                    'error': str(e)
                })
        
        self.results['model_training'] = training_results
        logger.info("✅ 모델 학습 완료!")
    
    async def optimize_models(self):
        """모델 최적화 단계"""
        logger.info("\n⚡ 4단계: 모델 최적화 시작...")
        
        optimization_results = []
        
        for exercise in self.config['model_training']['exercises']:
            model_path = f'models/best_{exercise}_model.h5'
            
            if not os.path.exists(model_path):
                logger.warning(f"  - {exercise} 모델 파일 없음")
                continue
            
            logger.info(f"  - {exercise} 모델 최적화 중...")
            
            cmd = [
                'python',
                'ai_models/optimization/mobile_optimizer.py',
                '--model', model_path,
                '--platform', 'all',
                '--output', f'optimization_report_{exercise}.json'
            ]
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"    ✓ {exercise} 모델 최적화 완료")
                    optimization_results.append({
                        'exercise': exercise,
                        'status': 'success'
                    })
                else:
                    logger.error(f"    ✗ {exercise} 모델 최적화 실패")
                    optimization_results.append({
                        'exercise': exercise,
                        'status': 'failed'
                    })
                    
            except Exception as e:
                logger.error(f"모델 최적화 오류: {e}")
                optimization_results.append({
                    'exercise': exercise,
                    'status': 'error',
                    'error': str(e)
                })
        
        self.results['optimization'] = optimization_results
        logger.info("✅ 모델 최적화 완료!")
    
    async def deploy_models(self):
        """모델 배포 단계"""
        logger.info("\n🚀 5단계: 모델 배포 시작...")
        
        deployment_tasks = []
        
        # 백엔드 API 통합
        if self.config['deployment']['backend_integration']:
            logger.info("  - 백엔드 API 통합...")
            task = self.integrate_backend_api()
            deployment_tasks.append(task)
        
        # 모바일 앱 통합
        if self.config['deployment']['mobile_integration']:
            logger.info("  - 모바일 앱 통합...")
            task = self.integrate_mobile_app()
            deployment_tasks.append(task)
        
        if deployment_tasks:
            results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
            self.results['deployment'] = results
        
        logger.info("✅ 모델 배포 완료!")
    
    async def integrate_backend_api(self):
        """백엔드 API 통합"""
        try:
            # 최적화된 모델을 백엔드로 복사
            os.makedirs('backend/app/ai_models/deployed', exist_ok=True)
            
            for exercise in self.config['model_training']['exercises']:
                tflite_path = f'optimized_models/tflite/model_{exercise}.tflite'
                if os.path.exists(tflite_path):
                    import shutil
                    shutil.copy(
                        tflite_path,
                        f'backend/app/ai_models/deployed/{exercise}_model.tflite'
                    )
            
            logger.info("    ✓ 백엔드 API 통합 완료")
            return {'status': 'success', 'component': 'backend'}
            
        except Exception as e:
            logger.error(f"백엔드 통합 실패: {e}")
            return {'status': 'error', 'component': 'backend', 'error': str(e)}
    
    async def integrate_mobile_app(self):
        """모바일 앱 통합"""
        try:
            # 최적화된 모델을 모바일 앱으로 복사
            os.makedirs('mobile/assets/models', exist_ok=True)
            
            for exercise in self.config['model_training']['exercises']:
                tflite_path = f'optimized_models/tflite/model_{exercise}.tflite'
                if os.path.exists(tflite_path):
                    import shutil
                    shutil.copy(
                        tflite_path,
                        f'mobile/assets/models/{exercise}_model.tflite'
                    )
            
            logger.info("    ✓ 모바일 앱 통합 완료")
            return {'status': 'success', 'component': 'mobile'}
            
        except Exception as e:
            logger.error(f"모바일 통합 실패: {e}")
            return {'status': 'error', 'component': 'mobile', 'error': str(e)}
    
    async def test_system(self):
        """시스템 테스트"""
        logger.info("\n🧪 6단계: 시스템 테스트...")
        
        test_results = {
            'backend_api': False,
            'mobile_app': False,
            'ai_inference': False,
            'crowdsourcing': False
        }
        
        # 백엔드 API 테스트
        try:
            import requests
            response = requests.get('http://localhost:8000/api/v1/health')
            test_results['backend_api'] = response.status_code == 200
            logger.info(f"  - 백엔드 API: {'✓' if test_results['backend_api'] else '✗'}")
        except:
            logger.info("  - 백엔드 API: ✗")
        
        # AI 추론 테스트
        try:
            tflite_path = 'optimized_models/tflite/model_squat.tflite'
            if os.path.exists(tflite_path):
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()
                test_results['ai_inference'] = True
            logger.info(f"  - AI 추론: {'✓' if test_results['ai_inference'] else '✗'}")
        except:
            logger.info("  - AI 추론: ✗")
        
        # 크라우드소싱 플랫폼 테스트
        try:
            import requests
            response = requests.get('http://localhost:8001/stats')
            test_results['crowdsourcing'] = response.status_code == 200
            logger.info(f"  - 크라우드소싱: {'✓' if test_results['crowdsourcing'] else '✗'}")
        except:
            logger.info("  - 크라우드소싱: ✗")
        
        self.results['testing'] = test_results
        
        # 성공률 계산
        success_rate = sum(test_results.values()) / len(test_results) * 100
        logger.info(f"\n테스트 성공률: {success_rate:.1f}%")
        
        if success_rate >= 75:
            logger.info("✅ 시스템 테스트 통과!")
        else:
            logger.warning("⚠️ 일부 테스트 실패 - 확인 필요")
    
    def print_collection_stats(self):
        """수집 통계 출력"""
        if 'data_collection' not in self.results:
            return
        
        logger.info("\n📊 데이터 수집 통계:")
        
        total_videos = 0
        total_frames = 0
        
        for result in self.results['data_collection']:
            if isinstance(result, dict) and result.get('status') == 'success':
                source = result.get('source', 'unknown')
                
                if source == 'youtube':
                    logger.info(f"  - YouTube: 수집 완료")
                elif source == 'social_media':
                    logger.info(f"  - Instagram: {result.get('instagram', 0)}개")
                    logger.info(f"  - TikTok: {result.get('tiktok', 0)}개")
                elif source == 'official_sites':
                    logger.info(f"  - 선수 데이터: {result.get('athletes', 0)}명")
                    logger.info(f"  - 대회 데이터: {result.get('competitions', 0)}개")
    
    def generate_final_report(self):
        """최종 보고서 생성"""
        report = {
            'pipeline_start': self.start_time.isoformat(),
            'pipeline_end': self.end_time.isoformat(),
            'duration': str(self.end_time - self.start_time),
            'results': self.results,
            'config': self.config
        }
        
        # JSON 보고서
        with open('pipeline_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Markdown 보고서
        md_report = f"""# 올림픽 AI 파이프라인 실행 보고서

## 📅 실행 정보
- **시작 시간**: {self.start_time}
- **종료 시간**: {self.end_time}
- **총 소요 시간**: {self.end_time - self.start_time}

## 📊 실행 결과

### 1. 데이터 수집
"""
        
        if 'data_collection' in self.results:
            for result in self.results['data_collection']:
                if isinstance(result, dict):
                    md_report += f"- **{result.get('source', 'unknown')}**: {result.get('status', 'unknown')}\n"
        
        if 'data_processing' in self.results:
            md_report += f"""
### 2. 데이터 전처리
- **총 샘플**: {self.results['data_processing'].get('total_samples', 0)}
- **학습 데이터**: {self.results['data_processing'].get('train_samples', 0)}
- **검증 데이터**: {self.results['data_processing'].get('val_samples', 0)}
- **테스트 데이터**: {self.results['data_processing'].get('test_samples', 0)}
"""
        
        if 'model_training' in self.results:
            md_report += "\n### 3. 모델 학습\n"
            for result in self.results['model_training']:
                md_report += f"- **{result['exercise']}**: {result['status']}\n"
        
        if 'testing' in self.results:
            test_results = self.results['testing']
            md_report += f"""
### 4. 시스템 테스트
- **백엔드 API**: {'✅' if test_results.get('backend_api') else '❌'}
- **AI 추론**: {'✅' if test_results.get('ai_inference') else '❌'}
- **크라우드소싱**: {'✅' if test_results.get('crowdsourcing') else '❌'}
"""
        
        md_report += """
## 🎯 다음 단계
1. 모바일 앱에서 테스트
2. 사용자 피드백 수집
3. 모델 지속적 개선
"""
        
        with open('pipeline_report.md', 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        logger.info(f"\n📄 보고서 저장: pipeline_report.md")


async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Olympic AI Pipeline Runner')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-deployment', action='store_true', help='Skip deployment')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = OlympicAIPipeline(config_path=args.config)
    
    # 옵션에 따라 단계 비활성화
    if args.skip_collection:
        for key in pipeline.config['data_collection']:
            pipeline.config['data_collection'][key]['enabled'] = False
    
    if args.skip_training:
        pipeline.config['model_training']['epochs'] = 0
    
    if args.skip_deployment:
        pipeline.config['deployment']['backend_integration'] = False
        pipeline.config['deployment']['mobile_integration'] = False
    
    await pipeline.run_full_pipeline()


if __name__ == "__main__":
    print("""
    ======================================================
         Olympic AI Full Pipeline Runner
    ======================================================
    
      1. Data Collection (YouTube, SNS, Official Sites)
      2. Data Processing & Augmentation
      3. AI Model Training (Transformer)
      4. Mobile Optimization (TFLite, CoreML)
      5. App Integration & Deployment
      6. System Testing
    
    ======================================================
    """)
    
    asyncio.run(main())