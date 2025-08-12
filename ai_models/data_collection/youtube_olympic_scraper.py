"""
유튜브에서 올림픽 선수 운동 영상 자동 수집 및 분석
YouTube Olympic Athletes Video Scraper and Analyzer
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pytube import YouTube, Search
import requests
from bs4 import BeautifulSoup
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import asyncio
import aiohttp
from tqdm import tqdm
import hashlib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OlympicDataCollector:
    """올림픽 선수 데이터 자동 수집기"""
    
    # 수집할 올림픽 선수 및 키워드
    OLYMPIC_ATHLETES = {
        'weightlifting': [
            'Lasha Talakhadze',  # 조지아, 역도 금메달리스트
            'Shi Zhiyong',       # 중국, 역도 금메달리스트
            'Lu Xiaojun',        # 중국, 역도 전설
            'Karlos Nasar',      # 불가리아, 신예 역도 선수
        ],
        'gymnastics': [
            'Simone Biles',      # 미국, 체조 GOAT
            'Kohei Uchimura',    # 일본, 체조 왕
            'Nadia Comaneci',    # 루마니아, 체조 전설
        ],
        'powerlifting': [
            'Eddie Hall',        # 영국, 스트롱맨
            'Hafthor Bjornsson', # 아이슬란드, 스트롱맨
            'Brian Shaw',        # 미국, 스트롱맨
        ],
        'crossfit': [
            'Mat Fraser',        # 미국, 크로스핏 챔피언
            'Tia-Clair Toomey',  # 호주, 크로스핏 챔피언
            'Rich Froning',      # 미국, 크로스핏 레전드
        ]
    }
    
    # 운동별 검색 키워드
    EXERCISE_KEYWORDS = {
        'squat': ['olympic squat', 'ATG squat', 'front squat', 'back squat', 'squat technique'],
        'deadlift': ['olympic deadlift', 'clean deadlift', 'snatch deadlift', 'deadlift form'],
        'bench_press': ['bench press technique', 'powerlifting bench', 'olympic bench press'],
        'clean_and_jerk': ['clean and jerk', 'olympic lifting', 'weightlifting technique'],
        'snatch': ['olympic snatch', 'snatch technique', 'weightlifting snatch']
    }
    
    def __init__(self, output_dir: str = 'olympic_dataset'):
        """
        Args:
            output_dir: 데이터 저장 디렉토리
        """
        self.output_dir = output_dir
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 데이터 저장 경로 생성
        os.makedirs(f'{output_dir}/videos', exist_ok=True)
        os.makedirs(f'{output_dir}/annotations', exist_ok=True)
        os.makedirs(f'{output_dir}/poses', exist_ok=True)
        
        self.collected_videos = []
        self.processed_data = []
    
    async def collect_olympic_videos(self, max_videos_per_athlete: int = 5) -> List[Dict]:
        """올림픽 선수 영상 자동 수집"""
        logger.info("Starting Olympic athlete video collection...")
        
        all_videos = []
        
        for sport, athletes in self.OLYMPIC_ATHLETES.items():
            for athlete in athletes:
                logger.info(f"Searching for {athlete} ({sport})...")
                
                # 운동별 키워드와 조합
                for exercise, keywords in self.EXERCISE_KEYWORDS.items():
                    for keyword in keywords[:2]:  # 키워드당 2개씩
                        search_query = f"{athlete} {keyword} technique analysis"
                        videos = await self.search_youtube_videos(
                            search_query, 
                            max_results=max_videos_per_athlete
                        )
                        
                        for video in videos:
                            video['athlete'] = athlete
                            video['sport'] = sport
                            video['exercise'] = exercise
                            video['quality_tier'] = 'olympic'  # 올림픽 수준
                            all_videos.append(video)
        
        # 공식 올림픽 채널 영상도 수집
        official_channels = [
            'Olympics',
            'International Weightlifting Federation',
            'FIG Channel',
            'IPF Powerlifting'
        ]
        
        for channel in official_channels:
            for exercise, keywords in self.EXERCISE_KEYWORDS.items():
                search_query = f"{channel} {keywords[0]} slow motion"
                videos = await self.search_youtube_videos(
                    search_query,
                    max_results=10
                )
                
                for video in videos:
                    video['channel'] = channel
                    video['exercise'] = exercise
                    video['quality_tier'] = 'official'
                    all_videos.append(video)
        
        self.collected_videos = all_videos
        logger.info(f"Collected {len(all_videos)} Olympic-level videos")
        
        return all_videos
    
    async def search_youtube_videos(self, query: str, max_results: int = 10) -> List[Dict]:
        """YouTube 영상 검색"""
        try:
            search = Search(query)
            videos = []
            
            for i, video in enumerate(search.results[:max_results]):
                try:
                    video_info = {
                        'title': video.title,
                        'url': video.watch_url,
                        'video_id': video.video_id,
                        'duration': video.length,
                        'views': video.views,
                        'channel': video.author,
                        'description': video.description[:500] if video.description else '',
                        'search_query': query,
                        'quality_score': self.estimate_video_quality(video)
                    }
                    videos.append(video_info)
                except Exception as e:
                    logger.warning(f"Error processing video: {e}")
                    continue
            
            return videos
            
        except Exception as e:
            logger.error(f"YouTube search failed for '{query}': {e}")
            return []
    
    def estimate_video_quality(self, video) -> float:
        """영상 품질 점수 추정"""
        score = 50.0  # 기본 점수
        
        # 조회수 기반 점수
        if video.views > 1000000:
            score += 20
        elif video.views > 100000:
            score += 10
        elif video.views > 10000:
            score += 5
        
        # 채널 신뢰도
        trusted_channels = ['Olympics', 'IWF', 'IPF', 'CrossFit']
        if any(channel in video.author for channel in trusted_channels):
            score += 15
        
        # 제목 키워드
        quality_keywords = ['technique', 'analysis', 'slow motion', 'olympic', 'world record']
        for keyword in quality_keywords:
            if keyword.lower() in video.title.lower():
                score += 3
        
        # 영상 길이 (적절한 길이)
        if 60 <= video.length <= 600:  # 1-10분
            score += 10
        
        return min(100, score)
    
    def download_and_process_video(self, video_info: Dict, output_path: str) -> Dict:
        """영상 다운로드 및 포즈 추출"""
        try:
            logger.info(f"Processing: {video_info['title']}")
            
            # YouTube 영상 다운로드
            yt = YouTube(video_info['url'])
            
            # 최고 품질 스트림 선택 (720p 이상 권장)
            stream = yt.streams.filter(
                progressive=True, 
                file_extension='mp4'
            ).order_by('resolution').desc().first()
            
            if not stream:
                stream = yt.streams.filter(file_extension='mp4').first()
            
            if not stream:
                logger.warning(f"No suitable stream found for {video_info['title']}")
                return None
            
            # 비디오 다운로드
            video_path = stream.download(
                output_path=f"{self.output_dir}/videos",
                filename=f"{video_info['video_id']}.mp4"
            )
            
            # 포즈 추출 및 분석
            pose_data = self.extract_poses_from_video(video_path, video_info)
            
            # 데이터 품질 검증
            if self.validate_pose_data(pose_data):
                # 메타데이터 저장
                metadata = {
                    'video_info': video_info,
                    'video_path': video_path,
                    'pose_data_path': f"{self.output_dir}/poses/{video_info['video_id']}.json",
                    'extraction_date': datetime.now().isoformat(),
                    'total_frames': len(pose_data),
                    'valid_frames': sum(1 for p in pose_data if p['landmarks']),
                    'quality_metrics': self.calculate_quality_metrics(pose_data)
                }
                
                # 포즈 데이터 저장
                with open(metadata['pose_data_path'], 'w') as f:
                    json.dump(pose_data, f, indent=2)
                
                # 메타데이터 저장
                with open(f"{self.output_dir}/annotations/{video_info['video_id']}.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"✅ Successfully processed: {video_info['title']}")
                return metadata
            else:
                logger.warning(f"Poor quality data from: {video_info['title']}")
                # 품질이 낮은 비디오는 삭제
                os.remove(video_path)
                return None
                
        except Exception as e:
            logger.error(f"Failed to process video {video_info['url']}: {e}")
            return None
    
    def extract_poses_from_video(self, video_path: str, video_info: Dict) -> List[Dict]:
        """비디오에서 포즈 데이터 추출"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pose_data = []
        frame_count = 0
        
        # 프로그레스 바
        pbar = tqdm(total=total_frames, desc="Extracting poses")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 포즈 감지
            results = self.pose.process(rgb_frame)
            
            frame_data = {
                'frame_id': frame_count,
                'timestamp': frame_count / fps,
                'video_id': video_info['video_id'],
                'athlete': video_info.get('athlete', 'Unknown'),
                'exercise': video_info.get('exercise', 'Unknown'),
                'landmarks': None,
                'world_landmarks': None,
                'visibility_score': 0
            }
            
            if results.pose_landmarks:
                # 2D 랜드마크
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_data['landmarks'] = landmarks
                
                # 3D 월드 랜드마크
                if results.pose_world_landmarks:
                    world_landmarks = []
                    for landmark in results.pose_world_landmarks.landmark:
                        world_landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    frame_data['world_landmarks'] = world_landmarks
                
                # 가시성 점수 계산
                frame_data['visibility_score'] = np.mean([l['visibility'] for l in landmarks])
                
                # 운동 단계 자동 라벨링
                frame_data['phase'] = self.detect_exercise_phase(landmarks, video_info.get('exercise'))
                
                # 품질 점수 추정
                frame_data['quality_score'] = self.estimate_form_quality(landmarks, video_info.get('exercise'))
            
            pose_data.append(frame_data)
            frame_count += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        return pose_data
    
    def detect_exercise_phase(self, landmarks: List[Dict], exercise: str) -> str:
        """운동 단계 자동 감지"""
        if not landmarks or len(landmarks) < 33:
            return 'unknown'
        
        # MediaPipe 랜드마크 인덱스
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
        if exercise in ['squat', 'front_squat', 'back_squat']:
            # 무릎 각도로 스쿼트 단계 판단
            knee_angle = self.calculate_angle(
                landmarks[LEFT_HIP],
                landmarks[LEFT_KNEE],
                landmarks[LEFT_ANKLE]
            )
            
            if knee_angle > 160:
                return 'standing'
            elif knee_angle > 120:
                return 'descent'
            elif knee_angle > 70:
                return 'bottom'
            else:
                return 'ascent'
                
        elif exercise in ['bench_press']:
            # 팔꿈치 각도로 벤치프레스 단계 판단
            elbow_angle = self.calculate_angle(
                landmarks[LEFT_SHOULDER],
                landmarks[LEFT_ELBOW],
                landmarks[LEFT_WRIST]
            )
            
            if elbow_angle > 160:
                return 'lockout'
            elif elbow_angle > 90:
                return 'descent'
            elif elbow_angle > 70:
                return 'bottom'
            else:
                return 'press'
                
        elif exercise in ['deadlift', 'clean_and_jerk', 'snatch']:
            # 엉덩이 높이로 데드리프트 단계 판단
            hip_height = (landmarks[LEFT_HIP]['y'] + landmarks[RIGHT_HIP]['y']) / 2
            knee_height = (landmarks[LEFT_KNEE]['y'] + landmarks[RIGHT_KNEE]['y']) / 2
            
            if hip_height < knee_height:
                return 'lockout'
            elif hip_height < knee_height + 0.1:
                return 'pull'
            else:
                return 'setup'
        
        return 'unknown'
    
    def calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """3점 사이의 각도 계산"""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def estimate_form_quality(self, landmarks: List[Dict], exercise: str) -> float:
        """올림픽 수준 기준으로 자세 품질 평가"""
        if not landmarks or len(landmarks) < 33:
            return 0.0
        
        score = 100.0
        
        # 기본 체크: 주요 관절이 보이는지
        key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        visibility_score = np.mean([landmarks[i]['visibility'] for i in key_joints])
        
        if visibility_score < 0.7:
            score -= 20
        
        # 운동별 세부 평가
        if exercise in ['squat', 'front_squat']:
            # 무릎 정렬 체크
            left_knee_x = landmarks[25]['x']
            left_ankle_x = landmarks[27]['x']
            knee_alignment = abs(left_knee_x - left_ankle_x)
            
            if knee_alignment > 0.1:  # 무릎이 발끝을 너무 넘음
                score -= 10
            
            # 척추 정렬 체크
            nose_y = landmarks[0]['y']
            hip_y = (landmarks[23]['y'] + landmarks[24]['y']) / 2
            spine_angle = abs(nose_y - hip_y)
            
            if spine_angle < 0.3:  # 너무 앞으로 숙임
                score -= 15
        
        return max(0, score)
    
    def validate_pose_data(self, pose_data: List[Dict]) -> bool:
        """포즈 데이터 품질 검증"""
        if not pose_data or len(pose_data) < 30:  # 최소 1초 분량
            return False
        
        # 유효한 프레임 비율
        valid_frames = sum(1 for p in pose_data if p['landmarks'] and p['visibility_score'] > 0.5)
        valid_ratio = valid_frames / len(pose_data)
        
        return valid_ratio > 0.7  # 70% 이상 유효해야 함
    
    def calculate_quality_metrics(self, pose_data: List[Dict]) -> Dict:
        """데이터 품질 메트릭 계산"""
        metrics = {
            'avg_visibility': 0,
            'phase_distribution': {},
            'avg_quality_score': 0,
            'continuity_score': 0,
            'stability_score': 0
        }
        
        if not pose_data:
            return metrics
        
        # 평균 가시성
        visibility_scores = [p['visibility_score'] for p in pose_data if p['landmarks']]
        metrics['avg_visibility'] = np.mean(visibility_scores) if visibility_scores else 0
        
        # 운동 단계 분포
        phases = [p.get('phase', 'unknown') for p in pose_data]
        for phase in set(phases):
            metrics['phase_distribution'][phase] = phases.count(phase) / len(phases)
        
        # 평균 품질 점수
        quality_scores = [p.get('quality_score', 0) for p in pose_data if p.get('quality_score')]
        metrics['avg_quality_score'] = np.mean(quality_scores) if quality_scores else 0
        
        # 연속성 점수 (프레임 간 부드러운 전환)
        if len(pose_data) > 1:
            continuity_scores = []
            for i in range(1, len(pose_data)):
                if pose_data[i]['landmarks'] and pose_data[i-1]['landmarks']:
                    # 프레임 간 랜드마크 변화량
                    diff = 0
                    for j in range(min(len(pose_data[i]['landmarks']), len(pose_data[i-1]['landmarks']))):
                        diff += abs(pose_data[i]['landmarks'][j]['x'] - pose_data[i-1]['landmarks'][j]['x'])
                        diff += abs(pose_data[i]['landmarks'][j]['y'] - pose_data[i-1]['landmarks'][j]['y'])
                    continuity_scores.append(1.0 / (1.0 + diff))  # 변화가 적을수록 높은 점수
            
            metrics['continuity_score'] = np.mean(continuity_scores) if continuity_scores else 0
        
        return metrics
    
    async def collect_google_images(self, max_images: int = 100) -> List[Dict]:
        """구글 이미지 검색으로 보조 데이터 수집"""
        logger.info("Collecting supplementary data from Google Images...")
        
        image_data = []
        
        # 운동별 이미지 검색
        search_queries = [
            "olympic weightlifting squat technique sequence",
            "powerlifting bench press form analysis",
            "olympic deadlift technique breakdown",
            "crossfit athletes perfect form",
            "olympic lifting slow motion",
        ]
        
        for query in search_queries:
            images = await self.search_google_images(query, max_images // len(search_queries))
            image_data.extend(images)
        
        logger.info(f"Collected {len(image_data)} reference images")
        return image_data
    
    async def search_google_images(self, query: str, max_results: int = 20) -> List[Dict]:
        """구글 이미지 검색 (교육 목적)"""
        # 실제 구현시 Google Custom Search API 사용 권장
        # 여기서는 예시 구조만 제공
        
        images = []
        
        # Google Custom Search API 사용 예시
        # API_KEY = "your_api_key"
        # CX = "your_search_engine_id"
        # url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={CX}&key={API_KEY}&searchType=image"
        
        # 임시 데이터 구조
        for i in range(max_results):
            images.append({
                'url': f'placeholder_{i}',
                'title': f'{query} - Image {i}',
                'source': 'google_images',
                'query': query
            })
        
        return images
    
    def create_training_dataset(self) -> Dict:
        """수집된 데이터로 학습 데이터셋 생성"""
        logger.info("Creating training dataset from collected data...")
        
        dataset = {
            'olympic_tier': [],  # 올림픽 선수 데이터
            'professional_tier': [],  # 프로 선수 데이터
            'reference_tier': [],  # 참조 데이터
            'metadata': {
                'total_videos': len(self.collected_videos),
                'total_frames': 0,
                'athletes': list(set([v.get('athlete', 'Unknown') for v in self.collected_videos])),
                'exercises': list(set([v.get('exercise', 'Unknown') for v in self.collected_videos])),
                'creation_date': datetime.now().isoformat()
            }
        }
        
        # 포즈 데이터 파일 로드 및 분류
        pose_files = os.listdir(f'{self.output_dir}/poses')
        
        for pose_file in pose_files:
            with open(f'{self.output_dir}/poses/{pose_file}', 'r') as f:
                pose_data = json.load(f)
            
            # 비디오 정보 찾기
            video_id = pose_file.replace('.json', '')
            video_info = next((v for v in self.collected_videos if v['video_id'] == video_id), None)
            
            if video_info:
                # 품질에 따라 티어 분류
                if video_info.get('quality_tier') == 'olympic':
                    dataset['olympic_tier'].extend(pose_data[:1000])  # 최대 1000 프레임
                elif video_info.get('quality_tier') == 'official':
                    dataset['professional_tier'].extend(pose_data[:1000])
                else:
                    dataset['reference_tier'].extend(pose_data[:500])
                
                dataset['metadata']['total_frames'] += len(pose_data)
        
        # 학습 데이터셋 저장
        with open(f'{self.output_dir}/training_dataset.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"✅ Training dataset created with {dataset['metadata']['total_frames']} frames")
        
        # 통계 출력
        print("\n📊 데이터셋 통계:")
        print(f"- 올림픽 티어: {len(dataset['olympic_tier'])} 프레임")
        print(f"- 프로페셔널 티어: {len(dataset['professional_tier'])} 프레임")
        print(f"- 참조 티어: {len(dataset['reference_tier'])} 프레임")
        print(f"- 총 선수 수: {len(dataset['metadata']['athletes'])}")
        print(f"- 운동 종류: {dataset['metadata']['exercises']}")
        
        return dataset
    
    async def run_full_pipeline(self, max_videos: int = 50):
        """전체 데이터 수집 파이프라인 실행"""
        logger.info("Starting full Olympic data collection pipeline...")
        
        # 1. YouTube 영상 수집
        await self.collect_olympic_videos(max_videos_per_athlete=3)
        
        # 2. 영상 다운로드 및 처리
        processed_count = 0
        for video in tqdm(self.collected_videos[:max_videos], desc="Processing videos"):
            result = self.download_and_process_video(video, self.output_dir)
            if result:
                self.processed_data.append(result)
                processed_count += 1
        
        # 3. 구글 이미지 보조 데이터 수집
        await self.collect_google_images(max_images=100)
        
        # 4. 학습 데이터셋 생성
        dataset = self.create_training_dataset()
        
        # 5. 최종 보고서 생성
        self.generate_collection_report()
        
        logger.info(f"✅ Pipeline complete! Processed {processed_count}/{len(self.collected_videos)} videos")
        
        return dataset
    
    def generate_collection_report(self):
        """데이터 수집 보고서 생성"""
        report = {
            'collection_date': datetime.now().isoformat(),
            'total_videos_found': len(self.collected_videos),
            'total_videos_processed': len(self.processed_data),
            'success_rate': len(self.processed_data) / len(self.collected_videos) * 100 if self.collected_videos else 0,
            'top_athletes': [],
            'quality_distribution': {},
            'exercise_distribution': {},
            'total_storage_used_mb': 0
        }
        
        # 선수별 통계
        athlete_counts = {}
        for video in self.collected_videos:
            athlete = video.get('athlete', 'Unknown')
            athlete_counts[athlete] = athlete_counts.get(athlete, 0) + 1
        
        report['top_athletes'] = sorted(athlete_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 품질 분포
        for data in self.processed_data:
            quality_tier = 'high' if data['quality_metrics']['avg_quality_score'] > 80 else 'medium' if data['quality_metrics']['avg_quality_score'] > 60 else 'low'
            report['quality_distribution'][quality_tier] = report['quality_distribution'].get(quality_tier, 0) + 1
        
        # 운동 종류 분포
        for video in self.collected_videos:
            exercise = video.get('exercise', 'Unknown')
            report['exercise_distribution'][exercise] = report['exercise_distribution'].get(exercise, 0) + 1
        
        # 스토리지 사용량
        video_dir = f'{self.output_dir}/videos'
        if os.path.exists(video_dir):
            total_size = sum(os.path.getsize(os.path.join(video_dir, f)) for f in os.listdir(video_dir))
            report['total_storage_used_mb'] = total_size / 1024 / 1024
        
        # 보고서 저장
        with open(f'{self.output_dir}/collection_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # 마크다운 보고서
        md_report = f"""# 올림픽 데이터 수집 보고서

## 📊 수집 통계
- **수집 일시**: {report['collection_date']}
- **총 영상 발견**: {report['total_videos_found']}개
- **처리 완료**: {report['total_videos_processed']}개
- **성공률**: {report['success_rate']:.1f}%
- **사용 용량**: {report['total_storage_used_mb']:.1f} MB

## 🏆 Top 10 선수
"""
        for athlete, count in report['top_athletes']:
            md_report += f"- {athlete}: {count}개 영상\n"
        
        md_report += f"""
## 📈 품질 분포
- 높음: {report['quality_distribution'].get('high', 0)}개
- 중간: {report['quality_distribution'].get('medium', 0)}개
- 낮음: {report['quality_distribution'].get('low', 0)}개

## 🏋️ 운동 종류
"""
        for exercise, count in report['exercise_distribution'].items():
            md_report += f"- {exercise}: {count}개\n"
        
        with open(f'{self.output_dir}/collection_report.md', 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        logger.info(f"Report saved to {self.output_dir}/collection_report.md")


async def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect Olympic athlete training data')
    parser.add_argument('--output_dir', type=str, default='olympic_dataset',
                       help='Output directory for collected data')
    parser.add_argument('--max_videos', type=int, default=50,
                       help='Maximum number of videos to process')
    parser.add_argument('--athletes_only', action='store_true',
                       help='Collect only specific athlete videos')
    
    args = parser.parse_args()
    
    # 데이터 수집기 초기화
    collector = OlympicDataCollector(args.output_dir)
    
    # 전체 파이프라인 실행
    dataset = await collector.run_full_pipeline(max_videos=args.max_videos)
    
    print("\n✅ 올림픽 데이터 수집 완료!")
    print(f"📁 데이터 저장 위치: {args.output_dir}")
    print(f"📊 학습 데이터셋: {args.output_dir}/training_dataset.json")
    print(f"📝 수집 보고서: {args.output_dir}/collection_report.md")


if __name__ == "__main__":
    asyncio.run(main())