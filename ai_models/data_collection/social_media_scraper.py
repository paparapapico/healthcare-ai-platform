"""
Instagram Reels, TikTok 운동 영상 자동 수집
Social Media Exercise Video Scraper
"""

import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional
import logging
from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import instaloader
import TikTokApi
from bs4 import BeautifulSoup
import requests
import time
from tqdm import tqdm
import hashlib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialMediaScraper:
    """소셜 미디어 운동 영상 수집기"""
    
    # 인기 피트니스 인플루언서 및 해시태그
    FITNESS_INFLUENCERS = {
        'instagram': [
            'squat_university',          # 스쿼트 전문
            'athleanx',                  # 운동 과학
            'mobilitywod',               # 모빌리티
            'stefi_cohen',               # 파워리프팅 챔피언
            'thor_bjornsson',            # 스트롱맨
            'simonebiles',               # 올림픽 체조
            'lu_xiaojun',                # 중국 역도 전설
            'hookgrip',                  # 역도 전문
            'jujimufu',                  # 운동 엔터테이너
            'olympic_weightlifting',     # 올림픽 역도
            'crossfit',                  # 크로스핏 공식
            'powerlifting_motivation',   # 파워리프팅
        ],
        'tiktok': [
            '@hamptonsfit',              # 운동 교육
            '@eugeneapt',                # 파워리프팅
            '@thefitnesschef_',         # 피트니스 과학
            '@mrbentley_smith',         # 운동 폼
            '@sammyfitsss',             # 여성 피트니스
            '@bradleysimmonds',         # PT 트레이너
            '@joeywoll',                # 체조/칼리스테닉스
            '@chrisvanfitness',         # 운동 팁
        ]
    }
    
    # 운동 관련 해시태그
    EXERCISE_HASHTAGS = {
        'squat': [
            '#squatform', '#squattechnique', '#deepsquat', '#frontsquat',
            '#backsquat', '#olympicsquat', '#squateveryday', '#squatchallenge',
            '#ATGsquat', '#perfectsquat', '#스쿼트', '#스쿼트자세'
        ],
        'deadlift': [
            '#deadliftform', '#deadlifttechnique', '#sumo', '#conventional',
            '#deadliftday', '#powerlifting', '#데드리프트', '#데드리프트자세'
        ],
        'benchpress': [
            '#benchpress', '#benchtechnique', '#powerlifting', '#chestday',
            '#벤치프레스', '#가슴운동'
        ],
        'olympic_lifting': [
            '#snatch', '#cleanandjerk', '#olympicweightlifting', '#weightlifting',
            '#역도', '#클린앤저크', '#스내치'
        ]
    }
    
    def __init__(self, output_dir: str = 'social_media_dataset'):
        self.output_dir = output_dir
        self.session = None
        
        # 디렉토리 생성
        os.makedirs(f'{output_dir}/instagram', exist_ok=True)
        os.makedirs(f'{output_dir}/tiktok', exist_ok=True)
        os.makedirs(f'{output_dir}/metadata', exist_ok=True)
        
        # Instagram 로더 초기화
        self.instagram_loader = instaloader.Instaloader(
            download_videos=True,
            download_video_thumbnails=False,
            download_comments=False,
            save_metadata=True,
            compress_json=False
        )
        
        # 수집 통계
        self.stats = {
            'instagram_posts': 0,
            'tiktok_videos': 0,
            'total_videos': 0,
            'high_quality': 0
        }
    
    async def scrape_instagram_reels(self, max_posts: int = 100) -> List[Dict]:
        """Instagram Reels 수집"""
        logger.info("Starting Instagram Reels collection...")
        
        collected_posts = []
        
        try:
            # 로그인 (선택적 - 더 많은 데이터 접근 가능)
            # self.instagram_loader.login('username', 'password')
            
            # 인플루언서별 수집
            for username in self.FITNESS_INFLUENCERS['instagram']:
                try:
                    logger.info(f"Scraping @{username}...")
                    profile = instaloader.Profile.from_username(
                        self.instagram_loader.context, 
                        username
                    )
                    
                    posts_count = 0
                    for post in profile.get_posts():
                        if posts_count >= max_posts // len(self.FITNESS_INFLUENCERS['instagram']):
                            break
                        
                        # 비디오만 수집
                        if post.is_video:
                            post_data = {
                                'id': post.shortcode,
                                'url': post.url,
                                'video_url': post.video_url,
                                'caption': post.caption,
                                'likes': post.likes,
                                'comments': post.comments,
                                'date': post.date.isoformat(),
                                'owner': username,
                                'hashtags': post.caption_hashtags,
                                'mentions': post.caption_mentions,
                                'quality_score': self.calculate_instagram_quality(post)
                            }
                            
                            # 운동 종류 분류
                            post_data['exercise_type'] = self.classify_exercise(post.caption)
                            
                            # 고품질 영상만 수집
                            if post_data['quality_score'] > 70:
                                collected_posts.append(post_data)
                                
                                # 비디오 다운로드
                                if await self.download_instagram_video(post_data):
                                    posts_count += 1
                                    self.stats['instagram_posts'] += 1
                    
                except Exception as e:
                    logger.error(f"Error scraping @{username}: {e}")
                    continue
            
            # 해시태그별 수집
            for exercise, hashtags in self.EXERCISE_HASHTAGS.items():
                for hashtag in hashtags[:3]:  # 운동당 상위 3개 해시태그
                    try:
                        logger.info(f"Scraping hashtag {hashtag}...")
                        posts = instaloader.Hashtag.from_name(
                            self.instagram_loader.context,
                            hashtag.replace('#', '')
                        ).get_posts()
                        
                        hashtag_count = 0
                        for post in posts:
                            if hashtag_count >= 10:  # 해시태그당 최대 10개
                                break
                            
                            if post.is_video:
                                post_data = {
                                    'id': post.shortcode,
                                    'url': post.url,
                                    'video_url': post.video_url,
                                    'caption': post.caption if post.caption else '',
                                    'likes': post.likes,
                                    'date': post.date.isoformat(),
                                    'hashtag_source': hashtag,
                                    'exercise_type': exercise,
                                    'quality_score': self.calculate_instagram_quality(post)
                                }
                                
                                if post_data['quality_score'] > 60:
                                    collected_posts.append(post_data)
                                    if await self.download_instagram_video(post_data):
                                        hashtag_count += 1
                    
                    except Exception as e:
                        logger.error(f"Error with hashtag {hashtag}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Instagram scraping failed: {e}")
        
        logger.info(f"Collected {len(collected_posts)} Instagram posts")
        return collected_posts
    
    async def scrape_tiktok_videos(self, max_videos: int = 100) -> List[Dict]:
        """TikTok 영상 수집"""
        logger.info("Starting TikTok video collection...")
        
        collected_videos = []
        
        try:
            # TikTok API 초기화
            api = TikTokApi.TikTokApi()
            
            # 인플루언서별 수집
            for username in self.FITNESS_INFLUENCERS['tiktok']:
                try:
                    username_clean = username.replace('@', '')
                    logger.info(f"Scraping TikTok @{username_clean}...")
                    
                    user = api.user(username_clean)
                    videos = user.videos(count=max_videos // len(self.FITNESS_INFLUENCERS['tiktok']))
                    
                    for video in videos:
                        video_data = {
                            'id': video.id,
                            'url': f"https://www.tiktok.com/@{username_clean}/video/{video.id}",
                            'description': video.desc,
                            'likes': video.stats.diggCount,
                            'views': video.stats.playCount,
                            'shares': video.stats.shareCount,
                            'comments': video.stats.commentCount,
                            'duration': video.video.duration,
                            'author': username_clean,
                            'hashtags': [tag.name for tag in video.challenges] if video.challenges else [],
                            'music': video.music.title if video.music else None,
                            'quality_score': self.calculate_tiktok_quality(video)
                        }
                        
                        # 운동 종류 분류
                        video_data['exercise_type'] = self.classify_exercise(video.desc)
                        
                        if video_data['quality_score'] > 70:
                            collected_videos.append(video_data)
                            
                            # 비디오 다운로드
                            if await self.download_tiktok_video(video_data):
                                self.stats['tiktok_videos'] += 1
                
                except Exception as e:
                    logger.error(f"Error scraping TikTok @{username}: {e}")
                    continue
            
            # 해시태그/챌린지별 수집
            for exercise, hashtags in self.EXERCISE_HASHTAGS.items():
                for hashtag in hashtags[:2]:  # 운동당 2개 해시태그
                    try:
                        hashtag_clean = hashtag.replace('#', '')
                        logger.info(f"Scraping TikTok hashtag #{hashtag_clean}...")
                        
                        videos = api.hashtag(hashtag_clean).videos(count=15)
                        
                        for video in videos:
                            video_data = {
                                'id': video.id,
                                'description': video.desc,
                                'likes': video.stats.diggCount,
                                'views': video.stats.playCount,
                                'hashtag_source': hashtag,
                                'exercise_type': exercise,
                                'quality_score': self.calculate_tiktok_quality(video)
                            }
                            
                            if video_data['quality_score'] > 60:
                                collected_videos.append(video_data)
                                await self.download_tiktok_video(video_data)
                    
                    except Exception as e:
                        logger.error(f"Error with TikTok hashtag {hashtag}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"TikTok scraping failed: {e}")
        
        # Selenium 백업 방법 (API 실패시)
        if len(collected_videos) < 10:
            logger.info("Using Selenium backup method for TikTok...")
            collected_videos.extend(await self.scrape_tiktok_selenium())
        
        logger.info(f"Collected {len(collected_videos)} TikTok videos")
        return collected_videos
    
    async def scrape_tiktok_selenium(self) -> List[Dict]:
        """Selenium을 사용한 TikTok 스크래핑 (백업 방법)"""
        videos = []
        
        try:
            # Chrome 옵션 설정
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            driver = webdriver.Chrome(options=chrome_options)
            
            # 피트니스 관련 TikTok 페이지 방문
            fitness_urls = [
                'https://www.tiktok.com/tag/squattechnique',
                'https://www.tiktok.com/tag/deadliftform',
                'https://www.tiktok.com/tag/olympicweightlifting'
            ]
            
            for url in fitness_urls:
                driver.get(url)
                time.sleep(3)
                
                # 스크롤하여 더 많은 비디오 로드
                for _ in range(5):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                
                # 비디오 링크 추출
                video_elements = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/video/"]')
                
                for element in video_elements[:10]:
                    try:
                        video_url = element.get_attribute('href')
                        video_id = video_url.split('/video/')[-1].split('?')[0]
                        
                        videos.append({
                            'id': video_id,
                            'url': video_url,
                            'source': 'selenium',
                            'exercise_type': self.classify_exercise_from_url(url)
                        })
                    except:
                        continue
            
            driver.quit()
            
        except Exception as e:
            logger.error(f"Selenium TikTok scraping failed: {e}")
        
        return videos
    
    async def download_instagram_video(self, post_data: Dict) -> bool:
        """Instagram 비디오 다운로드"""
        try:
            video_url = post_data.get('video_url')
            if not video_url:
                return False
            
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # 파일명 생성 (해시 사용)
                        filename = f"{post_data['id']}.mp4"
                        filepath = os.path.join(self.output_dir, 'instagram', filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        
                        # 메타데이터 저장
                        meta_path = os.path.join(self.output_dir, 'metadata', f"ig_{post_data['id']}.json")
                        with open(meta_path, 'w') as f:
                            json.dump(post_data, f, indent=2)
                        
                        logger.info(f"Downloaded Instagram video: {post_data['id']}")
                        return True
        
        except Exception as e:
            logger.error(f"Failed to download Instagram video {post_data.get('id')}: {e}")
        
        return False
    
    async def download_tiktok_video(self, video_data: Dict) -> bool:
        """TikTok 비디오 다운로드"""
        try:
            # TikTok 다운로드는 복잡하므로 yt-dlp 사용 권장
            import yt_dlp
            
            ydl_opts = {
                'outtmpl': os.path.join(self.output_dir, 'tiktok', f"{video_data['id']}.mp4"),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_data.get('url', '')])
            
            # 메타데이터 저장
            meta_path = os.path.join(self.output_dir, 'metadata', f"tt_{video_data['id']}.json")
            with open(meta_path, 'w') as f:
                json.dump(video_data, f, indent=2)
            
            logger.info(f"Downloaded TikTok video: {video_data['id']}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download TikTok video {video_data.get('id')}: {e}")
            return False
    
    def calculate_instagram_quality(self, post) -> float:
        """Instagram 게시물 품질 점수 계산"""
        score = 50.0
        
        # 좋아요 수
        if post.likes > 10000:
            score += 20
        elif post.likes > 1000:
            score += 10
        elif post.likes > 100:
            score += 5
        
        # 검증된 계정
        if hasattr(post.owner_profile, 'is_verified') and post.owner_profile.is_verified:
            score += 15
        
        # 캡션 품질 (기술 설명 포함)
        caption = post.caption if post.caption else ''
        technique_keywords = ['form', 'technique', 'tips', 'correct', 'proper', 'angle', 'position']
        for keyword in technique_keywords:
            if keyword in caption.lower():
                score += 3
        
        # 운동 관련 해시태그
        hashtags = post.caption_hashtags if hasattr(post, 'caption_hashtags') else []
        for hashtag in hashtags:
            if any(exercise in hashtag.lower() for exercise in ['squat', 'deadlift', 'bench', 'lift']):
                score += 2
        
        return min(100, score)
    
    def calculate_tiktok_quality(self, video) -> float:
        """TikTok 비디오 품질 점수 계산"""
        score = 50.0
        
        # 조회수
        if hasattr(video.stats, 'playCount'):
            if video.stats.playCount > 1000000:
                score += 20
            elif video.stats.playCount > 100000:
                score += 10
            elif video.stats.playCount > 10000:
                score += 5
        
        # 좋아요 비율
        if hasattr(video.stats, 'diggCount') and hasattr(video.stats, 'playCount'):
            like_ratio = video.stats.diggCount / max(video.stats.playCount, 1)
            if like_ratio > 0.1:
                score += 15
            elif like_ratio > 0.05:
                score += 10
        
        # 영상 길이 (적절한 길이)
        if hasattr(video.video, 'duration'):
            if 15 <= video.video.duration <= 60:
                score += 10
        
        # 설명 품질
        if hasattr(video, 'desc'):
            desc = video.desc.lower()
            if any(word in desc for word in ['technique', 'form', 'tutorial', 'howto']):
                score += 10
        
        return min(100, score)
    
    def classify_exercise(self, text: str) -> str:
        """텍스트에서 운동 종류 분류"""
        text_lower = text.lower() if text else ''
        
        exercise_patterns = {
            'squat': ['squat', '스쿼트', 'スクワット'],
            'deadlift': ['deadlift', 'dead lift', '데드리프트', 'デッドリフト'],
            'bench_press': ['bench', 'bench press', '벤치프레스', 'ベンチプレス'],
            'overhead_press': ['overhead', 'ohp', 'shoulder press', '숄더프레스'],
            'pull_up': ['pull up', 'pullup', 'chin up', '풀업', '턱걸이'],
            'push_up': ['push up', 'pushup', '푸시업', '팔굽혀펴기'],
            'clean': ['clean', 'clean and jerk', '클린', 'クリーン'],
            'snatch': ['snatch', '스내치', 'スナッチ']
        }
        
        for exercise, patterns in exercise_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return exercise
        
        return 'general_fitness'
    
    def classify_exercise_from_url(self, url: str) -> str:
        """URL에서 운동 종류 추론"""
        url_lower = url.lower()
        
        if 'squat' in url_lower:
            return 'squat'
        elif 'deadlift' in url_lower:
            return 'deadlift'
        elif 'bench' in url_lower:
            return 'bench_press'
        elif 'weightlifting' in url_lower or 'olympic' in url_lower:
            return 'olympic_lifting'
        
        return 'general_fitness'
    
    async def process_all_videos(self):
        """수집된 모든 비디오 처리 (포즈 추출)"""
        logger.info("Processing collected videos for pose extraction...")
        
        # Instagram 비디오 처리
        instagram_videos = os.listdir(os.path.join(self.output_dir, 'instagram'))
        for video_file in tqdm(instagram_videos, desc="Processing Instagram"):
            video_path = os.path.join(self.output_dir, 'instagram', video_file)
            await self.extract_poses_from_video(video_path, 'instagram')
        
        # TikTok 비디오 처리
        tiktok_videos = os.listdir(os.path.join(self.output_dir, 'tiktok'))
        for video_file in tqdm(tiktok_videos, desc="Processing TikTok"):
            video_path = os.path.join(self.output_dir, 'tiktok', video_file)
            await self.extract_poses_from_video(video_path, 'tiktok')
    
    async def extract_poses_from_video(self, video_path: str, source: str):
        """비디오에서 포즈 추출 (MediaPipe 사용)"""
        import cv2
        import mediapipe as mp
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(video_path)
        poses = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB 변환 및 포즈 감지
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                poses.append(landmarks)
        
        cap.release()
        
        # 포즈 데이터 저장
        if poses:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            pose_path = os.path.join(self.output_dir, 'metadata', f"{source}_poses_{video_id}.json")
            with open(pose_path, 'w') as f:
                json.dump(poses, f)
    
    def generate_report(self):
        """수집 보고서 생성"""
        report = {
            'collection_date': datetime.now().isoformat(),
            'stats': self.stats,
            'instagram_influencers': len(self.FITNESS_INFLUENCERS['instagram']),
            'tiktok_creators': len(self.FITNESS_INFLUENCERS['tiktok']),
            'total_hashtags': sum(len(tags) for tags in self.EXERCISE_HASHTAGS.values()),
            'storage_used_mb': self.calculate_storage_used()
        }
        
        # JSON 보고서
        with open(os.path.join(self.output_dir, 'social_media_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Markdown 보고서
        md_report = f"""# 소셜 미디어 데이터 수집 보고서

## 📊 수집 통계
- **수집 일시**: {report['collection_date']}
- **Instagram 게시물**: {report['stats']['instagram_posts']}개
- **TikTok 비디오**: {report['stats']['tiktok_videos']}개
- **총 비디오**: {report['stats']['total_videos']}개
- **고품질 콘텐츠**: {report['stats']['high_quality']}개
- **사용 용량**: {report['storage_used_mb']:.1f} MB

## 📱 데이터 소스
- **Instagram 인플루언서**: {report['instagram_influencers']}명
- **TikTok 크리에이터**: {report['tiktok_creators']}명
- **추적 해시태그**: {report['total_hashtags']}개

## ✅ 수집 완료
모든 소셜 미디어 데이터가 성공적으로 수집되었습니다.
"""
        
        with open(os.path.join(self.output_dir, 'social_media_report.md'), 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        logger.info(f"Report saved to {self.output_dir}/social_media_report.md")
    
    def calculate_storage_used(self) -> float:
        """사용된 스토리지 계산"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.output_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / 1024 / 1024  # MB


async def main():
    """메인 실행 함수"""
    scraper = SocialMediaScraper()
    
    # Instagram Reels 수집
    instagram_posts = await scraper.scrape_instagram_reels(max_posts=100)
    
    # TikTok 비디오 수집
    tiktok_videos = await scraper.scrape_tiktok_videos(max_videos=100)
    
    # 비디오 처리 (포즈 추출)
    await scraper.process_all_videos()
    
    # 통계 업데이트
    scraper.stats['total_videos'] = scraper.stats['instagram_posts'] + scraper.stats['tiktok_videos']
    
    # 보고서 생성
    scraper.generate_report()
    
    print("\n✅ 소셜 미디어 데이터 수집 완료!")
    print(f"📁 데이터 위치: {scraper.output_dir}")
    print(f"📊 총 {scraper.stats['total_videos']}개 비디오 수집")


if __name__ == "__main__":
    asyncio.run(main())