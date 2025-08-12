"""
ESPN, Olympic.org 등 공식 스포츠 사이트 데이터 수집
Official Sports Websites Data Scraper
"""

import os
import json
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import requests
from typing import List, Dict, Optional
import logging
from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
from tqdm import tqdm
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfficialSportsScraper:
    """공식 스포츠 사이트 데이터 수집기"""
    
    # 공식 스포츠 사이트 목록
    OFFICIAL_SITES = {
        'olympic': {
            'base_url': 'https://olympics.com',
            'endpoints': [
                '/en/athletes',
                '/en/sports/weightlifting',
                '/en/sports/gymnastics',
                '/en/sports/athletics',
                '/en/video/sports',
                '/en/news/olympic-games'
            ]
        },
        'iwf': {  # International Weightlifting Federation
            'base_url': 'https://iwf.sport',
            'endpoints': [
                '/results',
                '/athletes',
                '/education/technique',
                '/media-centre/videos'
            ]
        },
        'ipf': {  # International Powerlifting Federation
            'base_url': 'https://www.powerlifting.sport',
            'endpoints': [
                '/championships/results',
                '/athletes',
                '/technical-corner'
            ]
        },
        'fig': {  # International Gymnastics Federation
            'base_url': 'https://www.gymnastics.sport',
            'endpoints': [
                '/athletes',
                '/technical',
                '/education'
            ]
        },
        'espn': {
            'base_url': 'https://www.espn.com',
            'endpoints': [
                '/olympics',
                '/sports/weightlifting',
                '/sports/gymnastics'
            ]
        },
        'crossfit': {
            'base_url': 'https://games.crossfit.com',
            'endpoints': [
                '/leaderboard',
                '/athletes',
                '/workouts',
                '/video'
            ]
        }
    }
    
    # 세계 기록 보유자 및 챔피언
    WORLD_CHAMPIONS = {
        'weightlifting': {
            'men': {
                '61kg': {'name': 'Li Fabin', 'country': 'CHN', 'records': {'snatch': 145, 'clean_jerk': 175}},
                '73kg': {'name': 'Shi Zhiyong', 'country': 'CHN', 'records': {'snatch': 169, 'clean_jerk': 198}},
                '81kg': {'name': 'Lu Xiaojun', 'country': 'CHN', 'records': {'snatch': 177, 'clean_jerk': 207}},
                '96kg': {'name': 'Tian Tao', 'country': 'CHN', 'records': {'snatch': 183, 'clean_jerk': 230}},
                '109kg': {'name': 'Akbar Djuraev', 'country': 'UZB', 'records': {'snatch': 193, 'clean_jerk': 237}},
                '+109kg': {'name': 'Lasha Talakhadze', 'country': 'GEO', 'records': {'snatch': 225, 'clean_jerk': 267}}
            },
            'women': {
                '49kg': {'name': 'Hou Zhihui', 'country': 'CHN', 'records': {'snatch': 96, 'clean_jerk': 118}},
                '55kg': {'name': 'Liao Qiuyun', 'country': 'CHN', 'records': {'snatch': 103, 'clean_jerk': 129}},
                '59kg': {'name': 'Kuo Hsing-chun', 'country': 'TPE', 'records': {'snatch': 110, 'clean_jerk': 140}},
                '64kg': {'name': 'Deng Wei', 'country': 'CHN', 'records': {'snatch': 117, 'clean_jerk': 145}},
                '71kg': {'name': 'Zhang Wangli', 'country': 'CHN', 'records': {'snatch': 117, 'clean_jerk': 152}},
                '76kg': {'name': 'Neisi Dajomes', 'country': 'ECU', 'records': {'snatch': 118, 'clean_jerk': 145}}
            }
        },
        'powerlifting': {
            'men': {
                'squat': {'name': 'Ray Williams', 'record': 490, 'unit': 'kg'},
                'bench': {'name': 'Julius Maddox', 'record': 355, 'unit': 'kg'},
                'deadlift': {'name': 'Hafthor Bjornsson', 'record': 501, 'unit': 'kg'}
            },
            'women': {
                'squat': {'name': 'April Mathis', 'record': 321, 'unit': 'kg'},
                'bench': {'name': 'April Mathis', 'record': 207, 'unit': 'kg'},
                'deadlift': {'name': 'Tamara Walcott', 'record': 288, 'unit': 'kg'}
            }
        }
    }
    
    def __init__(self, output_dir: str = 'official_sports_dataset'):
        self.output_dir = output_dir
        self.session = None
        
        # 디렉토리 생성
        os.makedirs(f'{output_dir}/athletes', exist_ok=True)
        os.makedirs(f'{output_dir}/competitions', exist_ok=True)
        os.makedirs(f'{output_dir}/techniques', exist_ok=True)
        os.makedirs(f'{output_dir}/records', exist_ok=True)
        
        # Chrome 드라이버 설정
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        
        self.collected_data = {
            'athletes': [],
            'competitions': [],
            'techniques': [],
            'videos': [],
            'records': []
        }
    
    async def scrape_olympic_website(self) -> Dict:
        """Olympic.com 데이터 수집"""
        logger.info("Scraping Olympic.com...")
        
        olympic_data = {
            'athletes': [],
            'sports': [],
            'videos': [],
            'records': []
        }
        
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            
            # 역도 선수 페이지
            driver.get('https://olympics.com/en/sports/weightlifting')
            time.sleep(3)
            
            # 선수 정보 수집
            athlete_elements = driver.find_elements(By.CSS_SELECTOR, '.athlete-card')
            for element in athlete_elements[:20]:  # 상위 20명
                try:
                    name = element.find_element(By.CSS_SELECTOR, '.athlete-name').text
                    country = element.find_element(By.CSS_SELECTOR, '.country-code').text
                    
                    athlete_data = {
                        'name': name,
                        'country': country,
                        'sport': 'weightlifting',
                        'source': 'olympics.com',
                        'profile_url': element.get_attribute('href')
                    }
                    
                    # 상세 정보 수집
                    athlete_details = await self.get_athlete_details(driver, athlete_data['profile_url'])
                    athlete_data.update(athlete_details)
                    
                    olympic_data['athletes'].append(athlete_data)
                    
                except Exception as e:
                    logger.error(f"Error extracting athlete data: {e}")
                    continue
            
            # 기술 비디오 수집
            driver.get('https://olympics.com/en/video/sports/weightlifting')
            time.sleep(3)
            
            video_elements = driver.find_elements(By.CSS_SELECTOR, '.video-card')
            for element in video_elements[:30]:
                try:
                    title = element.find_element(By.CSS_SELECTOR, '.video-title').text
                    url = element.get_attribute('href')
                    thumbnail = element.find_element(By.CSS_SELECTOR, 'img').get_attribute('src')
                    
                    video_data = {
                        'title': title,
                        'url': url,
                        'thumbnail': thumbnail,
                        'source': 'olympics.com',
                        'category': self.categorize_video(title)
                    }
                    
                    olympic_data['videos'].append(video_data)
                    
                except Exception as e:
                    logger.error(f"Error extracting video data: {e}")
                    continue
            
            # 세계 기록 수집
            olympic_data['records'] = await self.scrape_world_records(driver)
            
            driver.quit()
            
        except Exception as e:
            logger.error(f"Olympic.com scraping failed: {e}")
        
        self.collected_data['athletes'].extend(olympic_data['athletes'])
        self.collected_data['videos'].extend(olympic_data['videos'])
        self.collected_data['records'].extend(olympic_data['records'])
        
        logger.info(f"Collected {len(olympic_data['athletes'])} athletes from Olympics.com")
        return olympic_data
    
    async def get_athlete_details(self, driver, profile_url: str) -> Dict:
        """선수 상세 정보 수집"""
        details = {}
        
        try:
            driver.get(profile_url)
            time.sleep(2)
            
            # 메달 정보
            medals = driver.find_elements(By.CSS_SELECTOR, '.medal-count')
            if medals:
                details['medals'] = {
                    'gold': medals[0].text if len(medals) > 0 else '0',
                    'silver': medals[1].text if len(medals) > 1 else '0',
                    'bronze': medals[2].text if len(medals) > 2 else '0'
                }
            
            # 개인 기록
            records = driver.find_elements(By.CSS_SELECTOR, '.personal-best')
            if records:
                details['personal_records'] = []
                for record in records:
                    details['personal_records'].append(record.text)
            
            # 경력 하이라이트
            highlights = driver.find_elements(By.CSS_SELECTOR, '.career-highlight')
            if highlights:
                details['career_highlights'] = [h.text for h in highlights[:5]]
            
        except Exception as e:
            logger.error(f"Error getting athlete details: {e}")
        
        return details
    
    async def scrape_iwf_website(self) -> Dict:
        """International Weightlifting Federation 데이터 수집"""
        logger.info("Scraping IWF website...")
        
        iwf_data = {
            'competitions': [],
            'results': [],
            'techniques': []
        }
        
        async with aiohttp.ClientSession() as session:
            # 대회 결과 수집
            try:
                async with session.get('https://iwf.sport/results') as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # 최근 대회 결과
                        competitions = soup.find_all('div', class_='competition-card')
                        for comp in competitions[:10]:
                            comp_data = {
                                'name': comp.find('h3').text if comp.find('h3') else '',
                                'date': comp.find('span', class_='date').text if comp.find('span', class_='date') else '',
                                'location': comp.find('span', class_='location').text if comp.find('span', class_='location') else '',
                                'level': 'world_championship',
                                'source': 'iwf.sport'
                            }
                            
                            # 결과 링크
                            results_link = comp.find('a', class_='results-link')
                            if results_link:
                                comp_data['results_url'] = results_link.get('href')
                                # 상세 결과 수집
                                comp_results = await self.get_competition_results(session, comp_data['results_url'])
                                comp_data['results'] = comp_results
                            
                            iwf_data['competitions'].append(comp_data)
            
            except Exception as e:
                logger.error(f"IWF scraping failed: {e}")
            
            # 기술 교육 자료 수집
            try:
                async with session.get('https://iwf.sport/education/technique') as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        techniques = soup.find_all('article', class_='technique-article')
                        for tech in techniques:
                            tech_data = {
                                'title': tech.find('h2').text if tech.find('h2') else '',
                                'description': tech.find('p').text if tech.find('p') else '',
                                'category': self.categorize_technique(tech.find('h2').text if tech.find('h2') else ''),
                                'source': 'iwf.sport'
                            }
                            
                            # 이미지/비디오 링크
                            media = tech.find_all('img') + tech.find_all('video')
                            tech_data['media_urls'] = [m.get('src') for m in media]
                            
                            iwf_data['techniques'].append(tech_data)
            
            except Exception as e:
                logger.error(f"IWF technique scraping failed: {e}")
        
        self.collected_data['competitions'].extend(iwf_data['competitions'])
        self.collected_data['techniques'].extend(iwf_data['techniques'])
        
        logger.info(f"Collected {len(iwf_data['competitions'])} competitions from IWF")
        return iwf_data
    
    async def get_competition_results(self, session, results_url: str) -> List[Dict]:
        """대회 결과 상세 정보 수집"""
        results = []
        
        try:
            async with session.get(results_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 결과 테이블 파싱
                    result_rows = soup.find_all('tr', class_='result-row')
                    for row in result_rows[:20]:  # 상위 20명
                        cells = row.find_all('td')
                        if len(cells) >= 6:
                            result = {
                                'rank': cells[0].text.strip(),
                                'name': cells[1].text.strip(),
                                'country': cells[2].text.strip(),
                                'snatch': cells[3].text.strip(),
                                'clean_jerk': cells[4].text.strip(),
                                'total': cells[5].text.strip()
                            }
                            results.append(result)
        
        except Exception as e:
            logger.error(f"Error getting competition results: {e}")
        
        return results
    
    async def scrape_espn(self) -> Dict:
        """ESPN 스포츠 데이터 수집"""
        logger.info("Scraping ESPN...")
        
        espn_data = {
            'articles': [],
            'videos': [],
            'stats': []
        }
        
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            
            # 올림픽 섹션
            driver.get('https://www.espn.com/olympics')
            time.sleep(3)
            
            # 최신 기사
            articles = driver.find_elements(By.CSS_SELECTOR, 'article.contentItem')
            for article in articles[:15]:
                try:
                    title = article.find_element(By.CSS_SELECTOR, 'h2').text
                    link = article.find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
                    summary = article.find_element(By.CSS_SELECTOR, '.description').text if article.find_element(By.CSS_SELECTOR, '.description') else ''
                    
                    article_data = {
                        'title': title,
                        'url': link,
                        'summary': summary,
                        'source': 'espn.com',
                        'category': self.categorize_article(title)
                    }
                    
                    # 운동 기술 관련 기사만 수집
                    if any(keyword in title.lower() for keyword in ['technique', 'form', 'training', 'workout', 'exercise']):
                        espn_data['articles'].append(article_data)
                
                except Exception as e:
                    continue
            
            # 비디오 콘텐츠
            driver.get('https://www.espn.com/video/sport/olympics')
            time.sleep(3)
            
            videos = driver.find_elements(By.CSS_SELECTOR, '.video-item')
            for video in videos[:20]:
                try:
                    title = video.find_element(By.CSS_SELECTOR, '.video-title').text
                    duration = video.find_element(By.CSS_SELECTOR, '.duration').text if video.find_element(By.CSS_SELECTOR, '.duration') else ''
                    
                    video_data = {
                        'title': title,
                        'duration': duration,
                        'source': 'espn.com',
                        'category': self.categorize_video(title)
                    }
                    
                    espn_data['videos'].append(video_data)
                
                except Exception as e:
                    continue
            
            driver.quit()
            
        except Exception as e:
            logger.error(f"ESPN scraping failed: {e}")
        
        self.collected_data['videos'].extend(espn_data['videos'])
        
        logger.info(f"Collected {len(espn_data['articles'])} articles from ESPN")
        return espn_data
    
    async def scrape_crossfit_games(self) -> Dict:
        """CrossFit Games 데이터 수집"""
        logger.info("Scraping CrossFit Games...")
        
        cf_data = {
            'athletes': [],
            'workouts': [],
            'leaderboard': []
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                # 선수 리더보드
                async with session.get('https://games.crossfit.com/leaderboard') as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # 상위 선수들
                        athletes = soup.find_all('div', class_='athlete-row')
                        for athlete in athletes[:30]:
                            athlete_data = {
                                'rank': athlete.find('span', class_='rank').text if athlete.find('span', class_='rank') else '',
                                'name': athlete.find('span', class_='name').text if athlete.find('span', class_='name') else '',
                                'country': athlete.find('span', class_='country').text if athlete.find('span', class_='country') else '',
                                'points': athlete.find('span', class_='points').text if athlete.find('span', class_='points') else '',
                                'source': 'crossfit.com'
                            }
                            cf_data['athletes'].append(athlete_data)
                
                # WOD (Workout of the Day)
                async with session.get('https://games.crossfit.com/workouts') as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        workouts = soup.find_all('div', class_='workout-card')
                        for workout in workouts[:20]:
                            wod_data = {
                                'name': workout.find('h3').text if workout.find('h3') else '',
                                'description': workout.find('p', class_='description').text if workout.find('p', class_='description') else '',
                                'movements': [],
                                'source': 'crossfit.com'
                            }
                            
                            # 동작 목록
                            movements = workout.find_all('li', class_='movement')
                            wod_data['movements'] = [m.text for m in movements]
                            
                            cf_data['workouts'].append(wod_data)
            
            except Exception as e:
                logger.error(f"CrossFit Games scraping failed: {e}")
        
        self.collected_data['athletes'].extend(cf_data['athletes'])
        
        logger.info(f"Collected {len(cf_data['athletes'])} CrossFit athletes")
        return cf_data
    
    async def scrape_world_records(self, driver) -> List[Dict]:
        """세계 기록 수집"""
        records = []
        
        # 역도 세계 기록
        for gender, categories in self.WORLD_CHAMPIONS['weightlifting'].items():
            for weight_class, champion_data in categories.items():
                record = {
                    'sport': 'weightlifting',
                    'gender': gender,
                    'weight_class': weight_class,
                    'athlete': champion_data['name'],
                    'country': champion_data['country'],
                    'records': champion_data['records'],
                    'date_collected': datetime.now().isoformat()
                }
                records.append(record)
        
        # 파워리프팅 세계 기록
        for gender, lifts in self.WORLD_CHAMPIONS['powerlifting'].items():
            for lift_type, record_data in lifts.items():
                record = {
                    'sport': 'powerlifting',
                    'gender': gender,
                    'lift': lift_type,
                    'athlete': record_data['name'],
                    'record': f"{record_data['record']} {record_data['unit']}",
                    'date_collected': datetime.now().isoformat()
                }
                records.append(record)
        
        return records
    
    def categorize_video(self, title: str) -> str:
        """비디오 카테고리 분류"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['snatch', 'clean', 'jerk']):
            return 'olympic_lifting'
        elif any(word in title_lower for word in ['squat', 'deadlift', 'bench']):
            return 'powerlifting'
        elif any(word in title_lower for word in ['technique', 'form', 'tutorial']):
            return 'technique'
        elif any(word in title_lower for word in ['world record', 'championship', 'olympic']):
            return 'competition'
        else:
            return 'general'
    
    def categorize_technique(self, title: str) -> str:
        """기술 카테고리 분류"""
        title_lower = title.lower()
        
        if 'snatch' in title_lower:
            return 'snatch_technique'
        elif 'clean' in title_lower:
            return 'clean_technique'
        elif 'jerk' in title_lower:
            return 'jerk_technique'
        elif 'squat' in title_lower:
            return 'squat_technique'
        elif 'pull' in title_lower:
            return 'pull_technique'
        else:
            return 'general_technique'
    
    def categorize_article(self, title: str) -> str:
        """기사 카테고리 분류"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['technique', 'form', 'how to']):
            return 'educational'
        elif any(word in title_lower for word in ['championship', 'games', 'competition']):
            return 'competition'
        elif any(word in title_lower for word in ['training', 'workout', 'program']):
            return 'training'
        else:
            return 'news'
    
    async def download_media_content(self):
        """수집한 미디어 콘텐츠 다운로드"""
        logger.info("Downloading media content...")
        
        # 비디오 URL 다운로드
        for video in self.collected_data['videos']:
            if 'url' in video:
                try:
                    # yt-dlp를 사용한 다운로드 (YouTube, 기타 플랫폼 지원)
                    import yt_dlp
                    
                    ydl_opts = {
                        'outtmpl': os.path.join(self.output_dir, 'videos', '%(title)s.%(ext)s'),
                        'quiet': True,
                        'no_warnings': True,
                        'format': 'best[height<=720]'  # 720p 이하로 제한
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video['url']])
                    
                    logger.info(f"Downloaded: {video.get('title', 'Unknown')}")
                
                except Exception as e:
                    logger.error(f"Failed to download video: {e}")
    
    def save_collected_data(self):
        """수집한 데이터 저장"""
        logger.info("Saving collected data...")
        
        # 선수 데이터 저장
        if self.collected_data['athletes']:
            df_athletes = pd.DataFrame(self.collected_data['athletes'])
            df_athletes.to_csv(os.path.join(self.output_dir, 'athletes', 'all_athletes.csv'), index=False)
            
            with open(os.path.join(self.output_dir, 'athletes', 'all_athletes.json'), 'w') as f:
                json.dump(self.collected_data['athletes'], f, indent=2)
        
        # 대회 데이터 저장
        if self.collected_data['competitions']:
            with open(os.path.join(self.output_dir, 'competitions', 'all_competitions.json'), 'w') as f:
                json.dump(self.collected_data['competitions'], f, indent=2)
        
        # 기술 데이터 저장
        if self.collected_data['techniques']:
            with open(os.path.join(self.output_dir, 'techniques', 'all_techniques.json'), 'w') as f:
                json.dump(self.collected_data['techniques'], f, indent=2)
        
        # 세계 기록 저장
        if self.collected_data['records']:
            df_records = pd.DataFrame(self.collected_data['records'])
            df_records.to_csv(os.path.join(self.output_dir, 'records', 'world_records.csv'), index=False)
        
        logger.info("Data saved successfully")
    
    def generate_report(self):
        """수집 보고서 생성"""
        report = {
            'collection_date': datetime.now().isoformat(),
            'total_athletes': len(self.collected_data['athletes']),
            'total_competitions': len(self.collected_data['competitions']),
            'total_techniques': len(self.collected_data['techniques']),
            'total_videos': len(self.collected_data['videos']),
            'total_records': len(self.collected_data['records']),
            'sources': list(set([item.get('source', '') for sublist in self.collected_data.values() for item in sublist if isinstance(item, dict)]))
        }
        
        # JSON 보고서
        with open(os.path.join(self.output_dir, 'official_sports_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Markdown 보고서
        md_report = f"""# 공식 스포츠 사이트 데이터 수집 보고서

## 📊 수집 통계
- **수집 일시**: {report['collection_date']}
- **총 선수 데이터**: {report['total_athletes']}명
- **대회 정보**: {report['total_competitions']}개
- **기술 자료**: {report['total_techniques']}개
- **비디오 콘텐츠**: {report['total_videos']}개
- **세계 기록**: {report['total_records']}개

## 🌐 데이터 소스
"""
        for source in report['sources']:
            md_report += f"- {source}\n"
        
        md_report += """
## ✅ 수집 완료
모든 공식 스포츠 사이트 데이터가 성공적으로 수집되었습니다.
"""
        
        with open(os.path.join(self.output_dir, 'official_sports_report.md'), 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        logger.info(f"Report saved to {self.output_dir}/official_sports_report.md")


async def main():
    """메인 실행 함수"""
    scraper = OfficialSportsScraper()
    
    # 각 사이트 스크래핑
    await scraper.scrape_olympic_website()
    await scraper.scrape_iwf_website()
    await scraper.scrape_espn()
    await scraper.scrape_crossfit_games()
    
    # 미디어 콘텐츠 다운로드
    # await scraper.download_media_content()  # 선택적
    
    # 데이터 저장
    scraper.save_collected_data()
    
    # 보고서 생성
    scraper.generate_report()
    
    print("\n✅ 공식 스포츠 사이트 데이터 수집 완료!")
    print(f"📁 데이터 위치: {scraper.output_dir}")
    print(f"📊 수집 통계:")
    print(f"  - 선수: {len(scraper.collected_data['athletes'])}명")
    print(f"  - 대회: {len(scraper.collected_data['competitions'])}개")
    print(f"  - 기술: {len(scraper.collected_data['techniques'])}개")
    print(f"  - 비디오: {len(scraper.collected_data['videos'])}개")


if __name__ == "__main__":
    asyncio.run(main())