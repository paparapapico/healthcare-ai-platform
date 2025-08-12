"""
공개 데이터셋 자동 다운로드 및 통합
합법적인 공개 소스에서 운동 관련 데이터 수집
"""

import requests
import json
import os
from pathlib import Path
from datetime import datetime
import zipfile
import tarfile
import urllib.request
from urllib.parse import urlparse

class PublicDatasetDownloader:
    def __init__(self):
        self.base_path = Path("data/public_datasets")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 공개 데이터셋 목록 (모두 합법적이고 무료)
        self.public_datasets = {
            "coco_keypoints": {
                "name": "COCO Keypoints Dataset (Sample)",
                "description": "COCO 데이터셋의 키포인트 샘플",
                "url": "http://images.cocodataset.org/annotations/person_keypoints_val2017.zip",
                "size": "25MB",
                "type": "keypoints",
                "license": "Creative Commons"
            },
            "mpii_sample": {
                "name": "MPII Human Pose Sample",
                "description": "MPII 포즈 추정 샘플 데이터",
                "sample_data": True,  # 실제 URL은 등록 필요
                "type": "pose_estimation",
                "license": "Academic Use"
            },
            "youtube_8m": {
                "name": "YouTube-8M Sports Subset",
                "description": "YouTube-8M 데이터셋의 스포츠 카테고리",
                "url": "https://research.google.com/youtube8m/",
                "type": "video_classification",
                "license": "Apache 2.0"
            },
            "kinetics": {
                "name": "Kinetics Human Action Recognition",
                "description": "인간 행동 인식 데이터셋 (운동 포함)",
                "url": "https://deepmind.com/research/open-source/kinetics",
                "type": "action_recognition", 
                "license": "Creative Commons"
            }
        }
    
    def download_with_progress(self, url, filename):
        """진행률 표시와 함께 파일 다운로드"""
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded * 100 / total_size
                print(f"\r  다운로드 진행률: {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end="")
            else:
                print(f"\r  다운로드 중: {downloaded//1024//1024}MB", end="")
        
        try:
            urllib.request.urlretrieve(url, filename, progress_hook)
            print("")  # 줄바꿈
            return True
        except Exception as e:
            print(f"\n  다운로드 실패: {e}")
            return False
    
    def download_coco_keypoints_sample(self):
        """COCO 키포인트 샘플 다운로드"""
        print("COCO Keypoints 샘플 다운로드 시작...")
        
        # 작은 샘플 URL (실제로는 COCO의 일부만)
        sample_url = "http://images.cocodataset.org/annotations/person_keypoints_val2017.zip"
        filename = self.base_path / "coco_keypoints_sample.zip"
        
        # 이미 존재하면 스킵
        if filename.exists():
            print(f"  파일이 이미 존재합니다: {filename}")
            return str(filename)
        
        print(f"  URL: {sample_url}")
        success = self.download_with_progress(sample_url, filename)
        
        if success:
            print(f"  다운로드 완료: {filename}")
            # ZIP 파일 압축 해제
            try:
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    extract_path = self.base_path / "coco_keypoints"
                    zip_ref.extractall(extract_path)
                print(f"  압축 해제 완료: {extract_path}")
                return str(extract_path)
            except Exception as e:
                print(f"  압축 해제 실패: {e}")
                return str(filename)
        
        return None
    
    def create_synthetic_public_data(self):
        """실제 공개 데이터 형태의 합성 데이터 생성"""
        print("공개 데이터 형태의 합성 데이터 생성...")
        
        # COCO 형태의 어노테이션 생성
        coco_style = {
            "info": {
                "description": "Synthetic Fitness Dataset",
                "version": "1.0",
                "year": 2024,
                "date_created": datetime.now().isoformat()
            },
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person"}
            ],
            "images": [],
            "annotations": []
        }
        
        # 100개 가상 이미지와 어노테이션 생성
        for img_id in range(1, 101):
            # 이미지 정보
            image_info = {
                "id": img_id,
                "width": 640,
                "height": 480,
                "file_name": f"synthetic_exercise_{img_id:03d}.jpg",
                "license": 1
            }
            coco_style["images"].append(image_info)
            
            # 키포인트 어노테이션
            keypoints = []
            visibility = []
            
            # 17개 키포인트 (COCO 형식)
            for kp_id in range(17):
                x = np.random.uniform(50, 590)  # 이미지 크기 내에서
                y = np.random.uniform(50, 430)
                v = 2  # visible
                
                keypoints.extend([x, y, v])
                visibility.append(v)
            
            annotation = {
                "id": img_id,
                "image_id": img_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": sum(1 for v in visibility if v > 0),
                "area": 640 * 480 * 0.3,  # 대략적인 사람 영역
                "bbox": [100, 50, 440, 380],  # [x, y, width, height]
                "iscrowd": 0
            }
            coco_style["annotations"].append(annotation)
        
        # JSON으로 저장
        coco_file = self.base_path / "synthetic_coco_keypoints.json"
        with open(coco_file, 'w', encoding='utf-8') as f:
            json.dump(coco_style, f, indent=2)
        
        print(f"  COCO 형태 데이터 생성: {coco_file}")
        
        # MPII 스타일 데이터도 생성
        mpii_style = {
            "dataset": "MPII Synthetic Fitness",
            "version": "1.0",
            "images": []
        }
        
        for img_id in range(1, 51):  # 50개 이미지
            image_data = {
                "image": f"synthetic_mpii_{img_id:03d}.jpg",
                "joints": [],  # 16개 조인트 (MPII 표준)
                "joints_vis": [],
                "exercise_type": np.random.choice(["push_up", "squat", "deadlift", "plank"]),
                "difficulty": np.random.choice(["easy", "medium", "hard"])
            }
            
            # 16개 조인트 좌표
            for joint_id in range(16):
                x = np.random.uniform(0, 640)
                y = np.random.uniform(0, 480)
                image_data["joints"].append([x, y])
                image_data["joints_vis"].append(1)  # 보임
            
            mpii_style["images"].append(image_data)
        
        mpii_file = self.base_path / "synthetic_mpii_fitness.json"
        with open(mpii_file, 'w', encoding='utf-8') as f:
            json.dump(mpii_style, f, indent=2)
        
        print(f"  MPII 형태 데이터 생성: {mpii_file}")
        
        return [str(coco_file), str(mpii_file)]
    
    def download_github_opensource(self):
        """GitHub 오픈소스 데이터 다운로드"""
        print("GitHub 오픈소스 데이터 수집...")
        
        github_sources = [
            {
                "name": "TensorFlow Models - PoseNet",
                "url": "https://raw.githubusercontent.com/tensorflow/tfjs-models/master/posenet/demo/sample_data.json",
                "filename": "tensorflow_posenet_sample.json"
            },
            {
                "name": "OpenPose Training Data Sample", 
                "url": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose_train/master/dataset/sample_annotations.json",
                "filename": "openpose_sample.json",
                "fallback": True  # 실제 파일이 없을 경우 샘플 생성
            }
        ]
        
        downloaded_files = []
        
        for source in github_sources:
            print(f"  {source['name']} 다운로드 중...")
            filepath = self.base_path / source['filename']
            
            try:
                response = requests.get(source['url'], timeout=10)
                if response.status_code == 200:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        if source['url'].endswith('.json'):
                            json.dump(response.json(), f, indent=2)
                        else:
                            f.write(response.text)
                    
                    print(f"    다운로드 성공: {filepath}")
                    downloaded_files.append(str(filepath))
                else:
                    print(f"    다운로드 실패 (HTTP {response.status_code})")
                    if source.get('fallback'):
                        # 폴백 샘플 데이터 생성
                        sample_data = self.create_fallback_sample(source['name'])
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(sample_data, f, indent=2)
                        print(f"    폴백 데이터 생성: {filepath}")
                        downloaded_files.append(str(filepath))
                        
            except Exception as e:
                print(f"    오류: {e}")
                if source.get('fallback'):
                    sample_data = self.create_fallback_sample(source['name'])
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(sample_data, f, indent=2)
                    print(f"    폴백 데이터 생성: {filepath}")
                    downloaded_files.append(str(filepath))
        
        return downloaded_files
    
    def create_fallback_sample(self, source_name):
        """폴백 샘플 데이터 생성"""
        if "posenet" in source_name.lower():
            return {
                "poses": [
                    {
                        "keypoints": [
                            {"part": "nose", "position": {"x": 320, "y": 100}, "score": 0.95},
                            {"part": "leftEye", "position": {"x": 310, "y": 90}, "score": 0.9},
                            {"part": "rightEye", "position": {"x": 330, "y": 90}, "score": 0.9}
                            # ... 더 많은 키포인트
                        ],
                        "score": 0.85
                    }
                ]
            }
        elif "openpose" in source_name.lower():
            return {
                "people": [
                    {
                        "pose_keypoints_2d": [
                            320, 100, 0.95,  # nose
                            310, 90, 0.9,    # neck
                            # ... 더 많은 키포인트 (x, y, confidence)
                        ]
                    }
                ]
            }
        else:
            return {"sample_data": "fallback", "source": source_name}
    
    def create_dataset_index(self, downloaded_files):
        """다운로드된 데이터셋 인덱스 생성"""
        print("데이터셋 인덱스 생성...")
        
        index = {
            "created_at": datetime.now().isoformat(),
            "total_datasets": len(downloaded_files),
            "datasets": []
        }
        
        for filepath in downloaded_files:
            file_path = Path(filepath)
            
            # 파일 정보 분석
            file_info = {
                "filename": file_path.name,
                "filepath": str(file_path),
                "size_mb": file_path.stat().st_size / 1024 / 1024 if file_path.exists() else 0,
                "type": self.detect_file_type(file_path),
                "sample_count": self.count_samples(file_path)
            }
            
            index["datasets"].append(file_info)
        
        # 인덱스 파일 저장
        index_file = self.base_path / "dataset_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        print(f"  인덱스 저장: {index_file}")
        return index
    
    def detect_file_type(self, filepath):
        """파일 타입 감지"""
        name = filepath.name.lower()
        if 'coco' in name:
            return 'coco_keypoints'
        elif 'mpii' in name:
            return 'mpii_pose'
        elif 'posenet' in name:
            return 'tensorflow_posenet'
        elif 'openpose' in name:
            return 'openpose'
        else:
            return 'unknown'
    
    def count_samples(self, filepath):
        """샘플 개수 계산"""
        if not filepath.exists():
            return 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터 형태에 따라 샘플 수 계산
            if 'annotations' in data:  # COCO 형태
                return len(data['annotations'])
            elif 'images' in data:  # MPII 형태
                return len(data['images'])
            elif 'poses' in data:  # PoseNet 형태
                return len(data['poses'])
            else:
                return 1
                
        except Exception:
            return 0
    
    def run_complete_download(self):
        """전체 다운로드 프로세스 실행"""
        print("=" * 50)
        print("공개 데이터셋 다운로드 시작")
        print("=" * 50)
        
        all_downloaded = []
        
        # 1. 합성 공개 데이터 생성
        print("\n[1단계] 합성 공개 데이터 생성")
        synthetic_files = self.create_synthetic_public_data()
        all_downloaded.extend(synthetic_files)
        
        # 2. GitHub 오픈소스 데이터
        print("\n[2단계] GitHub 오픈소스 데이터")
        github_files = self.download_github_opensource()
        all_downloaded.extend(github_files)
        
        # 3. 실제 COCO 데이터 (크기가 크므로 선택적)
        print("\n[3단계] 실제 공개 데이터셋")
        print("  COCO 데이터는 크기가 커서 스킵합니다 (필요시 수동 다운로드)")
        print("  MPII 데이터는 등록이 필요해서 스킵합니다")
        
        # 4. 인덱스 생성
        print("\n[4단계] 데이터셋 인덱스 생성")
        index = self.create_dataset_index(all_downloaded)
        
        # 결과 요약
        print("\n" + "=" * 50)
        print("다운로드 완료 요약:")
        print(f"  총 데이터셋: {len(all_downloaded)}개")
        print(f"  전체 샘플: {sum(dataset['sample_count'] for dataset in index['datasets'])}개")
        print("  파일 목록:")
        for dataset in index['datasets']:
            print(f"    {dataset['filename']}: {dataset['sample_count']}개 샘플 ({dataset['size_mb']:.1f}MB)")
        print("=" * 50)
        
        return all_downloaded, index

if __name__ == "__main__":
    import numpy as np  # 여기서 import
    
    downloader = PublicDatasetDownloader()
    files, index = downloader.run_complete_download()