"""
데이터 품질 검증 및 최종 정리
수집된 모든 데이터의 품질을 검증하고 학습용으로 정리
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

class DataQualityValidator:
    def __init__(self):
        self.data_dirs = {
            'virtual_athletes': Path('data/virtual_athletes'),
            'simple_generated': Path('data/simple_generated'), 
            'public_datasets': Path('data/public_datasets'),
            'enhanced_collection': Path('data/enhanced_collection')
        }
        
        self.output_dir = Path('data/validated_dataset')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality_report = {
            'validation_date': datetime.now().isoformat(),
            'total_samples': 0,
            'quality_distribution': {},
            'exercise_distribution': {},
            'skill_distribution': {},
            'data_sources': {},
            'issues_found': [],
            'recommendations': []
        }
    
    def validate_virtual_athletes_data(self):
        """가상 운동선수 데이터 검증"""
        print("가상 운동선수 데이터 검증 중...")
        
        data_dir = self.data_dirs['virtual_athletes']
        if not data_dir.exists():
            print("  가상 운동선수 데이터 없음")
            return []
        
        validated_data = []
        json_files = list(data_dir.glob("*.json"))
        
        for json_file in json_files:
            if json_file.name == 'metadata.json':
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # 데이터 품질 검사
                quality_score = self.assess_session_quality(session_data)
                
                if quality_score > 0.7:  # 70% 이상만 통과
                    validated_data.append({
                        'source': 'virtual_athletes',
                        'file': str(json_file),
                        'data': session_data,
                        'quality_score': quality_score,
                        'exercise_type': session_data.get('exercise_type'),
                        'athlete_level': session_data.get('athlete_level'),
                        'frame_count': len(session_data.get('frames', []))
                    })
                    
            except Exception as e:
                self.quality_report['issues_found'].append(f"가상 데이터 파일 오류: {json_file.name} - {e}")
        
        print(f"  검증 완료: {len(validated_data)}/{len(json_files)}개 파일 통과")
        return validated_data
    
    def validate_simple_generated_data(self):
        """간단 생성 데이터 검증"""
        print("간단 생성 데이터 검증 중...")
        
        data_dir = self.data_dirs['simple_generated']
        if not data_dir.exists():
            print("  간단 생성 데이터 없음")
            return []
        
        validated_data = []
        json_files = list(data_dir.glob("quick_generated_*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                
                sessions = dataset.get('sessions', [])
                
                for session in sessions:
                    quality_score = self.assess_session_quality(session)
                    
                    if quality_score > 0.6:  # 60% 이상 통과
                        validated_data.append({
                            'source': 'simple_generated',
                            'file': str(json_file),
                            'data': session,
                            'quality_score': quality_score,
                            'exercise_type': session.get('exercise_type'),
                            'skill_level': session.get('skill_level'),
                            'frame_count': len(session.get('frames', []))
                        })
                        
            except Exception as e:
                self.quality_report['issues_found'].append(f"간단 생성 데이터 오류: {json_file.name} - {e}")
        
        print(f"  검증 완료: {len(validated_data)}개 세션 통과")
        return validated_data
    
    def validate_public_datasets(self):
        """공개 데이터셋 검증"""
        print("공개 데이터셋 검증 중...")
        
        data_dir = self.data_dirs['public_datasets']
        if not data_dir.exists():
            print("  공개 데이터셋 없음")
            return []
        
        validated_data = []
        
        # COCO 형태 데이터 검증
        coco_file = data_dir / "synthetic_coco_keypoints.json"
        if coco_file.exists():
            try:
                with open(coco_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                
                annotations = coco_data.get('annotations', [])
                for ann in annotations[:50]:  # 처음 50개만
                    # COCO를 우리 형식으로 변환
                    converted_session = self.convert_coco_to_session(ann, coco_data)
                    quality_score = self.assess_session_quality(converted_session)
                    
                    if quality_score > 0.5:
                        validated_data.append({
                            'source': 'coco_synthetic',
                            'file': str(coco_file),
                            'data': converted_session,
                            'quality_score': quality_score,
                            'exercise_type': 'general_pose',
                            'skill_level': 'unknown',
                            'frame_count': 1
                        })
                        
            except Exception as e:
                self.quality_report['issues_found'].append(f"COCO 데이터 오류: {e}")
        
        # MPII 형태 데이터 검증
        mpii_file = data_dir / "synthetic_mpii_fitness.json"
        if mpii_file.exists():
            try:
                with open(mpii_file, 'r', encoding='utf-8') as f:
                    mpii_data = json.load(f)
                
                images = mpii_data.get('images', [])
                for img_data in images:
                    # MPII를 우리 형식으로 변환
                    converted_session = self.convert_mpii_to_session(img_data)
                    quality_score = self.assess_session_quality(converted_session)
                    
                    if quality_score > 0.5:
                        validated_data.append({
                            'source': 'mpii_synthetic',
                            'file': str(mpii_file),
                            'data': converted_session,
                            'quality_score': quality_score,
                            'exercise_type': img_data.get('exercise_type', 'unknown'),
                            'skill_level': img_data.get('difficulty', 'unknown'),
                            'frame_count': 1
                        })
                        
            except Exception as e:
                self.quality_report['issues_found'].append(f"MPII 데이터 오류: {e}")
        
        print(f"  검증 완료: {len(validated_data)}개 샘플 통과")
        return validated_data
    
    def assess_session_quality(self, session_data):
        """세션 데이터 품질 평가"""
        quality_score = 1.0
        
        # 필수 필드 확인
        required_fields = ['exercise_type']
        for field in required_fields:
            if field not in session_data:
                quality_score -= 0.2
        
        # 프레임 데이터 확인
        frames = session_data.get('frames', [])
        if not frames:
            return 0.0
        
        # 키포인트 품질 확인
        valid_frames = 0
        for frame in frames:
            keypoints = frame.get('keypoints', [])
            if self.validate_keypoints(keypoints):
                valid_frames += 1
        
        if len(frames) > 0:
            frame_quality = valid_frames / len(frames)
            quality_score *= frame_quality
        
        # 일관성 확인
        if len(frames) > 1:
            consistency_score = self.check_frame_consistency(frames)
            quality_score *= consistency_score
        
        return max(0, min(1, quality_score))
    
    def validate_keypoints(self, keypoints):
        """키포인트 유효성 검사"""
        if not keypoints or len(keypoints) < 10:
            return False
        
        valid_count = 0
        for kp in keypoints:
            if isinstance(kp, dict):
                x = kp.get('x', -1)
                y = kp.get('y', -1)
                visibility = kp.get('visibility', kp.get('confidence', 0))
                
                # 좌표와 가시성 검사
                if 0 <= x <= 1 and 0 <= y <= 1 and visibility > 0.5:
                    valid_count += 1
        
        return valid_count >= len(keypoints) * 0.7  # 70% 이상 유효
    
    def check_frame_consistency(self, frames):
        """프레임 간 일관성 검사"""
        if len(frames) < 2:
            return 1.0
        
        consistency_scores = []
        
        for i in range(1, min(len(frames), 10)):  # 최대 10개 프레임만 검사
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            # 키포인트 위치 변화 확인
            consistency = self.calculate_frame_similarity(
                prev_frame.get('keypoints', []),
                curr_frame.get('keypoints', [])
            )
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def calculate_frame_similarity(self, kp1, kp2):
        """두 프레임 간 유사도 계산"""
        if len(kp1) != len(kp2):
            return 0.5
        
        similarities = []
        
        for i in range(min(len(kp1), len(kp2))):
            if isinstance(kp1[i], dict) and isinstance(kp2[i], dict):
                x1, y1 = kp1[i].get('x', 0), kp1[i].get('y', 0)
                x2, y2 = kp2[i].get('x', 0), kp2[i].get('y', 0)
                
                # 유클리드 거리
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                # 거리가 0.1 이하면 유사한 것으로 간주
                similarity = max(0, 1 - distance/0.1)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def convert_coco_to_session(self, annotation, coco_data):
        """COCO 어노테이션을 우리 세션 형식으로 변환"""
        keypoints_raw = annotation.get('keypoints', [])
        
        # COCO 키포인트 (x, y, visibility) 형식을 우리 형식으로 변환
        keypoints = []
        for i in range(0, len(keypoints_raw), 3):
            if i+2 < len(keypoints_raw):
                x, y, v = keypoints_raw[i:i+3]
                # 이미지 크기로 정규화 (640x480 가정)
                keypoints.append({
                    'x': x / 640.0,
                    'y': y / 480.0,
                    'visibility': v / 2.0  # COCO는 0-2, 우리는 0-1
                })
        
        return {
            'exercise_type': 'general_pose',
            'skill_level': 'unknown',
            'frames': [{
                'rep_number': 1,
                'phase': 'static',
                'keypoints': keypoints,
                'form_score': 75.0,
                'timestamp': datetime.now().isoformat()
            }],
            'total_reps': 1,
            'timestamp': datetime.now().isoformat()
        }
    
    def convert_mpii_to_session(self, image_data):
        """MPII 이미지 데이터를 우리 세션 형식으로 변환"""
        joints = image_data.get('joints', [])
        joints_vis = image_data.get('joints_vis', [])
        
        keypoints = []
        for i, (joint, vis) in enumerate(zip(joints, joints_vis)):
            if len(joint) >= 2:
                x, y = joint[0], joint[1]
                keypoints.append({
                    'x': x / 640.0,  # 640x480 가정
                    'y': y / 480.0,
                    'visibility': float(vis) if vis else 0.0
                })
        
        return {
            'exercise_type': image_data.get('exercise_type', 'unknown'),
            'skill_level': image_data.get('difficulty', 'unknown'),
            'frames': [{
                'rep_number': 1,
                'phase': 'static',
                'keypoints': keypoints,
                'form_score': 70.0,
                'timestamp': datetime.now().isoformat()
            }],
            'total_reps': 1,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_quality_statistics(self, all_validated_data):
        """품질 통계 생성"""
        print("품질 통계 생성 중...")
        
        # 기본 통계
        self.quality_report['total_samples'] = len(all_validated_data)
        
        # 품질 점수 분포
        quality_scores = [item['quality_score'] for item in all_validated_data]
        self.quality_report['quality_distribution'] = {
            'mean': float(np.mean(quality_scores)),
            'std': float(np.std(quality_scores)),
            'min': float(np.min(quality_scores)),
            'max': float(np.max(quality_scores)),
            'median': float(np.median(quality_scores))
        }
        
        # 운동 타입별 분포
        exercise_counts = {}
        skill_counts = {}
        source_counts = {}
        
        for item in all_validated_data:
            # 운동 타입
            exercise = item.get('exercise_type', 'unknown')
            exercise_counts[exercise] = exercise_counts.get(exercise, 0) + 1
            
            # 실력 레벨
            skill = item.get('skill_level', 'unknown')
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            # 소스별
            source = item.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        self.quality_report['exercise_distribution'] = exercise_counts
        self.quality_report['skill_distribution'] = skill_counts
        self.quality_report['data_sources'] = source_counts
        
        # 권장사항 생성
        self.generate_recommendations()
        
    def generate_recommendations(self):
        """데이터 개선 권장사항 생성"""
        recommendations = []
        
        # 품질 점수 기반 권장사항
        avg_quality = self.quality_report['quality_distribution']['mean']
        if avg_quality < 0.8:
            recommendations.append(f"평균 품질 점수가 {avg_quality:.2f}로 낮습니다. 데이터 정제가 필요합니다.")
        
        # 분포 불균형 확인
        exercise_dist = self.quality_report['exercise_distribution']
        if len(exercise_dist) > 0:
            max_count = max(exercise_dist.values())
            min_count = min(exercise_dist.values())
            if max_count > min_count * 3:
                recommendations.append("운동 타입별 데이터 불균형이 심합니다. 부족한 운동의 데이터를 더 수집해야 합니다.")
        
        # 실력 레벨 분포 확인
        skill_dist = self.quality_report['skill_distribution']
        if 'advanced' in skill_dist and 'beginner' in skill_dist:
            if skill_dist.get('advanced', 0) < skill_dist.get('beginner', 0) * 0.3:
                recommendations.append("고급 실력 데이터가 부족합니다. 전문가 데이터 수집을 늘려야 합니다.")
        
        # 총 샘플 수 확인
        total_samples = self.quality_report['total_samples']
        if total_samples < 1000:
            recommendations.append(f"총 샘플 수({total_samples})가 적습니다. 최소 5000개 이상 권장됩니다.")
        
        self.quality_report['recommendations'] = recommendations
    
    def create_unified_dataset(self, all_validated_data):
        """통합 데이터셋 생성"""
        print("통합 데이터셋 생성 중...")
        
        # 학습용/검증용/테스트용 분할
        np.random.seed(42)  # 재현성을 위해
        np.random.shuffle(all_validated_data)
        
        total_size = len(all_validated_data)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.2)
        
        train_data = all_validated_data[:train_size]
        val_data = all_validated_data[train_size:train_size + val_size]
        test_data = all_validated_data[train_size + val_size:]
        
        # 각 세트별로 저장
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        saved_files = {}
        
        for split_name, split_data in splits.items():
            filename = self.output_dir / f"{split_name}_dataset.json"
            
            # 메타데이터와 함께 저장
            dataset = {
                'metadata': {
                    'split': split_name,
                    'total_samples': len(split_data),
                    'created_at': datetime.now().isoformat(),
                    'quality_threshold': 0.5,
                    'sources': list(set(item['source'] for item in split_data))
                },
                'samples': [item['data'] for item in split_data]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            saved_files[split_name] = str(filename)
            print(f"  {split_name} 데이터셋 저장: {filename} ({len(split_data)}개 샘플)")
        
        return saved_files
    
    def save_quality_report(self):
        """품질 보고서 저장"""
        report_file = self.output_dir / "quality_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.quality_report, f, indent=2, ensure_ascii=False)
        
        print(f"품질 보고서 저장: {report_file}")
        return str(report_file)
    
    def run_complete_validation(self):
        """전체 검증 프로세스 실행"""
        print("=" * 60)
        print("데이터 품질 검증 및 정리 시작")
        print("=" * 60)
        
        all_validated_data = []
        
        # 1. 각 소스별 데이터 검증
        print("\\n[1단계] 소스별 데이터 검증")
        
        # 가상 운동선수 데이터
        virtual_data = self.validate_virtual_athletes_data()
        all_validated_data.extend(virtual_data)
        
        # 간단 생성 데이터  
        simple_data = self.validate_simple_generated_data()
        all_validated_data.extend(simple_data)
        
        # 공개 데이터셋
        public_data = self.validate_public_datasets()
        all_validated_data.extend(public_data)
        
        # 2. 통계 생성
        print("\\n[2단계] 품질 통계 생성")
        self.generate_quality_statistics(all_validated_data)
        
        # 3. 통합 데이터셋 생성
        print("\\n[3단계] 통합 데이터셋 생성")
        dataset_files = self.create_unified_dataset(all_validated_data)
        
        # 4. 보고서 저장
        print("\\n[4단계] 품질 보고서 저장")
        report_file = self.save_quality_report()
        
        # 5. 최종 결과 출력
        print("\\n" + "=" * 60)
        print("검증 완료 결과:")
        print(f"  총 검증된 샘플: {len(all_validated_data)}개")
        print(f"  평균 품질 점수: {self.quality_report['quality_distribution']['mean']:.3f}")
        print("  데이터셋 분할:")
        for split, filepath in dataset_files.items():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"    {split}: {data['metadata']['total_samples']}개")
        print("  운동별 분포:")
        for exercise, count in self.quality_report['exercise_distribution'].items():
            print(f"    {exercise}: {count}개")
        print(f"  품질 보고서: {report_file}")
        
        if self.quality_report['recommendations']:
            print("  개선 권장사항:")
            for rec in self.quality_report['recommendations']:
                print(f"    - {rec}")
        
        print("=" * 60)
        
        return {
            'validated_samples': len(all_validated_data),
            'dataset_files': dataset_files,
            'quality_report': self.quality_report,
            'report_file': report_file
        }

if __name__ == "__main__":
    validator = DataQualityValidator()
    results = validator.run_complete_validation()