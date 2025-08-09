"""
Social Features API
친구, 챌린지, 리더보드 기능
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import random

from app.db.database import get_db
from app.models.models import (
    User, Friendship, Challenge, ChallengeParticipant,
    Workout, Achievement, UserAchievement
)

router = APIRouter(prefix="/api/v1/social", tags=["social"])

# ========================
# Schemas
# ========================

class FriendRequest(BaseModel):
    friend_email: str

class FriendshipResponse(BaseModel):
    id: int
    friend_id: int
    friend_name: str
    friend_email: str
    friend_health_score: float
    status: str
    created_at: datetime

class ChallengeCreate(BaseModel):
    title: str
    description: str
    exercise_type: str
    challenge_type: str = "individual"  # individual, team
    target_reps: Optional[int] = None
    target_duration: Optional[int] = None  # minutes
    target_calories: Optional[float] = None
    duration_days: int = 7
    is_public: bool = True

class ChallengeResponse(BaseModel):
    id: int
    title: str
    description: str
    exercise_type: str
    creator_name: str
    participant_count: int
    start_date: datetime
    end_date: datetime
    target_reps: Optional[int]
    target_duration: Optional[int]
    target_calories: Optional[float]
    is_active: bool
    my_progress: Optional[dict] = None

class LeaderboardEntry(BaseModel):
    rank: int
    user_id: int
    user_name: str
    score: float
    workout_count: int
    total_calories: float
    avg_form_score: float
    achievement_count: int

class ActivityFeedItem(BaseModel):
    id: int
    user_name: str
    user_id: int
    activity_type: str  # workout, achievement, challenge
    activity_data: dict
    timestamp: datetime

# ========================
# Friend Management
# ========================

@router.post("/friends/request")
async def send_friend_request(
    request: FriendRequest,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """친구 요청 보내기"""
    
    # Find friend by email
    friend = db.query(User).filter(User.email == request.friend_email).first()
    if not friend:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    
    if friend.id == current_user_id:
        raise HTTPException(status_code=400, detail="자기 자신에게 친구 요청을 보낼 수 없습니다")
    
    # Check if friendship already exists
    existing = db.query(Friendship).filter(
        ((Friendship.user_id == current_user_id) & (Friendship.friend_id == friend.id)) |
        ((Friendship.user_id == friend.id) & (Friendship.friend_id == current_user_id))
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="이미 친구 요청을 보냈거나 친구입니다")
    
    # Create friendship request
    friendship = Friendship(
        user_id=current_user_id,
        friend_id=friend.id,
        status="pending"
    )
    db.add(friendship)
    db.commit()
    
    return {"message": "친구 요청을 보냈습니다", "friend_name": friend.name}

@router.post("/friends/accept/{friendship_id}")
async def accept_friend_request(
    friendship_id: int,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """친구 요청 수락"""
    
    friendship = db.query(Friendship).filter(
        Friendship.id == friendship_id,
        Friendship.friend_id == current_user_id,
        Friendship.status == "pending"
    ).first()
    
    if not friendship:
        raise HTTPException(status_code=404, detail="친구 요청을 찾을 수 없습니다")
    
    friendship.status = "accepted"
    friendship.accepted_at = datetime.utcnow()
    
    # Create reverse friendship for bidirectional relationship
    reverse_friendship = Friendship(
        user_id=friendship.friend_id,
        friend_id=friendship.user_id,
        status="accepted",
        accepted_at=datetime.utcnow()
    )
    db.add(reverse_friendship)
    
    # Award achievement for first friend
    check_and_award_achievement(db, current_user_id, "첫 친구")
    
    db.commit()
    
    return {"message": "친구 요청을 수락했습니다"}

@router.get("/friends", response_model=List[FriendshipResponse])
async def get_friends(
    status: Optional[str] = Query(None, regex="^(pending|accepted|blocked)$"),
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """친구 목록 조회"""
    
    query = db.query(Friendship, User).join(
        User, User.id == Friendship.friend_id
    ).filter(
        Friendship.user_id == current_user_id
    )
    
    if status:
        query = query.filter(Friendship.status == status)
    
    friendships = query.all()
    
    return [
        FriendshipResponse(
            id=f.id,
            friend_id=u.id,
            friend_name=u.name,
            friend_email=u.email,
            friend_health_score=u.health_score,
            status=f.status,
            created_at=f.created_at
        )
        for f, u in friendships
    ]

@router.delete("/friends/{friend_id}")
async def remove_friend(
    friend_id: int,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """친구 삭제"""
    
    # Delete both directions of friendship
    db.query(Friendship).filter(
        ((Friendship.user_id == current_user_id) & (Friendship.friend_id == friend_id)) |
        ((Friendship.user_id == friend_id) & (Friendship.friend_id == current_user_id))
    ).delete()
    
    db.commit()
    
    return {"message": "친구를 삭제했습니다"}

# ========================
# Challenges
# ========================

@router.post("/challenges", response_model=ChallengeResponse)
async def create_challenge(
    challenge: ChallengeCreate,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """챌린지 생성"""
    
    user = db.query(User).filter(User.id == current_user_id).first()
    
    new_challenge = Challenge(
        creator_id=current_user_id,
        title=challenge.title,
        description=challenge.description,
        challenge_type=challenge.challenge_type,
        exercise_type=challenge.exercise_type,
        target_reps=challenge.target_reps,
        target_duration=challenge.target_duration,
        target_calories=challenge.target_calories,
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + timedelta(days=challenge.duration_days),
        is_public=challenge.is_public
    )
    
    db.add(new_challenge)
    db.flush()
    
    # Creator automatically joins the challenge
    participant = ChallengeParticipant(
        challenge_id=new_challenge.id,
        user_id=current_user_id
    )
    db.add(participant)
    db.commit()
    
    return ChallengeResponse(
        id=new_challenge.id,
        title=new_challenge.title,
        description=new_challenge.description,
        exercise_type=new_challenge.exercise_type,
        creator_name=user.name,
        participant_count=1,
        start_date=new_challenge.start_date,
        end_date=new_challenge.end_date,
        target_reps=new_challenge.target_reps,
        target_duration=new_challenge.target_duration,
        target_calories=new_challenge.target_calories,
        is_active=new_challenge.is_active
    )

@router.post("/challenges/{challenge_id}/join")
async def join_challenge(
    challenge_id: int,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """챌린지 참가"""
    
    challenge = db.query(Challenge).filter(
        Challenge.id == challenge_id,
        Challenge.is_active == True
    ).first()
    
    if not challenge:
        raise HTTPException(status_code=404, detail="챌린지를 찾을 수 없습니다")
    
    # Check if already joined
    existing = db.query(ChallengeParticipant).filter(
        ChallengeParticipant.challenge_id == challenge_id,
        ChallengeParticipant.user_id == current_user_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="이미 참가한 챌린지입니다")
    
    participant = ChallengeParticipant(
        challenge_id=challenge_id,
        user_id=current_user_id
    )
    db.add(participant)
    db.commit()
    
    return {"message": "챌린지에 참가했습니다"}

@router.get("/challenges", response_model=List[ChallengeResponse])
async def get_challenges(
    filter_type: str = Query("all", regex="^(all|my|active|completed)$"),
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """챌린지 목록 조회"""
    
    query = db.query(Challenge)
    
    if filter_type == "my":
        # My challenges
        query = query.join(ChallengeParticipant).filter(
            ChallengeParticipant.user_id == current_user_id
        )
    elif filter_type == "active":
        query = query.filter(
            Challenge.is_active == True,
            Challenge.end_date > datetime.utcnow()
        )
    elif filter_type == "completed":
        query = query.filter(Challenge.end_date <= datetime.utcnow())
    
    challenges = query.all()
    
    response = []
    for challenge in challenges:
        # Get participant count
        participant_count = db.query(ChallengeParticipant).filter(
            ChallengeParticipant.challenge_id == challenge.id
        ).count()
        
        # Get creator name
        creator = db.query(User).filter(User.id == challenge.creator_id).first()
        
        # Get current user's progress if participating
        my_progress = None
        participant = db.query(ChallengeParticipant).filter(
            ChallengeParticipant.challenge_id == challenge.id,
            ChallengeParticipant.user_id == current_user_id
        ).first()
        
        if participant:
            my_progress = {
                "current_reps": participant.current_reps,
                "current_duration": participant.current_duration,
                "current_calories": participant.current_calories,
                "rank": participant.rank
            }
        
        response.append(ChallengeResponse(
            id=challenge.id,
            title=challenge.title,
            description=challenge.description,
            exercise_type=challenge.exercise_type,
            creator_name=creator.name if creator else "Unknown",
            participant_count=participant_count,
            start_date=challenge.start_date,
            end_date=challenge.end_date,
            target_reps=challenge.target_reps,
            target_duration=challenge.target_duration,
            target_calories=challenge.target_calories,
            is_active=challenge.is_active,
            my_progress=my_progress
        ))
    
    return response

@router.get("/challenges/{challenge_id}/leaderboard")
async def get_challenge_leaderboard(
    challenge_id: int,
    db: Session = Depends(get_db)
):
    """챌린지 리더보드"""
    
    participants = db.query(
        ChallengeParticipant, User
    ).join(
        User, User.id == ChallengeParticipant.user_id
    ).filter(
        ChallengeParticipant.challenge_id == challenge_id
    ).order_by(
        ChallengeParticipant.current_reps.desc(),
        ChallengeParticipant.current_calories.desc()
    ).all()
    
    leaderboard = []
    for rank, (participant, user) in enumerate(participants, 1):
        leaderboard.append({
            "rank": rank,
            "user_id": user.id,
            "user_name": user.name,
            "current_reps": participant.current_reps,
            "current_duration": participant.current_duration,
            "current_calories": participant.current_calories,
            "progress_percentage": calculate_progress(participant, challenge_id, db)
        })
    
    return leaderboard

# ========================
# Global Leaderboard
# ========================

@router.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_global_leaderboard(
    period: str = Query("week", regex="^(day|week|month|all)$"),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """글로벌 리더보드"""
    
    # Calculate date range
    if period == "day":
        start_date = datetime.utcnow() - timedelta(days=1)
    elif period == "week":
        start_date = datetime.utcnow() - timedelta(days=7)
    elif period == "month":
        start_date = datetime.utcnow() - timedelta(days=30)
    else:
        start_date = datetime.min
    
    # Get user stats
    users = db.query(User).filter(User.is_active == True).all()
    
    leaderboard_data = []
    for user in users:
        # Get workout stats for period
        workouts = db.query(Workout).filter(
            Workout.user_id == user.id,
            Workout.start_time >= start_date
        ).all()
        
        if not workouts:
            continue
        
        total_calories = sum(w.calories_burned or 0 for w in workouts)
        avg_form_score = sum(w.avg_form_score or 0 for w in workouts) / len(workouts) if workouts else 0
        
        # Get achievement count
        achievement_count = db.query(UserAchievement).filter(
            UserAchievement.user_id == user.id
        ).count()
        
        # Calculate composite score
        score = (
            len(workouts) * 10 +  # 10 points per workout
            total_calories * 0.1 +  # 0.1 points per calorie
            avg_form_score * 2 +  # 2x form score
            achievement_count * 5  # 5 points per achievement
        )
        
        leaderboard_data.append({
            "user_id": user.id,
            "user_name": user.name,
            "score": score,
            "workout_count": len(workouts),
            "total_calories": total_calories,
            "avg_form_score": avg_form_score,
            "achievement_count": achievement_count
        })
    
    # Sort by score
    leaderboard_data.sort(key=lambda x: x["score"], reverse=True)
    
    # Add ranks and limit
    leaderboard = []
    for rank, data in enumerate(leaderboard_data[:limit], 1):
        leaderboard.append(LeaderboardEntry(
            rank=rank,
            **data
        ))
    
    return leaderboard

# ========================
# Activity Feed
# ========================

@router.get("/activity-feed", response_model=List[ActivityFeedItem])
async def get_activity_feed(
    current_user_id: int = 1,  # TODO: Get from auth
    limit: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """친구들의 활동 피드"""
    
    # Get friend IDs
    friendships = db.query(Friendship).filter(
        Friendship.user_id == current_user_id,
        Friendship.status == "accepted"
    ).all()
    
    friend_ids = [f.friend_id for f in friendships]
    friend_ids.append(current_user_id)  # Include self
    
    activities = []
    
    # Recent workouts
    recent_workouts = db.query(Workout, User).join(
        User, User.id == Workout.user_id
    ).filter(
        Workout.user_id.in_(friend_ids),
        Workout.start_time >= datetime.utcnow() - timedelta(days=7)
    ).order_by(Workout.start_time.desc()).limit(limit).all()
    
    for workout, user in recent_workouts:
        activities.append(ActivityFeedItem(
            id=workout.id,
            user_name=user.name,
            user_id=user.id,
            activity_type="workout",
            activity_data={
                "exercise_type": workout.exercise_type,
                "duration": workout.duration,
                "reps": workout.total_reps,
                "calories": workout.calories_burned,
                "form_score": workout.avg_form_score
            },
            timestamp=workout.start_time
        ))
    
    # Recent achievements
    recent_achievements = db.query(
        UserAchievement, User, Achievement
    ).join(
        User, User.id == UserAchievement.user_id
    ).join(
        Achievement, Achievement.id == UserAchievement.achievement_id
    ).filter(
        UserAchievement.user_id.in_(friend_ids),
        UserAchievement.earned_at >= datetime.utcnow() - timedelta(days=7)
    ).order_by(UserAchievement.earned_at.desc()).limit(limit).all()
    
    for user_achievement, user, achievement in recent_achievements:
        activities.append(ActivityFeedItem(
            id=user_achievement.id,
            user_name=user.name,
            user_id=user.id,
            activity_type="achievement",
            activity_data={
                "achievement_name": achievement.name,
                "achievement_description": achievement.description,
                "achievement_icon": achievement.icon,
                "points": achievement.points
            },
            timestamp=user_achievement.earned_at
        ))
    
    # Sort by timestamp
    activities.sort(key=lambda x: x.timestamp, reverse=True)
    
    return activities[:limit]

# ========================
# Helper Functions
# ========================

def check_and_award_achievement(db: Session, user_id: int, achievement_name: str):
    """업적 달성 확인 및 수여"""
    
    achievement = db.query(Achievement).filter(
        Achievement.name == achievement_name
    ).first()
    
    if not achievement:
        return
    
    # Check if already earned
    existing = db.query(UserAchievement).filter(
        UserAchievement.user_id == user_id,
        UserAchievement.achievement_id == achievement.id
    ).first()
    
    if not existing:
        user_achievement = UserAchievement(
            user_id=user_id,
            achievement_id=achievement.id,
            progress=100.0
        )
        db.add(user_achievement)

def calculate_progress(participant: ChallengeParticipant, challenge_id: int, db: Session) -> float:
    """챌린지 진행률 계산"""
    
    challenge = db.query(Challenge).filter(Challenge.id == challenge_id).first()
    if not challenge:
        return 0.0
    
    progress_values = []
    
    if challenge.target_reps and challenge.target_reps > 0:
        progress_values.append((participant.current_reps / challenge.target_reps) * 100)
    
    if challenge.target_duration and challenge.target_duration > 0:
        progress_values.append((participant.current_duration / challenge.target_duration) * 100)
    
    if challenge.target_calories and challenge.target_calories > 0:
        progress_values.append((participant.current_calories / challenge.target_calories) * 100)
    
    if progress_values:
        return min(100.0, sum(progress_values) / len(progress_values))
    
    return 0.0