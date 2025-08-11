# backend/app/models/user.py에 추가
# User 모델에 subscription relationship 추가
from sqlalchemy.orm import relationship

# User 클래스에 추가할 내용:
subscription = relationship("UserSubscription", back_populates="user", uselist=False)