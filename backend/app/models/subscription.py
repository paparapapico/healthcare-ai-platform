# backend/app/models/subscription.py
from sqlalchemy import Column, String, DateTime, Boolean, Float, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from .base import Base

class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)  # Basic, Premium, Pro
    stripe_price_id = Column(String(100), unique=True, nullable=False)
    price = Column(Float, nullable=False)  # 월 구독료
    currency = Column(String(3), default="USD")
    interval = Column(String(20), default="month")  # month, year
    features = Column(Text)  # JSON string of features
    max_exercises_per_day = Column(Integer, default=10)
    max_ai_analysis = Column(Integer, default=5)
    premium_content_access = Column(Boolean, default=False)
    priority_support = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    subscriptions = relationship("UserSubscription", back_populates="plan")

class UserSubscription(Base):
    __tablename__ = "user_subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    plan_id = Column(UUID(as_uuid=True), ForeignKey("subscription_plans.id"), nullable=False)
    stripe_subscription_id = Column(String(100), unique=True)
    stripe_customer_id = Column(String(100))
    status = Column(String(20), default="active")  # active, canceled, past_due, unpaid
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    cancel_at_period_end = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="subscription")
    plan = relationship("SubscriptionPlan", back_populates="subscriptions")
    payments = relationship("Payment", back_populates="subscription")

class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("user_subscriptions.id"))
    stripe_payment_intent_id = Column(String(100), unique=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    status = Column(String(20))  # succeeded, pending, failed, canceled
    payment_method = Column(String(50))  # card, bank_transfer, etc.
    description = Column(String(255))
    metadata = Column(Text)  # JSON for additional data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    subscription = relationship("UserSubscription", back_populates="payments")