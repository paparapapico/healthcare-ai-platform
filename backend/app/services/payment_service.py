# backend/app/services/payment_service.py
import stripe
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from ..core.config import settings
from ..models.subscription import UserSubscription, SubscriptionPlan, Payment
from ..models.user import User
from datetime import datetime, timedelta
import json

# Stripe 설정
stripe.api_key = settings.STRIPE_SECRET_KEY

class PaymentService:
    
    @staticmethod
    async def create_stripe_customer(user: User) -> str:
        """Stripe 고객 생성"""
        try:
            customer = stripe.Customer.create(
                email=user.email,
                name=user.name,
                metadata={
                    'user_id': str(user.id)
                }
            )
            return customer.id
        except stripe.error.StripeError as e:
            raise Exception(f"Stripe customer creation failed: {str(e)}")
    
    @staticmethod
    async def create_subscription_plans(db: Session):
        """기본 구독 플랜 생성"""
        plans_data = [
            {
                "name": "Basic",
                "price": 9.99,
                "features": json.dumps([
                    "10 exercises per day",
                    "Basic AI analysis",
                    "Standard support"
                ]),
                "max_exercises_per_day": 10,
                "max_ai_analysis": 5,
                "premium_content_access": False,
                "priority_support": False
            },
            {
                "name": "Premium", 
                "price": 19.99,
                "features": json.dumps([
                    "50 exercises per day",
                    "Advanced AI analysis",
                    "Premium content access",
                    "Priority support"
                ]),
                "max_exercises_per_day": 50,
                "max_ai_analysis": 20,
                "premium_content_access": True,
                "priority_support": True
            },
            {
                "name": "Pro",
                "price": 39.99,
                "features": json.dumps([
                    "Unlimited exercises",
                    "Full AI analysis",
                    "All premium content",
                    "24/7 priority support",
                    "Custom training plans"
                ]),
                "max_exercises_per_day": -1,  # -1 = unlimited
                "max_ai_analysis": -1,
                "premium_content_access": True,
                "priority_support": True
            }
        ]
        
        for plan_data in plans_data:
            # Stripe에서 가격 생성
            stripe_price = stripe.Price.create(
                currency="usd",
                unit_amount=int(plan_data["price"] * 100),  # cents
                recurring={"interval": "month"},
                product_data={
                    "name": f"HealthcareAI {plan_data['name']} Plan"
                }
            )
            
            # DB에 플랜 저장
            plan = SubscriptionPlan(
                name=plan_data["name"],
                stripe_price_id=stripe_price.id,
                price=plan_data["price"],
                features=plan_data["features"],
                max_exercises_per_day=plan_data["max_exercises_per_day"],
                max_ai_analysis=plan_data["max_ai_analysis"],
                premium_content_access=plan_data["premium_content_access"],
                priority_support=plan_data["priority_support"]
            )
            
            db.add(plan)
        
        db.commit()
    
    @staticmethod
    async def create_checkout_session(
        user: User, 
        plan_id: str, 
        db: Session
    ) -> Dict[str, Any]:
        """Stripe Checkout 세션 생성"""
        try:
            # 플랜 조회
            plan = db.query(SubscriptionPlan).filter(
                SubscriptionPlan.id == plan_id
            ).first()
            
            if not plan:
                raise Exception("Subscription plan not found")
            
            # Stripe 고객이 없으면 생성
            stripe_customer_id = await PaymentService.create_stripe_customer(user)
            
            # Checkout 세션 생성
            session = stripe.checkout.Session.create(
                customer=stripe_customer_id,
                payment_method_types=['card'],
                line_items=[{
                    'price': plan.stripe_price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=settings.STRIPE_SUCCESS_URL + f"?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=settings.STRIPE_CANCEL_URL,
                metadata={
                    'user_id': str(user.id),
                    'plan_id': str(plan.id)
                }
            )
            
            return {
                'checkout_url': session.url,
                'session_id': session.id
            }
            
        except stripe.error.StripeError as e:
            raise Exception(f"Stripe checkout creation failed: {str(e)}")
    
    @staticmethod
    async def handle_successful_payment(
        session_id: str,
        db: Session
    ) -> UserSubscription:
        """결제 성공 처리"""
        try:
            # Stripe 세션 조회
            session = stripe.checkout.Session.retrieve(session_id)
            subscription = stripe.Subscription.retrieve(session.subscription)
            
            user_id = session.metadata.get('user_id')
            plan_id = session.metadata.get('plan_id')
            
            # 기존 구독 취소
            existing_subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id,
                UserSubscription.status == "active"
            ).first()
            
            if existing_subscription:
                existing_subscription.status = "canceled"
                existing_subscription.cancel_at_period_end = True
            
            # 새 구독 생성
            new_subscription = UserSubscription(
                user_id=user_id,
                plan_id=plan_id,
                stripe_subscription_id=subscription.id,
                stripe_customer_id=subscription.customer,
                status=subscription.status,
                current_period_start=datetime.fromtimestamp(subscription.current_period_start),
                current_period_end=datetime.fromtimestamp(subscription.current_period_end)
            )
            
            db.add(new_subscription)
            
            # 결제 기록 생성
            payment = Payment(
                user_id=user_id,
                subscription_id=new_subscription.id,
                stripe_payment_intent_id=session.payment_intent,
                amount=subscription.items.data[0].price.unit_amount / 100,
                status="succeeded",
                payment_method="card",
                description=f"Subscription payment for {subscription.items.data[0].price.nickname}"
            )
            
            db.add(payment)
            db.commit()
            
            return new_subscription
            
        except stripe.error.StripeError as e:
            db.rollback()
            raise Exception(f"Payment processing failed: {str(e)}")
    
    @staticmethod
    async def cancel_subscription(
        user_id: str,
        db: Session,
        immediate: bool = False
    ) -> UserSubscription:
        """구독 취소"""
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id,
                UserSubscription.status == "active"
            ).first()
            
            if not subscription:
                raise Exception("Active subscription not found")
            
            # Stripe에서 구독 취소
            if immediate:
                stripe.Subscription.delete(subscription.stripe_subscription_id)
                subscription.status = "canceled"
            else:
                stripe.Subscription.modify(
                    subscription.stripe_subscription_id,
                    cancel_at_period_end=True
                )
                subscription.cancel_at_period_end = True
            
            subscription.updated_at = datetime.utcnow()
            db.commit()
            
            return subscription
            
        except stripe.error.StripeError as e:
            db.rollback()
            raise Exception(f"Subscription cancellation failed: {str(e)}")
    
    @staticmethod
    def check_user_permissions(user: User, feature: str) -> bool:
        """사용자 권한 확인"""
        if not user.subscription or user.subscription.status != "active":
            # 무료 사용자 제한
            return feature in ["basic_exercises", "limited_ai_analysis"]
        
        plan = user.subscription.plan
        
        feature_permissions = {
            "premium_content": plan.premium_content_access,
            "priority_support": plan.priority_support,
            "unlimited_exercises": plan.max_exercises_per_day == -1,
            "advanced_ai": plan.max_ai_analysis > 10
        }
        
        return feature_permissions.get(feature, False)
    
    @staticmethod
    def get_usage_limits(user: User) -> Dict[str, int]:
        """사용자 사용 제한 조회"""
        if not user.subscription or user.subscription.status != "active":
            return {
                "max_exercises_per_day": 3,
                "max_ai_analysis": 1
            }
        
        plan = user.subscription.plan
        return {
            "max_exercises_per_day": plan.max_exercises_per_day,
            "max_ai_analysis": plan.max_ai_analysis
        }