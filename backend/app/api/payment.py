# backend/app/api/payment.py
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..models.subscription import SubscriptionPlan, UserSubscription, Payment
from ..services.payment_service import PaymentService
from ..core.config import settings
import stripe
import logging

router = APIRouter(prefix="/payment", tags=["payment"])

# Stripe 웹훅 서명 검증용
stripe.api_key = settings.STRIPE_SECRET_KEY

@router.get("/plans")
async def get_subscription_plans(db: Session = Depends(get_db)):
    """구독 플랜 목록 조회"""
    try:
        plans = db.query(SubscriptionPlan).filter(
            SubscriptionPlan.is_active == True
        ).all()
        
        return {
            "success": True,
            "data": [
                {
                    "id": str(plan.id),
                    "name": plan.name,
                    "price": plan.price,
                    "currency": plan.currency,
                    "interval": plan.interval,
                    "features": plan.features,
                    "max_exercises_per_day": plan.max_exercises_per_day,
                    "max_ai_analysis": plan.max_ai_analysis,
                    "premium_content_access": plan.premium_content_access,
                    "priority_support": plan.priority_support
                }
                for plan in plans
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch subscription plans: {str(e)}"
        )

@router.post("/checkout")
async def create_checkout_session(
    plan_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Stripe Checkout 세션 생성"""
    try:
        # 이미 활성 구독이 있는지 확인
        existing_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == current_user.id,
            UserSubscription.status == "active"
        ).first()
        
        if existing_subscription:
            # 기존 구독이 있으면 업그레이드/다운그레이드 처리
            pass  # 향후 구현
        
        session_data = await PaymentService.create_checkout_session(
            current_user, plan_id, db
        )
        
        return {
            "success": True,
            "data": session_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/success")
async def payment_success(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """결제 성공 처리"""
    try:
        subscription = await PaymentService.handle_successful_payment(
            session_id, db
        )
        
        return {
            "success": True,
            "message": "Payment successful! Your subscription is now active.",
            "data": {
                "subscription_id": str(subscription.id),
                "plan_name": subscription.plan.name,
                "status": subscription.status,
                "current_period_end": subscription.current_period_end.isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/subscription")
async def get_user_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """사용자 구독 정보 조회"""
    try:
        subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == current_user.id,
            UserSubscription.status.in_(["active", "past_due"])
        ).first()
        
        if not subscription:
            return {
                "success": True,
                "data": {
                    "subscription": None,
                    "plan": "free",
                    "limits": PaymentService.get_usage_limits(current_user)
                }
            }
        
        return {
            "success": True,
            "data": {
                "subscription": {
                    "id": str(subscription.id),
                    "status": subscription.status,
                    "current_period_start": subscription.current_period_start.isoformat(),
                    "current_period_end": subscription.current_period_end.isoformat(),
                    "cancel_at_period_end": subscription.cancel_at_period_end
                },
                "plan": {
                    "id": str(subscription.plan.id),
                    "name": subscription.plan.name,
                    "price": subscription.plan.price,
                    "currency": subscription.plan.currency,
                    "features": subscription.plan.features
                },
                "limits": PaymentService.get_usage_limits(current_user)
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/cancel")
async def cancel_subscription(
    immediate: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """구독 취소"""
    try:
        subscription = await PaymentService.cancel_subscription(
            str(current_user.id), db, immediate
        )
        
        return {
            "success": True,
            "message": "Subscription canceled successfully",
            "data": {
                "subscription_id": str(subscription.id),
                "status": subscription.status,
                "cancel_at_period_end": subscription.cancel_at_period_end,
                "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/payments")
async def get_payment_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """결제 내역 조회"""
    try:
        payments = db.query(Payment).filter(
            Payment.user_id == current_user.id
        ).order_by(Payment.created_at.desc()).limit(20).all()
        
        return {
            "success": True,
            "data": [
                {
                    "id": str(payment.id),
                    "amount": payment.amount,
                    "currency": payment.currency,
                    "status": payment.status,
                    "payment_method": payment.payment_method,
                    "description": payment.description,
                    "created_at": payment.created_at.isoformat()
                }
                for payment in payments
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Stripe 웹훅 처리"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # 웹훅 이벤트 처리
    try:
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            await PaymentService.handle_successful_payment(session['id'], db)
            
        elif event['type'] == 'customer.subscription.updated':
            subscription = event['data']['object']
            await _handle_subscription_updated(subscription, db)
            
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            await _handle_subscription_canceled(subscription, db)
            
        elif event['type'] == 'invoice.payment_succeeded':
            invoice = event['data']['object']
            await _handle_payment_succeeded(invoice, db)
            
        elif event['type'] == 'invoice.payment_failed':
            invoice = event['data']['object']
            await _handle_payment_failed(invoice, db)
            
        return {"success": True}
        
    except Exception as e:
        logging.error(f"Webhook processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

async def _handle_subscription_updated(stripe_subscription: Dict, db: Session):
    """구독 업데이트 처리"""
    subscription = db.query(UserSubscription).filter(
        UserSubscription.stripe_subscription_id == stripe_subscription['id']
    ).first()
    
    if subscription:
        subscription.status = stripe_subscription['status']
        subscription.current_period_start = datetime.fromtimestamp(
            stripe_subscription['current_period_start']
        )
        subscription.current_period_end = datetime.fromtimestamp(
            stripe_subscription['current_period_end']
        )
        subscription.cancel_at_period_end = stripe_subscription.get('cancel_at_period_end', False)
        subscription.updated_at = datetime.utcnow()
        db.commit()

async def _handle_subscription_canceled(stripe_subscription: Dict, db: Session):
    """구독 취소 처리"""
    subscription = db.query(UserSubscription).filter(
        UserSubscription.stripe_subscription_id == stripe_subscription['id']
    ).first()
    
    if subscription:
        subscription.status = "canceled"
        subscription.updated_at = datetime.utcnow()
        db.commit()

async def _handle_payment_succeeded(invoice: Dict, db: Session):
    """결제 성공 처리"""
    if invoice.get('subscription'):
        subscription = db.query(UserSubscription).filter(
            UserSubscription.stripe_subscription_id == invoice['subscription']
        ).first()
        
        if subscription:
            # 결제 기록 생성
            payment = Payment(
                user_id=subscription.user_id,
                subscription_id=subscription.id,
                stripe_payment_intent_id=invoice['payment_intent'],
                amount=invoice['amount_paid'] / 100,
                currency=invoice['currency'],
                status="succeeded",
                payment_method="card",
                description=f"Subscription renewal - {invoice['lines']['data'][0]['description']}"
            )
            db.add(payment)
            db.commit()

async def _handle_payment_failed(invoice: Dict, db: Session):
    """결제 실패 처리"""
    if invoice.get('subscription'):
        subscription = db.query(UserSubscription).filter(
            UserSubscription.stripe_subscription_id == invoice['subscription']
        ).first()
        
        if subscription:
            subscription.status = "past_due"
            subscription.updated_at = datetime.utcnow()
            
            # 실패한 결제 기록 생성
            payment = Payment(
                user_id=subscription.user_id,
                subscription_id=subscription.id,
                amount=invoice['amount_due'] / 100,
                currency=invoice['currency'],
                status="failed",
                payment_method="card",
                description=f"Failed payment - {invoice['lines']['data'][0]['description']}"
            )
            db.add(payment)
            db.commit()