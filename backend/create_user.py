from app.db.database import SessionLocal
from app.models.user import User
from app.core.security import get_password_hash

db = SessionLocal()
try:
    user = User(
        email='admin@test.com',
        username='admin',
        password_hash=get_password_hash('admin123'),
        full_name='Admin User',
        is_active=True
    )
    db.add(user)
    db.commit()
    print('SUCCESS: User created!')
    print('Login with: admin@test.com / admin123')
except Exception as e:
    print(f'Error: {e}')
finally:
    db.close()