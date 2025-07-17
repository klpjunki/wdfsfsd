from datetime import datetime, timedelta

def apply_boost(user, hours=24):
    now = datetime.utcnow()
    if not user.boost_expiry or user.boost_expiry < now:
        user.boost_expiry = now + timedelta(hours=hours)
    else:
        user.boost_expiry += timedelta(hours=hours)
