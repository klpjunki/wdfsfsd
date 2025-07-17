import requests
import os
from app.config import settings

def check_telegram_subscription(user_id: int) -> bool:
    """Проверяет подписку пользователя через Telegram Bot API"""
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/getChatMember"
    params = {
        "chat_id": settings.TELEGRAM_CHANNEL_ID,
        "user_id": user_id
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = data.get('result', {}).get('status', '')
            return status in ['member', 'administrator', 'creator']
        return False
    except Exception as e:
        print(f"Telegram API error: {e}")
        return False
