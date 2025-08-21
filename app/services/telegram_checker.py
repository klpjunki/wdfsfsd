import requests
import os

def check_telegram_subscription(user_id: int) -> bool:
    """Проверяет подписку пользователя через Telegram Bot API"""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    channel_id = os.getenv("TELEGRAM_CHANNEL_ID") 
    
    if not bot_token or not channel_id:
        print("TELEGRAM_BOT_TOKEN или TELEGRAM_CHANNEL_ID не установлены")
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/getChatMember"
    params = {
        "chat_id": channel_id,
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
