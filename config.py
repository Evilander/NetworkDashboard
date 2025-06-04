import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file if present
load_dotenv()

# Basic logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = os.getenv('DB_PATH', 'network_traffic.db')

# Refresh settings
AUTO_REFRESH_DEFAULT = os.getenv('AUTO_REFRESH_DEFAULT', 'True') == 'True'
REFRESH_INTERVAL_DEFAULT = int(os.getenv('REFRESH_INTERVAL_DEFAULT', '10'))
DEFAULT_TIMEFRAME = os.getenv('DEFAULT_TIMEFRAME', 'day')

# Memory management
MEMORY_WARNING_THRESHOLD = int(os.getenv('MEMORY_WARNING_THRESHOLD', '70'))
MEMORY_CRITICAL_THRESHOLD = int(os.getenv('MEMORY_CRITICAL_THRESHOLD', '85'))
MEMORY_CHECK_INTERVAL = int(os.getenv('MEMORY_CHECK_INTERVAL', '300'))

# LLM / AI API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', os.getenv('LLM_API_KEY'))
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

LLM_API_ENDPOINT = os.getenv('LLM_API_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o')

# Freshdesk configuration
FRESHDESK_DOMAIN = os.getenv('FRESHDESK_DOMAIN')
FRESHDESK_API_KEY = os.getenv('FRESHDESK_API_KEY')
