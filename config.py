from dotenv import load_dotenv
import os

load_dotenv()


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL", "groq-2.0-pro")