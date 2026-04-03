"""LLM integration helpers."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from core.config import Settings


class LLMService:
    """Factory for the application LLM client."""

    def __init__(self, settings: Settings) -> None:
        provider = settings.llm_provider.strip().lower()
        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
            self.client = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=settings.llm_temperature,
                timeout=settings.request_timeout_seconds,
            )
            return

        if provider == "gemini":
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")
            self.client = ChatGoogleGenerativeAI(
                google_api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                temperature=settings.llm_temperature,
                timeout=settings.request_timeout_seconds,
            )
            return

        raise ValueError("Unsupported LLM_PROVIDER. Use 'openai' or 'gemini'.")
