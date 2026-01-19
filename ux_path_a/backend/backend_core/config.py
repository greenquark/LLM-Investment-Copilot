"""
Configuration settings for UX Path A backend.

Uses environment variables with sensible defaults.
Can also load from centralized LLM config in config/env.backtest.yaml.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # CORS - supports comma-separated list from env or defaults to localhost
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"
    
    # Database - defaults to SQLite for local, PostgreSQL for production
    DATABASE_URL: str = "sqlite:///./ux_path_a.db"
    
    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = ""  # Will be loaded from centralized config if empty
    OPENAI_MAX_TOKENS: int = 2000
    OPENAI_TEMPERATURE: float = 0.7
    
    # Token budgets (INV-LLM-03)
    MAX_TOKENS_PER_SESSION: int = 100000
    MAX_TOKENS_PER_MESSAGE: int = 4000
    
    # Security
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Cache
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour

    # Web search (optional tool)
    WEB_SEARCH_ENABLED: bool = True
    # auto | tavily | duckduckgo
    WEB_SEARCH_PROVIDER: str = "auto"
    TAVILY_API_KEY: str = ""
    WEB_SEARCH_TIMEOUT_SECONDS: float = 12.0
    WEB_SEARCH_DEFAULT_MAX_RESULTS: int = 5

    # Web search: 2-step pipeline (search -> extract top docs)
    WEB_SEARCH_EXTRACT_ENABLED: bool = True
    # How many of the search results to extract full text for (kept small for latency/cost)
    WEB_SEARCH_EXTRACT_MAX_DOCS: int = 3
    # Hard cap on extracted text returned per document (avoid huge tool payloads)
    WEB_SEARCH_EXTRACT_MAX_CHARS: int = 4000
    WEB_SEARCH_EXTRACT_TIMEOUT_SECONDS: float = 18.0
    
    # Platform integration
    PLATFORM_ROOT: Path = Path(__file__).parent.parent.parent.parent
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        """Initialize settings and load from centralized LLM config if needed."""
        super().__init__(**kwargs)
        
        # Handle PORT from Railway (Railway sets PORT env var)
        import os
        if 'PORT' in os.environ:
            self.PORT = int(os.environ['PORT'])

        # Local dev convenience: allow reading secrets from config/secrets.yaml (gitignored)
        # so developers can run tools/scripts without exporting env vars.
        # Hosted deployments should still set env vars (Railway/Vercel).
        if not self.TAVILY_API_KEY:
            try:
                secrets_path = (self.PLATFORM_ROOT / "config" / "secrets.yaml")
                if secrets_path.exists():
                    try:
                        import yaml  # type: ignore
                    except Exception:
                        yaml = None  # type: ignore
                    if yaml is not None:
                        raw = secrets_path.read_text(encoding="utf-8")
                        data = yaml.safe_load(raw) if raw.strip() else None
                        if isinstance(data, dict):
                            ws = data.get("web_search")
                            if isinstance(ws, dict):
                                key = ws.get("tavily_api_key")
                                if isinstance(key, str) and key.strip():
                                    self.TAVILY_API_KEY = key.strip()
            except Exception:
                # Never fail app startup due to missing/invalid local secrets file.
                pass
        
        # Parse CORS_ORIGINS from string to list
        if isinstance(self.CORS_ORIGINS, str):
            self.CORS_ORIGINS = [origin.strip() for origin in self.CORS_ORIGINS.split(',') if origin.strip()]
        
        # If OPENAI_MODEL is not set via env var, try to load from centralized config
        if not self.OPENAI_MODEL:
            try:
                # Try to import and load centralized LLM config
                import sys
                project_root = self.PLATFORM_ROOT
                sys.path.insert(0, str(project_root))
                
                from core.utils.llm_config import load_llm_config
                llm_config = load_llm_config(project_root=project_root)
                self.OPENAI_MODEL = llm_config.get("primary_model", "gpt-5-mini")
                
                # Also update defaults from centralized config if not set via env
                if self.OPENAI_MAX_TOKENS == 2000:  # Default value
                    self.OPENAI_MAX_TOKENS = llm_config.get("max_tokens", 2000)
                if self.OPENAI_TEMPERATURE == 0.7:  # Default value
                    self.OPENAI_TEMPERATURE = llm_config.get("temperature", 0.7)
            except Exception:
                # Fallback to default if centralized config can't be loaded
                if not self.OPENAI_MODEL:
                    self.OPENAI_MODEL = "gpt-5-mini"


settings = Settings()
