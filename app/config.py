from pydantic import BaseModel, Field
import os 
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    database_url: str = Field(
        default=os.getenv("DATABASE_URL", "sqlite:///./data.db")
    )
    api_title: str = Field(
        default=os.getenv("API_TITLE", "Somnomat Webservice")
    )
    api_version: str = Field(
        default=os.getenv("API_VERSION", "0.1.0")
    )
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o
        ]
    )

settings = Settings()