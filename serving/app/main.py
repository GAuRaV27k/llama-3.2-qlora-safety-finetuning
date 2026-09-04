from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints

from serving.app.model import classify_safety


app = FastAPI(title="aegis safety")


def query_cleaning(user: str) -> str:
    return user.strip().lower()


class UserQuery(BaseModel):
    user_query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ] = Field(
        ...,
        description="User-entered prompt",
    )


@app.get("/")
def api_info() -> dict:
    return {
        "name": "Aegis-Serve",
        "description": "Llama + QLoRA safety classification service",
        "endpoints": ["GET /", "GET /health", "POST /classify"],
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "healthy"}


@app.post("/classify")
def classify(query: UserQuery) -> dict:
    clean = query_cleaning(query.user_query)
    response = classify_safety(clean)

    return {
        "input": clean,
        **response,
    }
