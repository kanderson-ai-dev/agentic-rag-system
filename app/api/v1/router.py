"""Aggregates all v1 API routers."""

from fastapi import APIRouter

from app.api.v1.routes import rag, review

api_v1_router = APIRouter()
api_v1_router.include_router(rag.router)
api_v1_router.include_router(review.router)
