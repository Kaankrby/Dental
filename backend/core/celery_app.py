from celery import Celery
from backend.core.config import settings

celery_app = Celery("worker", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

celery_app.conf.task_routes = {
    "backend.worker.tasks.*": {"queue": "main-queue"},
}
