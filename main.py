from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

from src.routers.menuRouter import router as file_router
from src.routers.resturantRouter import router as resturant_router
from src.db import init_db
from src.routers.resturantRouter import router

from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):

    await init_db()
    print("Tables created")

    yield

app = FastAPI(
    title="File Processing API",
    version="1.0.0",
    lifespan=lifespan   # ye line missing thi
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)

app.include_router(
    file_router,
    prefix="/api",
    tags=["File Processing"]
)

app.include_router(
    resturant_router,
    prefix="/api",
    tags=["Resturant"]
)


@app.get("/")
def health_check():
    return {
        "message": "API is running"
    }
