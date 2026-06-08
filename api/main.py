from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import bbn

app = FastAPI(title="pybbn")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bbn.router, prefix="/api")
