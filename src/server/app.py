from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.server.routes.upload import router as upload_docs

load_dotenv()

app = FastAPI(
    title="agentic_rag",
    version="1.0.0"
)

PREFIX = "/api/v1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(upload_docs, prefix=PREFIX, tags=["uploads"])
