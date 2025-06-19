import os

from dotenv import load_dotenv
from llama_parse import LlamaParse, ResultType

load_dotenv()

parser = LlamaParse(
    api_key=str(os.getenv("LLAMA_API_KEY")),
    result_type=ResultType.MD,
    verbose=True,
)
