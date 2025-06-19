from typing import List

from fastapi import UploadFile, APIRouter, File, HTTPException, status

from src.services.parser import parser
from src.services.preprocessing import save_embeddings
from src.utils.exceptions import invalid_file_exception
from src.utils.valid_files import valid_file_types

router = APIRouter()


async def process_and_save_hi_res(file: UploadFile = File(...)):
    file_name = file.filename
    try:
        parsed = await parser.aload_data(file_path=await file.read(), extra_info={'file_name': file_name})
        print(f"total parsed: {len(parsed)}")
        for doc_idx, content in enumerate(parsed):
            docs = content.text
            print(f"Processing document {doc_idx}, length: {len(docs)}")
            if docs:
                print(f"Docs: {docs}")
                embedded = save_embeddings(f"{file_name}_{doc_idx}", docs)
                if embedded:
                    print(f"DOC_INFO | STATUS: Saved")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post("/document/hi-res")
async def upload_docs_(files: List[UploadFile] = File(...)):
    try:
        for file in files:
            if file.content_type not in valid_file_types:
                raise invalid_file_exception

        for file in files:
            await process_and_save_hi_res(file)

        return {"message": "All documents added successfully"}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
