from fastapi import HTTPException,status

invalid_file_exception = HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Invalid File Type"
    )
