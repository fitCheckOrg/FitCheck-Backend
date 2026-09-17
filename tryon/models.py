from pydantic import BaseModel


class TryOnResult(BaseModel):
    url: str
    key: str