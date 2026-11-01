from pydantic import BaseModel, Field


class SQLStatement(BaseModel):
    query: str = Field(min_length=1)
