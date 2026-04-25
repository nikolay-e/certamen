from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class UserRegistration(BaseModel):
    username: Annotated[
        str, Field(max_length=50, min_length=3, title="Username")
    ]
    password: Annotated[
        str, Field(max_length=128, min_length=8, title="Password")
    ]


class UserLogin(BaseModel):
    username: Annotated[
        str, Field(max_length=50, min_length=3, title="Username")
    ]
    password: Annotated[str, Field(title="Password")]


class UserResponse(BaseModel):
    id: Annotated[int, Field(title="Id")]
    username: Annotated[str, Field(title="Username")]
    is_admin: Annotated[bool | None, Field(False, title="Is Admin")]


class TokenResponse(BaseModel):
    token: Annotated[str, Field(title="Token")]
    refresh_token: Annotated[str, Field(title="Refresh Token")]
    expires_in: Annotated[str | None, Field("15m", title="Expires In")]
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    refresh_token: Annotated[str, Field(title="Refresh Token")]


class WSAuthMessage(BaseModel):
    type: Literal["auth"]
    token: Annotated[str, Field(min_length=20)]


class WSGetModelsRequest(BaseModel):
    type: Literal["get_models"]


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GOOGLE = "google"


class Models(BaseModel):
    provider: Provider | None = None
    name: str | None = None
    display_name: str | None = None


class WSModelsResponse(BaseModel):
    type: Literal["models"]
    models: dict[str, Models]


class WSGetNodesRequest(BaseModel):
    type: Literal["get_nodes"]


class WSExecutionStartEvent(BaseModel):
    type: Literal["execution_start"]
    execution_id: Annotated[UUID, Field(alias="executionId")]


class WSExecutionCompleteEvent(BaseModel):
    type: Literal["execution_complete"]
    results: dict[str, Any]


class Code(Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CYCLE_DETECTED = "CYCLE_DETECTED"
    NODE_ERROR = "NODE_ERROR"
    TIMEOUT = "TIMEOUT"


class WSExecutionErrorEvent(BaseModel):
    type: Literal["execution_error"]
    error: str
    code: Code | None = None


class WSNodeStartEvent(BaseModel):
    type: Literal["node_start"]
    node_id: Annotated[str, Field(alias="nodeId")]


class WSNodeCompleteEvent(BaseModel):
    type: Literal["node_complete"]
    node_id: Annotated[str, Field(alias="nodeId")]
    outputs: dict[str, Any]


class Code1(Enum):
    TIMEOUT = "TIMEOUT"
    INVALID_INPUT = "INVALID_INPUT"
    MODEL_ERROR = "MODEL_ERROR"
    UNKNOWN = "UNKNOWN"


class WSNodeErrorEvent(BaseModel):
    type: Literal["node_error"]
    node_id: Annotated[str, Field(alias="nodeId")]
    error: str
    code: Code1 | None = None


class Code2(Enum):
    RATE_LIMIT = "RATE_LIMIT"
    INVALID_JSON = "INVALID_JSON"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NODE_NOT_FOUND = "NODE_NOT_FOUND"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    UNAUTHORIZED = "UNAUTHORIZED"


class WSErrorResponse(BaseModel):
    type: Literal["error"]
    message: str
    code: Code2 | None = None
    details: dict[str, Any] | None = None
