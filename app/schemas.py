from enum import Enum
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Dict, Optional, List
from datetime import datetime, date
from uuid import UUID

class AuthTokenRequest(BaseModel):
    client_id: str
    client_secret: str
    
class TokenValidationRequest(BaseModel):
    token: str

class AuthenticatedRequest(BaseModel):
    token: str

class TokenValidationResponse(BaseModel):
    valid: bool
    message: str

class AuthenticatedRequest(BaseModel):
    token: str
    
# ==================================================
#              ORGANIZATION SCHEMAS 
# ==================================================

class OrganizationBase(BaseModel):
    name: str
    address: Optional[str] = None
    cnpj: Optional[str] = None
    ein: Optional[str] = None

class OrganizationCreate(OrganizationBase):
    token: str

class OrganizationUpdate(BaseModel):
    token: str
    name: Optional[str] = None
    address: Optional[str] = None
    cnpj: Optional[str] = None
    ein: Optional[str] = None

class OrganizationResponse(BaseModel):
    id: UUID
    name: str
    address: Optional[str]
    cnpj: Optional[str]
    ein: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class OrganizationDetailResponse(OrganizationResponse):
    statistics: dict = {}

class OrganizationListResponse(BaseModel):
    organizations: List[OrganizationResponse]
    total_count: int
    page: int
    page_size: int

class OrganizationFilter(BaseModel):
    token: str
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)

class OrganizationSearchRequest(BaseModel):
    token: str
    query: str
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)

class CNPJValidationRequest(BaseModel):
    token: str
    cnpj: str

class CNPJValidationResponse(BaseModel):
    cnpj: str
    is_valid_format: bool
    is_available: bool
    cleaned_cnpj: str

class EINValidationRequest(BaseModel):
    token: str
    ein: str

class EINValidationResponse(BaseModel):
    ein: str
    is_valid_format: bool
    is_available: bool
    cleaned_ein: str

class DeactivationRequest(BaseModel):
    token: str
    reason: Optional[str] = None

class ReactivationRequest(BaseModel):
    token: str

class OrganizationSettingsRequest(BaseModel):
    token: str
    settings: dict
