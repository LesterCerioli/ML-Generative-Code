from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware 
from typing import List, Dict, Any
from pydantic import BaseModel
from app import schemas
from app.crud import user_crud
from app.database import db
from app.services.auth_service import auth_token_service
import jwt

app = FastAPI(
    title="AI Generative Code Generator",
    description="Microservice to manage users and authentications",
    version="1.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    db.init_db()


async def validate_token_from_body(token: str) -> Dict[str, Any]:
    """
    Dependency that validates JWT token from request body and returns token data
    """
    if not token:
        raise HTTPException(status_code=401, detail="Token is required")
    
    if not auth_token_service.validate_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    
    try:
        decoded_token = jwt.decode(token, auth_token_service.jwt_secret, algorithms=["HS256"])
        return {
            "client_id": decoded_token.get("client_id"),
            "token": token
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")


class AuthenticatedUserCreate(BaseModel):
    token: str
    name: str
    email: str
    password: str
    role: str
    organization_name: str

class AuthenticatedUserLogin(BaseModel):
    token: str
    email: str
    password: str
    role: str

class AuthenticatedRequest(BaseModel):
    token: str
    organization_name: str

class AuthenticatedUserUpdate(BaseModel):
    token: str
    name: Optional[str] = None
    email: Optional[str] = None
    

class AuthenticatedPasswordChange(BaseModel):
    token: str
    current_password: str
    new_password: str

class AuthenticatedDeleteRequest(BaseModel):
    token: str
    organization_name: str

class HealthCheckRequest(BaseModel):
    token: str

class RootRequest(BaseModel):
    token: str


@app.post("/auth/token")
async def generate_auth_token(auth_request: schemas.AuthTokenRequest):
    """
    Generate JWT authentication token
    
    - **client_id**: Client identifier
    - **client_secret**: Client secret
    """
    try:
        
        if not auth_request.client_id or not auth_request.client_secret:
            raise HTTPException(
                status_code=400, 
                detail="Both 'client_id' and 'client_secret' are required"
            )
        
        result = auth_token_service.generate_token(
            auth_request.client_id,
            auth_request.client_secret
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/auth/validate")
async def validate_auth_token(validate_request: schemas.TokenValidationRequest):
    """
    Validate JWT authentication token
    
    - **token**: JWT token to validate
    """
    is_valid = auth_token_service.validate_token(validate_request.token)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"valid": True, "message": "Token is valid"}

@app.get("/auth/token/{client_id}")
async def get_valid_token(client_id: str):
    """Get valid token for client_id"""
    token = auth_token_service.get_valid_token(client_id)
    if not token:
        raise HTTPException(status_code=404, detail="No valid token found")
    return {"token": token}

@app.delete("/auth/cleanup")
async def cleanup_tokens():
    """Clean up expired tokens (admin endpoint)"""
    deleted_count = auth_token_service.cleanup_expired_tokens()
    return {"message": f"Cleaned up {deleted_count} expired tokens"}


@app.post("/users/register", response_model=schemas.UserResponse)
async def register_user(user: AuthenticatedUserCreate):
    """
    Register a new user (Requires authentication token in body)
    
    - **token**: JWT token in request body
    - **name**: User's full name
    - **email**: User's email (must be unique per organization)
    - **password**: User's password
    - **organization_name**: Organization name
    """
    
    token_data = await validate_token_from_body(user.token)
    print(f"Register user request from client: {token_data['client_id']}")
    
    
    user_data = user.dict()
    user_data.pop('token')
    
    result = user_crud.create_user(
        name=user_data['name'],
        email=user_data['email'],
        password=user_data['password'],
        organization_name=user_data['organization_name']
    )
    
    if not result:
        raise HTTPException(
            status_code=400, 
            detail="Failed to create user. Organization may not exist or email already registered."
        )
    
    return result

@app.post("/users/login", response_model=schemas.UserResponse)
async def login_user(login: AuthenticatedUserLogin):
    """
    Authenticate user (Requires authentication token in body)
    
    - **token**: JWT token in request body
    - **email**: User's email
    - **password**: User's password
    
    """
    
    token_data = await validate_token_from_body(login.token)
    print(f"Login attempt from client: {token_data['client_id']}")
    
    
    login_data = login.dict()
    login_data.pop('token')
    
    result = user_crud.authenticate_user(
        email=login_data['email'],
        password=login_data['password'],
        
    )
    
    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials or organization")
    
    return result

@app.post("/users/{user_id}", response_model=schemas.UserResponse)
async def get_user(user_id: str, request: AuthenticatedRequest):
    """
    Get user by ID (Requires authentication token in body)
    
    - **token**: JWT token in request body
    - **user_id**: User UUID
    - **organization_name**: Organization name for validation
    """
    
    token_data = await validate_token_from_body(request.token)
    
    result = user_crud.get_user_by_id(user_id, request.organization_name)
    
    if not result:
        raise HTTPException(
            status_code=404, 
            detail="User not found or doesn't belong to this organization"
        )
    
    return result

@app.post("/users", response_model=List[schemas.UserResponse])
async def get_organization_users(request: AuthenticatedRequest):
    """
    Get all users in an organization (Requires authentication token in body)
    
    - **token**: JWT token in request body
    - **organization_name**: Organization name
    """
    
    token_data = await validate_token_from_body(request.token)
    
    result = user_crud.get_organization_users(request.organization_name)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    return result

@app.put("/users/{user_id}", response_model=schemas.UserResponse)
async def update_user(user_id: str, request: AuthenticatedUserUpdate, organization_name: str):
    """
    Update user information (Requires authentication token in body)
    
    - **token**: JWT token in request body
    - **user_id**: User UUID
    - **organization_name**: Organization name for validation
    """
    
    token_data = await validate_token_from_body(request.token)
    
    
    update_data = request.dict(exclude_unset=True)
    update_data.pop('token')
    
    result = user_crud.update_user(
        user_id, 
        update_data, 
        organization_name
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="User not found or update failed")
    
    return result

@app.post("/users/{user_id}/change-password")
async def change_password(user_id: str, request: AuthenticatedPasswordChange, organization_name: str):
    """
    Change user password (Requires authentication token in body)
    
    - **token**: JWT token in request body
    - **user_id**: User UUID
    - **current_password**: Current password
    - **new_password**: New password
    - **organization_name**: Organization name for validation
    """
    
    token_data = await validate_token_from_body(request.token)
    
    
    password_data = request.dict()
    password_data.pop('token')
    
    success = user_crud.change_user_password(
        user_id=user_id,
        current_password=password_data['current_password'],
        new_password=password_data['new_password'],
        organization_name=organization_name
    )
    
    if not success:
        raise HTTPException(
            status_code=400, 
            detail="Password change failed. Invalid current password or user not found."
        )
    
    return {"message": "Password changed successfully"}

@app.delete("/users/{user_id}")
async def delete_user(user_id: str, request: AuthenticatedDeleteRequest):
    """
    Delete a user (soft delete) (Requires authentication token in body)
    
    - **token**: JWT token in request body
    - **user_id**: User UUID
    - **organization_name**: Organization name for validation
    """
    
    token_data = await validate_token_from_body(request.token)
    
    success = user_crud.delete_user(user_id, request.organization_name)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User deleted successfully"}


@app.post("/health")
async def health_check(request: HealthCheckRequest):
    """
    Health check endpoint (Requires authentication token in body)
    
    - **token**: JWT token in request body
    """
    token_data = await validate_token_from_body(request.token)
    
    return {
        "status": "healthy", 
        "service": "user-microservice",
        "version": "1.0.0",
        "authenticated_client": token_data['client_id']
    }

@app.post("/")
async def root(request: RootRequest):
    """
    Root endpoint with API information (Requires authentication token in body)
    
    - **token**: JWT token in request body
    """
    token_data = await validate_token_from_body(request.token)
    
    return {
        "message": "User Microservice API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "authenticated_client": token_data['client_id']
    }


@app.get("/docs", include_in_schema=False)
async def get_docs():
    """Documentation endpoint (no authentication required)"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/redoc", include_in_schema=False)
async def get_redoc():
    """ReDoc endpoint (no authentication required)"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/redoc")