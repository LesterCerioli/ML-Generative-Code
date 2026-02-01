import json
import os
import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any, List, Tuple
from app.config import config
import contextlib
import uuid
from datetime import date, datetime, timedelta
from uuid import UUID


class Database:
    def __init__(self):
        self.connection_string = config.DATABASE_URL
    
    @contextlib.contextmanager
    def get_connection(self):
        """Context manager to handle database connections"""
        conn = psycopg2.connect(
            self.connection_string,
            cursor_factory=RealDictCursor
        )
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        """Initializes the database tables"""
        print("INFO: Database tables already exist, skipping table creation")
    
    def organization_exists(self, organization_name: str) -> bool:
        """Checks if an organization exists by name (case-insensitive)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT EXISTS (SELECT 1 FROM public.organizations WHERE LOWER(TRIM(name)) = LOWER(TRIM(%s))) AS exists",
                        (organization_name,)
                    )
                    result = cursor.fetchone()
                    return result['exists']
        except Exception as e:
            print(f"Error checking organization: {e}")
            return False
    
    def get_organization_id(self, organization_name: str) -> Optional[str]:
        """Gets the organization ID by name (case-insensitive with debug)"""
        try:
            print(f"DEBUG: Searching for organization: '{organization_name}'")
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, name FROM public.organizations WHERE LOWER(TRIM(name)) = LOWER(TRIM(%s))",
                        (organization_name,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        print(f"DEBUG: Organization found - ID: {result['id']}, Name: '{result['name']}'")
                        return result['id']
                    else:
                        cursor.execute("SELECT id, name FROM public.organizations")
                        all_orgs = cursor.fetchall()
                        print(f"DEBUG: Available organizations: {[dict(org) for org in all_orgs]}")
                        return None
        except Exception as e:
            print(f"Error fetching organization: {e}")
            return None
        
class AsyncDatabase:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.connection_string = config.DATABASE_URL  # Use a mesma connection string
    
    async def connect(self):
        """Create async connection pool"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                dsn=self.connection_string,
                min_size=1,
                max_size=20,
                command_timeout=60
            )
    
    @contextlib.asynccontextmanager
    async def get_connection(self):
        """Async context manager to handle database connections"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            try:
                yield connection
            finally:
                
                pass
    
    async def close(self):
        """Close async connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
        
    
        
    

# Global database instance
db = Database()
async_db = AsyncDatabase()