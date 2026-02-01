from typing import Optional, Dict, Any
import time
import re
import logging
from app.database import db
from app.services.user_service import UserService


logger = logging.getLogger(__name__)

class UserCRUD:
    def __init__(self):
        self.user_service = UserService()
    
    def _validate_email(self, email: str) -> bool:
        
        if not email or not isinstance(email, str):
            return False
        if len(email) > 255:
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _sanitize_string(self, text: str, max_length: int = 255) -> Optional[str]:
        
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        if len(text) > max_length:
            return None
        
        text = text.replace('--', '').replace('/*', '').replace('*/', '')
        return text
    
    def _validate_uuid(self, uuid_str: str) -> bool:
        
        if not uuid_str or not isinstance(uuid_str, str):
            return False
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(uuid_pattern, uuid_str.lower()))
    
    def _log_error_secure(self, method: str, error: Exception):
        
        error_type = type(error).__name__
        logger.error(f"Error in UserCRUD.{method}: {error_type}")
    
    def create_user(self, name: str, email: str, password: str, 
                   organization_name: str) -> Optional[Dict[str, Any]]:
        try:
            
            name = self._sanitize_string(name)
            if not name:
                return None
                
            if not self._validate_email(email):
                return None
                
            if not password or len(password) < 8:
                return None
                
            org_name = self._sanitize_string(organization_name)
            if not org_name:
                return None
            
            result = self.user_service.create_user(name, email, password, org_name)
            if result:
                result.pop('password', None)
                result.pop('organization_id', None)
            return result
        except Exception as e:
            self._log_error_secure("create_user", e)
            return None
    
    def authenticate_user(self, email: str, password: str, organization_name: str) -> Optional[Dict[str, Any]]:
        try:
            
            if not self._validate_email(email):
                return None
                
            if not password:
                return None
                
            org_name = self._sanitize_string(organization_name)
            if not org_name:
                return None
            
            auth_result = self.user_service.authenticate_user(email, password, org_name)
            return auth_result
        except Exception as e:
            self._log_error_secure("authenticate_user", e)
            return None
    
    def change_user_password(self, user_id: str, current_password: str, 
                           new_password: str, organization_name: str) -> bool:
        try:
            
            if not self._validate_uuid(user_id):
                time.sleep(0.1)  
                return False
                
            if not current_password or not new_password:
                return False
                
            if len(new_password) < 8:
                return False
                
            org_name = self._sanitize_string(organization_name)
            if not org_name:
                return False
            
            org_id = self.user_service.get_organization_id_by_name(org_name)
            if not org_id:
                time.sleep(0.1)  # Constant time response
                return False
            
            
            try:
                with db.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute('''
                            SELECT password FROM public.users 
                            WHERE id = %s AND organization_id = %s AND deleted_at IS NULL
                        ''', (user_id, org_id))
                        user_data = cursor.fetchone()
                        
                        
                        if not user_data:
                            
                            dummy_hash = "$2b$12$" + "0" * 53
                            self.user_service.verify_password(current_password, dummy_hash)
                            time.sleep(0.1)
                            return False
                        
                        if not self.user_service.verify_password(current_password, user_data['password']):
                            return False
                        
                        new_hashed_password = self.user_service.hash_password(new_password)
                                                
                        cursor.execute('''
                            UPDATE public.users 
                            SET password = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s AND organization_id = %s AND deleted_at IS NULL
                        ''', (new_hashed_password, user_id, org_id))
                        
                        conn.commit()
                        return cursor.rowcount > 0
                        
            except Exception as e:
                self._log_error_secure("change_user_password_db", e)
                return False
                    
        except Exception as e:
            self._log_error_secure("change_user_password", e)
            return False
    
    def get_user_by_id(self, user_id: str, organization_name: str) -> Optional[Dict[str, Any]]:
        try:
            
            if not self._validate_uuid(user_id):
                return None
                
            org_name = self._sanitize_string(organization_name)
            if not org_name:
                return None
            
            org_id = self.user_service.get_organization_id_by_name(org_name)
            if not org_id:
                return None
            
            try:
                with db.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute('''
                            SELECT id, name, email, organization_id, created_at, updated_at
                            FROM public.users 
                            WHERE id = %s AND organization_id = %s AND deleted_at IS NULL
                        ''', (user_id, org_id))
                        
                        user = cursor.fetchone()
                        if user:
                            user_dict = dict(user)
                            user_dict.pop('organization_id', None)
                            return user_dict
                        return None
                        
            except Exception as e:
                self._log_error_secure("get_user_by_id_db", e)
                return None
            
        except Exception as e:
            self._log_error_secure("get_user_by_id", e)
            return None
    
    def get_user_by_email(self, email: str, organization_name: str) -> Optional[Dict[str, Any]]:
        try:
            
            if not self._validate_email(email):
                return None
                
            org_name = self._sanitize_string(organization_name)
            if not org_name:
                return None
            
            org_id = self.user_service.get_organization_id_by_name(org_name)
            if not org_id:
                return None
            
            try:
                with db.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute('''
                            SELECT id, name, email, organization_id, created_at, updated_at
                            FROM public.users 
                            WHERE email = %s AND organization_id = %s AND deleted_at IS NULL
                        ''', (email, org_id))
                        
                        user = cursor.fetchone()
                        if user:
                            user_dict = dict(user)
                            user_dict.pop('organization_id', None)
                            return user_dict
                        return None
                        
            except Exception as e:
                self._log_error_secure("get_user_by_email_db", e)
                return None
            
        except Exception as e:
            self._log_error_secure("get_user_by_email", e)
            return None
    
    def get_organization_users(self, organization_name: str) -> Optional[Dict[str, Any]]:
        try:
            
            org_name = self._sanitize_string(organization_name)
            if not org_name:
                return None
            
            org_id = self.user_service.get_organization_id_by_name(org_name)
            if not org_id:
                return None
            
            try:
                with db.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute('''
                            SELECT id, name, email, created_at, updated_at
                            FROM public.users 
                            WHERE organization_id = %s AND deleted_at IS NULL
                            ORDER BY created_at DESC
                        ''', (org_id,))
                        
                        users = cursor.fetchall()
                        if users:
                            return [dict(user) for user in users]
                        return []
                        
            except Exception as e:
                self._log_error_secure("get_organization_users_db", e)
                return None
            
        except Exception as e:
            self._log_error_secure("get_organization_users", e)
            return None
    
    def update_user(self, user_id: str, update_data: Dict[str, Any], organization_name: str) -> Optional[Dict[str, Any]]:
        try:
            
            if not self._validate_uuid(user_id):
                return None
                
            if not update_data or not isinstance(update_data, dict):
                return None
                
            org_name = self._sanitize_string(organization_name)
            if not org_name:
                return None
            
            org_id = self.user_service.get_organization_id_by_name(org_name)
            if not org_id:
                return None
            
            
            allowed_fields = {'name', 'email'}
            
            set_clauses = []
            values = []
            
            for field, value in update_data.items():
                
                if field == 'name':
                    sanitized_value = self._sanitize_string(str(value))
                    if sanitized_value:
                        set_clauses.append("name = %s")
                        values.append(sanitized_value)
                elif field == 'email':
                    if self._validate_email(str(value)):
                        set_clauses.append("email = %s")
                        values.append(value)
            
            if not set_clauses:
                return None
            
            values.extend([user_id, org_id])
            
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    
                    query = f'''
                        UPDATE public.users 
                        SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s AND organization_id = %s AND deleted_at IS NULL
                        RETURNING id, name, email, created_at, updated_at
                    '''
                    
                    cursor.execute(query, values)
                    
                    result = cursor.fetchone()
                    conn.commit()
                    return dict(result) if result else None
                    
        except Exception as e:
            self._log_error_secure("update_user", e)
            return None
    
    def delete_user(self, user_id: str, organization_name: str) -> bool:
        try:
            
            if not self._validate_uuid(user_id):
                return False
                
            org_name = self._sanitize_string(organization_name)
            if not org_name:
                return False
            
            org_id = self.user_service.get_organization_id_by_name(org_name)
            if not org_id:
                return False
            
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        UPDATE public.users 
                        SET deleted_at = CURRENT_TIMESTAMP
                        WHERE id = %s AND organization_id = %s AND deleted_at IS NULL
                    ''', (user_id, org_id))
                    
                    conn.commit()
                    return cursor.rowcount > 0
                    
        except Exception as e:
            self._log_error_secure("delete_user", e)
            return False
    
    def reset_password(self, email: str, new_password: str, organization_name: str) -> bool:
        try:
            
            if not self._validate_email(email):
                return False
                
            if not new_password or len(new_password) < 8:
                return False
                
            org_name = self._sanitize_string(organization_name)
            if not org_name:
                return False
            
            org_id = self.user_service.get_organization_id_by_name(org_name)
            if not org_id:
                return False
            
            hashed_password = self.user_service.hash_password(new_password)
            
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        UPDATE public.users 
                        SET password = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE email = %s AND organization_id = %s AND deleted_at IS NULL
                    ''', (hashed_password, email, org_id))
                    
                    conn.commit()
                    return cursor.rowcount > 0
                    
        except Exception as e:
            self._log_error_secure("reset_password", e)
            return False


user_crud = UserCRUD()