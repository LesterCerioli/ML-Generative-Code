from typing import Optional, Dict, Any
from app.database import db
from app.services.user_service import UserService  # Importa a classe, não a instância

class UserCRUD:
    def __init__(self):
        self.user_service = UserService()  # Instância local
    
    def create_user(self, name: str, email: str, password: str, 
                   organization_name: str) -> Optional[Dict[str, Any]]:
        
        try:
            
            result = self.user_service.create_user(name, email, password, organization_name)
            if result:
                # Remove sensitive data before returning
                result.pop('password', None)
                result.pop('organization_id', None)
            return result
            
        except Exception as e:
            print(f"Error creating user: {e}")
            return None
    
    def authenticate_user(self, email: str, password: str, organization_name: str) -> Optional[Dict[str, Any]]:
        
        try:
            # Usa authenticate_user em vez de authenticate_user_by_role
            auth_result = self.user_service.authenticate_user(email, password, organization_name)
            return auth_result
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
    
    def change_user_password(self, user_id: str, current_password: str, 
                           new_password: str, organization_name: str) -> bool:
        
        try:
            
            org_id = self.user_service.get_organization_id_by_name(organization_name)
            if not org_id:
                return False
                
            
            user = self.get_user_by_id(user_id, organization_name)
            if not user:
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
                print(f"Database error changing password: {e}")
                return False
                    
        except Exception as e:
            print(f"Error changing password: {e}")
            return False
    
    def get_user_by_id(self, user_id: str, organization_name: str) -> Optional[Dict[str, Any]]:
        
        try:
            org_id = self.user_service.get_organization_id_by_name(organization_name)
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
                print(f"Database error getting user: {e}")
                return None
            
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    def get_user_by_email(self, email: str, organization_name: str) -> Optional[Dict[str, Any]]:
        
        try:
            org_id = self.user_service.get_organization_id_by_name(organization_name)
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
                print(f"Database error getting user by email: {e}")
                return None
            
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None
    
    def get_organization_users(self, organization_name: str) -> Optional[Dict[str, Any]]:
        
        try:
            org_id = self.user_service.get_organization_id_by_name(organization_name)
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
                            # Converter para lista de dicionários
                            return [dict(user) for user in users]
                        return []
                        
            except Exception as e:
                print(f"Database error getting organization users: {e}")
                return None
            
        except Exception as e:
            print(f"Error getting organization users: {e}")
            return None
    
    def update_user(self, user_id: str, update_data: Dict[str, Any], organization_name: str) -> Optional[Dict[str, Any]]:
        
        try:
            org_id = self.user_service.get_organization_id_by_name(organization_name)
            if not org_id:
                return None
            
            
            allowed_fields = ['name', 'email']
                        
            set_clauses = []
            values = []
            for field, value in update_data.items():
                if field in allowed_fields:
                    set_clauses.append(f"{field} = %s")
                    values.append(value)
            
            if not set_clauses:
                return None
            
            values.extend([user_id, org_id])
            
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f'''
                        UPDATE public.users 
                        SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s AND organization_id = %s AND deleted_at IS NULL
                        RETURNING id, name, email, created_at, updated_at
                    ''', values)
                    
                    result = cursor.fetchone()
                    conn.commit()
                    return dict(result) if result else None
                    
        except Exception as e:
            print(f"Error updating user: {e}")
            return None
    
    def delete_user(self, user_id: str, organization_name: str) -> bool:
        
        try:
            org_id = self.user_service.get_organization_id_by_name(organization_name)
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
            print(f"Error deleting user: {e}")
            return False
    
    def reset_password(self, email: str, new_password: str, organization_name: str) -> bool:
        
        try:
            org_id = self.user_service.get_organization_id_by_name(organization_name)
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
            print(f"Error resetting password: {e}")
            return False


user_crud = UserCRUD()