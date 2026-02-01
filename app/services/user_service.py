import bcrypt
from typing import Optional, Dict, Any, List
from app.database import db

class UserService:
    def hash_password(self, password: str) -> str:
        
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed_password.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifies if plain password matches the bcrypt hash"""
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except (ValueError, TypeError):
            return False
    
    def get_organization_id_by_name(self, organization_name: str) -> Optional[str]:
        """Gets organization ID by name (case-insensitive with debug)"""
        print(f"DEBUG: Searching for organization name: '{organization_name}'")
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    # ✅ PARAMETERIZED QUERY
                    cursor.execute('''
                        SELECT id FROM public.organizations 
                        WHERE LOWER(name) = LOWER(%s)
                    ''', (organization_name,))
                    result = cursor.fetchone()
                    print(f"DEBUG: Case-insensitive match result: {result}")
                    return result['id'] if result else None
        except Exception as e:
            print(f"Error fetching organization (case-insensitive): {e}")
            return None
    
    def get_organization_id_exact(self, organization_name: str) -> Optional[str]:
        """Exact match for organization name (including spaces)"""
        try:
            print(f"DEBUG: Exact search for: '{organization_name}'")
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    # ✅ PARAMETERIZED QUERY
                    cursor.execute(
                        "SELECT id FROM public.organizations WHERE name = %s",
                        (organization_name,)
                    )
                    result = cursor.fetchone()
                    print(f"DEBUG: Exact match result: {result}")
                    return result['id'] if result else None
        except Exception as e:
            print(f"Error fetching organization (exact): {e}")
            return None
    
    def get_organization_id_trim(self, organization_name: str) -> Optional[str]:
        """Trimmed match for organization name"""
        try:
            print(f"DEBUG: Trimmed search for: '{organization_name}'")
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    # ✅ PARAMETERIZED QUERY
                    cursor.execute(
                        "SELECT id FROM public.organizations WHERE TRIM(name) = TRIM(%s)",
                        (organization_name,)
                    )
                    result = cursor.fetchone()
                    print(f"DEBUG: Trimmed match result: {result}")
                    return result['id'] if result else None
        except Exception as e:
            print(f"Error fetching organization (trim): {e}")
            return None
    
    def get_all_organizations(self) -> List[Dict[str, Any]]:
        """Get all organizations for debugging"""
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    # ✅ NO PARAMETERS NEEDED - static query
                    cursor.execute("SELECT id, name FROM public.organizations")
                    results = cursor.fetchall()
                    org_list = [dict(result) for result in results]
                    print(f"DEBUG: All organizations in DB: {org_list}")
                    return org_list
        except Exception as e:
            print(f"Error fetching organizations: {e}")
            return []
    
    def organization_exists(self, organization_name: str) -> bool:
        """Checks if organization exists"""
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    # ✅ PARAMETERIZED QUERY
                    cursor.execute(
                        "SELECT EXISTS(SELECT 1 FROM public.organizations WHERE name = %s) as exists",
                        (organization_name,)
                    )
                    result = cursor.fetchone()
                    return result['exists'] if result else False
        except Exception as e:
            print(f"Error checking organization existence: {e}")
            return False
    
    def authenticate_user(self, email: str, password: str, organization_name: str) -> Optional[Dict[str, Any]]:
        """Authenticates user by verifying password against stored hash"""
        try:
            print(f"DEBUG: Authenticating user for org: '{organization_name}'")
                        
            if not self.organization_exists(organization_name):
                print(f"DEBUG: Organization '{organization_name}' does not exist")
                return None
            
            org_id = self.get_organization_id_by_name(organization_name)
            if not org_id:
                print(f"DEBUG: Could not get ID for organization '{organization_name}'")
                return None
            
            print(f"DEBUG: Organization ID found: {org_id}")
            
            # ✅ PARAMETERIZED QUERY for user authentication
            try:
                with db.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute('''
                            SELECT id, name, email, password, created_at
                            FROM public.users 
                            WHERE email = %s 
                            AND organization_id = %s 
                            AND deleted_at IS NULL
                        ''', (email, org_id))
                        user_data = cursor.fetchone()
                        
                        if not user_data:
                            print(f"DEBUG: User with email '{email}' not found in organization {org_id}")
                            return None
                        
                        user_dict = dict(user_data)
                        
                        if not self.verify_password(password, user_dict['password']):
                            print("DEBUG: Password verification failed")
                            return None
                        
                        user_dict.pop('password', None)
                        print("DEBUG: Authentication successful")
                        return user_dict
                        
            except Exception as e:
                print(f"Error fetching user: {e}")
                return None
                    
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
        
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Finds user by email across all organizations"""
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    # ✅ PARAMETERIZED QUERY
                    cursor.execute('''
                        SELECT id, name, email, password, created_at, organization_id
                        FROM public.users 
                        WHERE email = %s AND deleted_at IS NULL
                    ''', (email,))
                    result = cursor.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            print(f"Error fetching user by email: {e}")
            return None
             
    def create_user(self, name: str, email: str, password: str, organization_name: str) -> Optional[Dict[str, Any]]:
        """Creates a new user with organization validation"""
        try:
            print(f"DEBUG: Creating user for organization: '{organization_name}'")
            print(f"DEBUG: User details - Name: {name}, Email: {email}")
                        
            all_orgs = self.get_all_organizations()
                        
            org_id = self.get_organization_id_exact(organization_name)
            if not org_id:
                org_id = self.get_organization_id_trim(organization_name)
            if not org_id:
                org_id = self.get_organization_id_by_name(organization_name)
            
            print(f"DEBUG: Final organization ID found: {org_id}")
            
            if not org_id:
                org_names = [org['name'] for org in all_orgs]
                error_msg = f"Organization '{organization_name}' not found. Available organizations: {org_names}"
                print(f"VALIDATION ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            hashed_password = self.hash_password(password)
            print("DEBUG: Password hashed successfully")
            
            # ✅ PARAMETERIZED QUERY for creating user
            try:
                with db.get_connection() as conn:
                    with conn.cursor() as cursor:
                        # Check if user already exists
                        cursor.execute('''
                            SELECT id FROM public.users 
                            WHERE email = %s AND organization_id = %s AND deleted_at IS NULL
                        ''', (email, org_id))
                        existing_user = cursor.fetchone()
                        
                        if existing_user:
                            print(f"DEBUG: User with email '{email}' already exists in organization {org_id}")
                            return None
                        
                        # Insert new user
                        cursor.execute('''
                            INSERT INTO public.users 
                            (name, email, password, organization_id, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                            RETURNING id, name, email, created_at
                        ''', (name, email, hashed_password, org_id))
                        
                        result = cursor.fetchone()
                        conn.commit()
                        
                        if result:
                            print(f"DEBUG: User created successfully: {dict(result)}")
                            return dict(result)
                        else:
                            print("DEBUG: User creation failed")
                            return None
                            
            except Exception as e:
                print(f"Database error creating user: {e}")
                return None
            
        except ValueError as e:
            print(f"VALIDATION ERROR: {e}")
            return None
        except Exception as e:
            print(f"ERROR creating user: {e}")
            return None
    
    def reset_password_by_email(self, email: str, new_password: str) -> bool:
        """Updates user password"""
        try:
            user_data = self.get_user_by_email(email)
            if not user_data:
                return False

            hashed_password = self.hash_password(new_password)

            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    # ✅ PARAMETERIZED QUERY
                    cursor.execute('''
                        UPDATE public.users 
                        SET password = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE email = %s AND deleted_at IS NULL
                    ''', (hashed_password, email))
                    conn.commit()
                    
                    return cursor.rowcount > 0  # Returns True if a row was updated
        except Exception as e:
            print(f"Error resetting password: {e}")
            return False



user_service = UserService()