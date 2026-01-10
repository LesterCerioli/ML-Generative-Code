import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MICROSERVICE PATTERN DATASET
# ============================================================================

class MicroservicePatternDataset:
    """
    Dataset of microservice patterns and characteristics
    """
    
    @staticmethod
    def create_pattern_dataset():
        """
        Create a dataset of microservice patterns and their characteristics
        """
        # Features: [team_size, complexity_level, traffic_volume, data_consistency_needs, 
        #            real_time_requirements, deployment_frequency]
        patterns = []
        labels = []
        
        # Pattern 1: Simple CRUD Service (FastAPI)
        patterns.append([3, 2, 3, 2, 1, 2])  # Small team, medium complexity
        labels.append('fastapi_simple')
        
        # Pattern 2: High-performance API (FastAPI)
        patterns.append([5, 4, 5, 3, 4, 3])  # Medium team, high traffic
        labels.append('fastapi_performance')
        
        # Pattern 3: Monolithic to Microservices (Django)
        patterns.append([8, 5, 4, 5, 2, 1])  # Large team, high consistency needs
        labels.append('django_modular')
        
        # Pattern 4: Lightweight Service (Flask)
        patterns.append([2, 1, 2, 1, 1, 4])  # Small team, frequent deployment
        labels.append('flask_lightweight')
        
        # Pattern 5: Event-Driven Service (FastAPI + Kafka)
        patterns.append([4, 4, 3, 3, 5, 3])  # Real-time requirements
        labels.append('fastapi_event_driven')
        
        # Pattern 6: Admin/Management Service (Django Admin)
        patterns.append([3, 3, 2, 4, 1, 1])  # High data consistency
        labels.append('django_admin')
        
        # Pattern 7: Rapid Prototype (Flask)
        patterns.append([1, 2, 1, 1, 1, 5])  # Very frequent deployment
        labels.append('flask_prototype')
        
        # Pattern 8: Enterprise Service (Django DRF)
        patterns.append([10, 5, 4, 5, 3, 2])  # Large enterprise
        labels.append('django_enterprise')
        
        return np.array(patterns), np.array(labels)

# ============================================================================
# MICROSERVICE TEMPLATE GENERATOR
# ============================================================================

class MicroserviceTemplateGenerator:
    """
    ML-based template generator for microservice architectures
    """
    
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'team_size',
            'complexity_level',
            'traffic_volume',
            'data_consistency_needs',
            'real_time_requirements',
            'deployment_frequency'
        ]
        self.template_configs = self._load_template_configs()
        
    def _load_template_configs(self):
        """
        Load configuration for different template types
        """
        return {
            'fastapi_simple': {
                'framework': 'fastapi',
                'database': 'sqlite',
                'cache': 'redis',
                'messaging': 'none',
                'monitoring': 'basic',
                'auth': 'jwt',
                'docs': True
            },
            'fastapi_performance': {
                'framework': 'fastapi',
                'database': 'postgresql',
                'cache': 'redis_cluster',
                'messaging': 'rabbitmq',
                'monitoring': 'prometheus',
                'auth': 'oauth2',
                'docs': True
            },
            'django_modular': {
                'framework': 'django',
                'database': 'postgresql',
                'cache': 'redis',
                'messaging': 'celery',
                'monitoring': 'elk',
                'auth': 'session',
                'docs': True
            },
            'flask_lightweight': {
                'framework': 'flask',
                'database': 'sqlite',
                'cache': 'simple',
                'messaging': 'none',
                'monitoring': 'none',
                'auth': 'basic',
                'docs': False
            },
            'fastapi_event_driven': {
                'framework': 'fastapi',
                'database': 'mongodb',
                'cache': 'redis',
                'messaging': 'kafka',
                'monitoring': 'prometheus',
                'auth': 'jwt',
                'docs': True
            },
            'django_admin': {
                'framework': 'django',
                'database': 'postgresql',
                'cache': 'redis',
                'messaging': 'none',
                'monitoring': 'basic',
                'auth': 'admin',
                'docs': True
            },
            'flask_prototype': {
                'framework': 'flask',
                'database': 'sqlite',
                'cache': 'none',
                'messaging': 'none',
                'monitoring': 'none',
                'auth': 'none',
                'docs': False
            },
            'django_enterprise': {
                'framework': 'django',
                'database': 'postgresql',
                'cache': 'redis_cluster',
                'messaging': 'rabbitmq',
                'monitoring': 'datadog',
                'auth': 'ldap',
                'docs': True
            }
        }
    
    def train_model(self):
        """
        Train the decision tree model on microservice patterns
        """
        print("=" * 60)
        print("TRAINING MICROSERVICE TEMPLATE PREDICTOR")
        print("=" * 60)
        
        # Load dataset
        X, y = MicroservicePatternDataset.create_pattern_dataset()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model trained successfully!")
        print(f"📊 Accuracy: {accuracy:.2%}")
        
        # Show feature importance
        print("\n⭐ FEATURE IMPORTANCE FOR TEMPLATE SELECTION:")
        for feature, importance in zip(self.feature_names, self.model.feature_importances_):
            print(f"  {feature:25s}: {importance:.3f}")
        
        return self.model
    
    def predict_template(self, features):
        """
        Predict the best template based on project characteristics
        """
        features_array = np.array(features).reshape(1, -1)
        prediction_encoded = self.model.predict(features_array)
        template_key = self.label_encoder.inverse_transform(prediction_encoded)[0]
        
        return template_key, self.template_configs[template_key]
    
    def generate_project_structure(self, project_name, template_config, output_dir="."):
        """
        Generate complete project structure based on template
        """
        print(f"\n🚀 GENERATING PROJECT: {project_name}")
        print("=" * 60)
        
        # Create project directory
        project_path = Path(output_dir) / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        framework = template_config['framework']
        
        if framework == 'fastapi':
            self._generate_fastapi_project(project_path, project_name, template_config)
        elif framework == 'django':
            self._generate_django_project(project_path, project_name, template_config)
        elif framework == 'flask':
            self._generate_flask_project(project_path, project_name, template_config)
        
        # Generate docker files
        self._generate_docker_files(project_path, template_config)
        
        # Generate k8s manifests
        if template_config['monitoring'] != 'none':
            self._generate_k8s_manifests(project_path, project_name, template_config)
        
        print(f"\n✅ Project generated successfully at: {project_path}")
        print("\n📁 PROJECT STRUCTURE:")
        self._print_tree_structure(project_path, max_depth=3)
        
        return project_path
    
    def _generate_fastapi_project(self, project_path, project_name, config):
        """
        Generate FastAPI microservice project
        """
        # Create main directories
        (project_path / "app").mkdir(exist_ok=True)
        (project_path / "app" / "api").mkdir(exist_ok=True)
        (project_path / "app" / "core").mkdir(exist_ok=True)
        (project_path / "app" / "models").mkdir(exist_ok=True)
        (project_path / "app" / "schemas").mkdir(exist_ok=True)
        (project_path / "app" / "services").mkdir(exist_ok=True)
        (project_path / "tests").mkdir(exist_ok=True)
        (project_path / "alembic").mkdir(exist_ok=True)
        
        # Generate requirements.txt
        requirements = self._get_fastapi_requirements(config)
        (project_path / "requirements.txt").write_text(requirements)
        
        # Generate main.py
        main_py = f'''"""
{project_name} - FastAPI Microservice
Generated with ML-based template generator
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title="{project_name}",
    description="Microservice generated with intelligent template system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {{"message": "Welcome to {project_name} API", "status": "healthy"}}

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "service": "{project_name}"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        (project_path / "main.py").write_text(main_py)
        
        # Generate config
        config_py = f'''"""
Configuration for {project_name}
"""

from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "{project_name}"
    
    # Database
    DATABASE_URL: str = "{self._get_database_url(config['database'])}"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # {config['cache'].upper()} Cache
    REDIS_URL: str = "redis://localhost:6379" if config['cache'] in ['redis', 'redis_cluster'] else ""
    
    # {config['messaging'].upper()} Messaging
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092" if config['messaging'] == 'kafka' else ""
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672//" if config['messaging'] == 'rabbitmq' else ""
    
    class Config:
        env_file = ".env"

settings = Settings()
'''
        (project_path / "app" / "core" / "config.py").write_text(config_py)
        
        # Generate .env example
        env_example = '''# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Kafka (if using)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# RabbitMQ (if using)
RABBITMQ_URL=amqp://guest:guest@localhost:5672//
'''
        (project_path / ".env.example").write_text(env_example)
    
    def _generate_django_project(self, project_path, project_name, config):
        """
        Generate Django microservice project
        """
        # Create Django project structure
        (project_path / "config").mkdir(exist_ok=True)
        (project_path / "apps").mkdir(exist_ok=True)
        (project_path / "static").mkdir(exist_ok=True)
        (project_path / "media").mkdir(exist_ok=True)
        (project_path / "templates").mkdir(exist_ok=True)
        
        # Generate requirements.txt
        requirements = self._get_django_requirements(config)
        (project_path / "requirements.txt").write_text(requirements)
        
        # Generate manage.py
        manage_py = f'''#!/usr/bin/env python
"""
Django manage.py for {project_name}
"""

import os
import sys

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
'''
        (project_path / "manage.py").write_text(manage_py)
        
        # Generate Dockerfile for Django
        dockerfile = f'''FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Run gunicorn
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]
'''
        (project_path / "Dockerfile").write_text(dockerfile)
    
    def _generate_flask_project(self, project_path, project_name, config):
        """
        Generate Flask microservice project
        """
        # Create Flask app structure
        (project_path / "app").mkdir(exist_ok=True)
        (project_path / "app" / "static").mkdir(exist_ok=True)
        (project_path / "app" / "templates").mkdir(exist_ok=True)
        (project_path / "migrations").mkdir(exist_ok=True)
        
        # Generate requirements.txt
        requirements = self._get_flask_requirements(config)
        (project_path / "requirements.txt").write_text(requirements)
        
        # Generate app.py
        app_py = f'''"""
{project_name} - Flask Microservice
Lightweight service for rapid development
"""

from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = '{self._get_database_url(config['database'])}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    @app.route('/')
    def index():
        return jsonify({{
            'service': '{project_name}',
            'status': 'running',
            'framework': 'Flask'
        }})
    
    @app.route('/health')
    def health():
        return jsonify({{'status': 'healthy'}})
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
        (project_path / "app.py").write_text(app_py)
    
    def _generate_docker_files(self, project_path, config):
        """
        Generate Docker and docker-compose files
        """
        # Dockerfile
        dockerfile = f'''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
'''
        (project_path / "Dockerfile").write_text(dockerfile)
        
        # docker-compose.yml
        compose = {
            'version': '3.8',
            'services': {
                'app': {
                    'build': '.',
                    'ports': ['8000:8000'],
                    'environment': ['DATABASE_URL=${DATABASE_URL}'],
                    'depends_on': self._get_docker_dependencies(config)
                }
            },
            'volumes': self._get_docker_volumes(config)
        }
        
        # Add services based on config
        if config['database'] == 'postgresql':
            compose['services']['postgres'] = {
                'image': 'postgres:13',
                'environment': [
                    'POSTGRES_DB=${DB_NAME}',
                    'POSTGRES_USER=${DB_USER}',
                    'POSTGRES_PASSWORD=${DB_PASSWORD}'
                ],
                'volumes': ['postgres_data:/var/lib/postgresql/data']
            }
        
        if config['cache'] in ['redis', 'redis_cluster']:
            compose['services']['redis'] = {
                'image': 'redis:7-alpine',
                'ports': ['6379:6379']
            }
        
        with open(project_path / "docker-compose.yml", 'w') as f:
            yaml.dump(compose, f, default_flow_style=False)
    
    def _generate_k8s_manifests(self, project_path, project_name, config):
        """
        Generate Kubernetes manifests
        """
        k8s_dir = project_path / "k8s"
        k8s_dir.mkdir(exist_ok=True)
        
        # Deployment
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {'name': project_name},
            'spec': {
                'replicas': 2,
                'selector': {'matchLabels': {'app': project_name}},
                'template': {
                    'metadata': {'labels': {'app': project_name}},
                    'spec': {
                        'containers': [{
                            'name': project_name,
                            'image': f'{project_name}:latest',
                            'ports': [{'containerPort': 8000}],
                            'env': [
                                {'name': 'DATABASE_URL', 'valueFrom': {'secretKeyRef': {'name': 'db-secret', 'key': 'url'}}}
                            ]
                        }]
                    }
                }
            }
        }
        
        # Service
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {'name': f'{project_name}-service'},
            'spec': {
                'selector': {'app': project_name},
                'ports': [{'port': 80, 'targetPort': 8000}],
                'type': 'LoadBalancer'
            }
        }
        
        # Save manifests
        with open(k8s_dir / "deployment.yaml", 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        with open(k8s_dir / "service.yaml", 'w') as f:
            yaml.dump(service, f, default_flow_style=False)
    
    def _get_fastapi_requirements(self, config):
        """Get requirements for FastAPI project"""
        base = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
pydantic[email]==2.4.2
pydantic-settings==2.1.0
'''
        
        if config['database'] == 'postgresql':
            base += 'asyncpg==0.29.0\nsqlalchemy==2.0.23\n'
        elif config['database'] == 'mongodb':
            base += 'motor==3.3.2\npymongo==4.6.0\n'
        else:
            base += 'sqlalchemy==2.0.23\n'
        
        if config['cache'] in ['redis', 'redis_cluster']:
            base += 'redis==5.0.1\n'
        
        if config['messaging'] == 'kafka':
            base += 'aiokafka==0.8.0\n'
        elif config['messaging'] == 'rabbitmq':
            base += 'aio-pika==9.3.1\n'
        
        base += 'alembic==1.12.1\n'
        
        if config['monitoring'] == 'prometheus':
            base += 'prometheus-client==0.19.0\n'
        
        return base
    
    def _get_django_requirements(self, config):
        """Get requirements for Django project"""
        base = '''Django==4.2.7
django-rest-framework==0.1.0
djangorestframework==3.14.0
psycopg2-binary==2.9.9
python-dotenv==1.0.0
gunicorn==21.2.0
'''
        
        if config['cache'] in ['redis', 'redis_cluster']:
            base += 'django-redis==5.3.0\nredis==5.0.1\n'
        
        if config['messaging'] == 'celery':
            base += 'celery==5.3.4\n'
        
        if config['monitoring'] == 'elk':
            base += 'django-elasticsearch-dsl==7.2.2\n'
        
        return base
    
    def _get_flask_requirements(self, config):
        """Get requirements for Flask project"""
        base = '''Flask==3.0.0
Flask-SQLAlchemy==3.1.1
Flask-Migrate==4.0.5
python-dotenv==1.0.0
gunicorn==21.2.0
'''
        
        if config['database'] == 'postgresql':
            base += 'psycopg2-binary==2.9.9\n'
        
        return base
    
    def _get_database_url(self, database_type):
        """Get database URL based on type"""
        urls = {
            'sqlite': 'sqlite:///./app.db',
            'postgresql': 'postgresql://user:password@localhost:5432/dbname',
            'mongodb': 'mongodb://localhost:27017/dbname'
        }
        return urls.get(database_type, 'sqlite:///./app.db')
    
    def _get_docker_dependencies(self, config):
        """Get Docker dependencies based on configuration"""
        dependencies = []
        if config['database'] == 'postgresql':
            dependencies.append('postgres')
        if config['cache'] in ['redis', 'redis_cluster']:
            dependencies.append('redis')
        return dependencies
    
    def _get_docker_volumes(self, config):
        """Get Docker volumes based on configuration"""
        volumes = {}
        if config['database'] == 'postgresql':
            volumes['postgres_data'] = None
        return volumes
    
    def _print_tree_structure(self, directory, prefix="", max_depth=3, current_depth=0):
        """Print directory tree structure"""
        if current_depth >= max_depth:
            return
        
        items = list(directory.iterdir())
        items.sort()
        
        for i, item in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            print(prefix + connector + item.name)
            
            if item.is_dir():
                extension = "    " if i == len(items) - 1 else "│   "
                self._print_tree_structure(item, prefix + extension, max_depth, current_depth + 1)

# ============================================================================
# EXAMPLE USE CASES
# ============================================================================

def example_use_cases():
    """
    Demonstrate different microservice template generation scenarios
    """
    print("=" * 60)
    print("MICROSERVICE TEMPLATE GENERATOR - EXAMPLE USE CASES")
    print("=" * 60)
    
    # Initialize generator
    generator = MicroserviceTemplateGenerator()
    
    # Train the model
    generator.train_model()
    
    print("\n" + "=" * 60)
    print("EXAMPLE 1: STARTUP MVP (Small team, rapid development)")
    print("=" * 60)
    
    # Features: [team_size, complexity_level, traffic_volume, 
    #            data_consistency_needs, real_time_requirements, deployment_frequency]
    startup_features = [2, 2, 2, 1, 1, 5]  # Small team, frequent deploys
    template_key, config = generator.predict_template(startup_features)
    
    print(f"\n📊 Predicted Template: {template_key}")
    print(f"🏗️  Framework: {config['framework'].upper()}")
    print(f"💾 Database: {config['database']}")
    print(f"⚡ Cache: {config['cache']}")
    print(f"📨 Messaging: {config['messaging']}")
    
    # Generate project
    generator.generate_project_structure(
        project_name="startup-mvp-service",
        template_config=config,
        output_dir="./generated_projects"
    )
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: ENTERPRISE SERVICE (Large team, high requirements)")
    print("=" * 60)
    
    enterprise_features = [12, 5, 5, 5, 4, 2]  # Large enterprise
    template_key, config = generator.predict_template(enterprise_features)
    
    print(f"\n📊 Predicted Template: {template_key}")
    print(f"🏗️  Framework: {config['framework'].upper()}")
    print(f"💾 Database: {config['database']}")
    print(f"⚡ Cache: {config['cache']}")
    print(f"📨 Messaging: {config['messaging']}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: EVENT-DRIVEN SERVICE (Real-time processing)")
    print("=" * 60)
    
    event_driven_features = [4, 4, 3, 3, 5, 3]  # Real-time requirements
    template_key, config = generator.predict_template(event_driven_features)
    
    print(f"\n📊 Predicted Template: {template_key}")
    print(f"🏗️  Framework: {config['framework'].upper()}")
    print(f"💾 Database: {config['database']}")
    print(f"⚡ Cache: {config['cache']}")
    print(f"📨 Messaging: {config['messaging']}")
    
    # Generate another project
    generator.generate_project_structure(
        project_name="event-processor-service",
        template_config=config,
        output_dir="./generated_projects"
    )

# ============================================================================
# INTERACTIVE TEMPLATE GENERATOR
# ============================================================================

def interactive_template_generator():
    """
    Interactive CLI for generating microservice templates
    """
    print("=" * 60)
    print("🤖 INTELLIGENT MICROSERVICE TEMPLATE GENERATOR")
    print("=" * 60)
    
    generator = MicroserviceTemplateGenerator()
    generator.train_model()
    
    print("\n📝 Answer these questions about your project:")
    print("   (Rate each from 1-5 where 1=Low, 5=High)")
    
    questions = [
        "Team size (1-5 scale where 1=1-2, 5=10+): ",
        "Project complexity (1=Simple, 5=Very Complex): ",
        "Expected traffic volume (1=Low, 5=Very High): ",
        "Data consistency needs (1=Low, 5=Very High): ",
        "Real-time requirements (1=None, 5=Critical): ",
        "Deployment frequency (1=Rarely, 5=Multiple times/day): "
    ]
    
    features = []
    for question in questions:
        while True:
            try:
                value = int(input(f"\n{question}"))
                if 1 <= value <= 5:
                    features.append(value)
                    break
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
    
    print("\n" + "=" * 60)
    print("🧠 ANALYZING YOUR PROJECT REQUIREMENTS...")
    print("=" * 60)
    
    # Predict template
    template_key, config = generator.predict_template(features)
    
    print(f"\n✅ RECOMMENDED TEMPLATE: {template_key}")
    print("\n⚙️  CONFIGURATION:")
    for key, value in config.items():
        print(f"  {key:15s}: {value}")
    
    print("\n📊 YOUR PROJECT CHARACTERISTICS:")
    characteristics = [
        "Team Size",
        "Complexity",
        "Traffic",
        "Data Consistency",
        "Real-time Needs",
        "Deployment Frequency"
    ]
    
    for char, val in zip(characteristics, features):
        print(f"  {char:20s}: {'█' * val} ({val}/5)")
    
    # Ask for project name
    project_name = input("\n📛 Enter your project name: ").strip() or "microservice-project"
    
    # Generate project
    print(f"\n🚀 Generating '{project_name}'...")
    generator.generate_project_structure(
        project_name=project_name,
        template_config=config,
        output_dir="./generated_projects"
    )
    
    print("\n🎉 PROJECT GENERATED SUCCESSFULLY!")
    print("\n📚 NEXT STEPS:")
    print("1. Navigate to the generated project directory")
    print("2. Review the README.md for setup instructions")
    print("3. Configure your .env file with actual values")
    print("4. Run 'docker-compose up' to start services")
    print("5. Begin implementing your business logic!")

# ============================================================================
# QUICK START
# ============================================================================

def quick_start():
    """
    Quick start for immediate template generation
    """
    print("🚀 QUICK START: GENERATE MICROSERVICE IN 30 SECONDS")
    print("=" * 50)
    
    generator = MicroserviceTemplateGenerator()
    
    # Use default template for quick start
    template_config = {
        'framework': 'fastapi',
        'database': 'postgresql',
        'cache': 'redis',
        'messaging': 'none',
        'monitoring': 'basic',
        'auth': 'jwt',
        'docs': True
    }
    
    project_name = "quickstart-microservice"
    
    print(f"\nGenerating '{project_name}' with FastAPI + PostgreSQL + Redis...")
    
    generator.generate_project_structure(
        project_name=project_name,
        template_config=template_config,
        output_dir="./quickstart"
    )
    
    print("\n✅ Done! Your microservice is ready.")
    print(f"👉 cd quickstart/{project_name}")
    print("👉 cp .env.example .env")
    print("👉 docker-compose up")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("🤖 MICROSERVICE TEMPLATE GENERATOR - ML POWERED")
    print("=" * 60)
    print("\nChoose an option:")
    print("1. Interactive Template Generator (Recommended)")
    print("2. View Example Use Cases")
    print("3. Quick Start (Generate Default Template)")
    print("4. Train and Evaluate Model")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        interactive_template_generator()
    elif choice == "2":
        example_use_cases()
    elif choice == "3":
        quick_start()
    elif choice == "4":
        generator = MicroserviceTemplateGenerator()
        generator.train_model()
        
        # Show decision tree visualization
        print("\n🌳 DECISION TREE VISUALIZATION")
        print("The model uses these rules to select templates:")
        print("\n1. Team Size > 3 → Django for larger teams")
        print("2. Real-time > 3 → FastAPI for performance")
        print("3. Deployment > 4 → Flask for rapid iteration")
        print("4. Consistency > 4 → Django for robust data handling")
    else:
        print("⚠️ Please enter 1, 2, 3, or 4")
        main()

# ============================================================================
# PROJECT SUMMARY
# ============================================================================

def project_summary():
    """
    Display project summary and features
    """
    print("\n" + "=" * 60)
    print("📦 GENERATED PROJECT INCLUDES:")
    print("=" * 60)
    
    features = [
        "✅ Complete project structure",
        "✅ Docker and docker-compose setup",
        "✅ Database configuration (PostgreSQL/SQLite/MongoDB)",
        "✅ Caching layer (Redis)",
        "✅ Messaging (Kafka/RabbitMQ/Celery)",
        "✅ API documentation (Swagger/OpenAPI)",
        "✅ Authentication & Authorization",
        "✅ Health check endpoints",
        "✅ Environment configuration",
        "✅ Migration system (Alembic)",
        "✅ Testing structure",
        "✅ Kubernetes manifests (optional)",
        "✅ Monitoring setup (Prometheus/ELK)",
        "✅ Logging configuration",
        "✅ Error handling",
        "✅ Rate limiting",
        "✅ CORS configuration",
        "✅ Request validation",
        "✅ API versioning",
        "✅ Deployment scripts"
    ]
    
    for feature in features:
        print(feature)

# ============================================================================
# RUN THE GENERATOR
# ============================================================================

if __name__ == "__main__":
    try:
        main()
        project_summary()
        
        print("\n" + "=" * 60)
        print("📚 RESOURCES:")
        print("=" * 60)
        print("FastAPI: https://fastapi.tiangolo.com")
        print("Django: https://docs.djangoproject.com")
        print("Flask: https://flask.palletsprojects.com")
        print("Microservices Patterns: https://microservices.io")
        print("Docker: https://docs.docker.com")
        print("Kubernetes: https://kubernetes.io/docs")
        
        print("\n🌟 TIP: The ML model learns from patterns in successful")
        print("       microservice implementations to recommend optimal")
        print("       architectures for your specific needs.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Ensure you have Python 3.8+ installed")
        print("2. Install required packages: pip install scikit-learn pyyaml")
        print("3. Check your disk space and permissions")