# Deployment Guide 🚀

Comprehensive guide for deploying LangChain Playbook applications in various environments - from local development to production cloud deployments.

## 📋 Deployment Overview

### Deployment Scenarios
- **📖 Learning Environment**: Local setup for education
- **🔧 Development**: Team development and testing
- **🌐 Production**: Live applications and services
- **☁️ Cloud**: Scalable cloud deployments
- **🐳 Containerized**: Docker and Kubernetes

---

## 🏠 Local Development Deployment

### Basic Setup
```bash
# Clone and setup
git clone https://github.com/your-org/Langchain-Playbook.git
cd Langchain-Playbook

# Create environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run examples
python basics/01_getting_started/hello_langchain.py
```

### Jupyter Development Server
```bash
# Install Jupyter
pip install jupyter

# Start Jupyter server
jupyter notebook --ip=127.0.0.1 --port=8888

# Or JupyterLab
pip install jupyterlab
jupyter lab --ip=127.0.0.1 --port=8888
```

### API Development Server
```bash
# Run API service locally
cd projects/api_service
python api_service.py run

# Or with uvicorn
uvicorn api_service:app --reload --host 127.0.0.1 --port 8000
```

---

## 🏢 Production Deployment

### Production Checklist
- [ ] **Security**: API keys secured, no hardcoded secrets
- [ ] **Performance**: Connection pooling, caching enabled
- [ ] **Monitoring**: Logging, metrics, health checks
- [ ] **Scalability**: Load balancing, auto-scaling
- [ ] **Reliability**: Error handling, graceful degradation
- [ ] **Backup**: Data backup and recovery plans

### Environment Variables
```bash
# Production environment variables
export ENVIRONMENT=production
export DEBUG=false
export LOG_LEVEL=INFO

# API Configuration
export OPENAI_API_KEY=your_production_key
export API_RATE_LIMIT=1000
export API_TIMEOUT=30

# Database Configuration (if applicable)
export DATABASE_URL=postgresql://user:pass@host:5432/db
export REDIS_URL=redis://host:6379

# Security
export SECRET_KEY=your_secret_key
export ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

### Production Requirements
```txt
# requirements-prod.txt
# Core dependencies
langchain>=0.2.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0

# Production additions
gunicorn>=21.2.0
redis>=5.0.0
psycopg2-binary>=2.9.0
prometheus-client>=0.19.0
structlog>=23.1.0

# Monitoring
sentry-sdk>=1.38.0
newrelic
datadog

# Security
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
```

### Production Configuration
```python
# config/production.py
import os
from pathlib import Path

class ProductionConfig:
    # Environment
    ENVIRONMENT = "production"
    DEBUG = False
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = int(os.getenv("PORT", 8000))
    API_WORKERS = int(os.getenv("WEB_CONCURRENCY", 4))
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("API_RATE_LIMIT", 1000))
    RATE_LIMIT_WINDOW = 3600  # 1 hour
    
    # Timeouts
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", 30))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "json"
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    REDIS_URL = os.getenv("REDIS_URL")
    
    # Monitoring
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    NEW_RELIC_LICENSE_KEY = os.getenv("NEW_RELIC_LICENSE_KEY")
```

---

## 🐳 Docker Deployment

### Basic Dockerfile
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "projects.api_service.api_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-stage Production Dockerfile
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Update PATH
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "projects.api_service.api_service:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Docker Compose Production
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  langchain-api:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/langchain_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    restart: unless-stopped
    networks:
      - langchain-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - langchain-api
    restart: unless-stopped
    networks:
      - langchain-network

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=langchain_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - langchain-network

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    networks:
      - langchain-network

volumes:
  postgres_data:

networks:
  langchain-network:
    driver: bridge
```

### Nginx Configuration
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream langchain_api {
        server langchain-api:8000;
    }

    server {
        listen 80;
        server_name yourdomain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://langchain_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://langchain_api/health;
            access_log off;
        }
    }
}
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### ECS with Fargate
```yaml
# aws-task-definition.json
{
  "family": "langchain-playbook",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "langchain-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/langchain-playbook:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/langchain-playbook",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Lambda Deployment
```python
# lambda_handler.py
import json
from mangum import Mangum
from projects.api_service.api_service import app

# Lambda handler
handler = Mangum(app)

def lambda_handler(event, context):
    return handler(event, context)
```

```yaml
# serverless.yml
service: langchain-playbook

provider:
  name: aws
  runtime: python3.11
  region: us-west-2
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
  
functions:
  api:
    handler: lambda_handler.handler
    events:
      - http:
          path: /{proxy+}
          method: any
    timeout: 30
    memorySize: 1024

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
```

### Google Cloud Platform

#### Cloud Run Deployment
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/langchain-playbook', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/langchain-playbook']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'langchain-playbook'
      - '--image'
      - 'gcr.io/$PROJECT_ID/langchain-playbook'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
```

#### App Engine Deployment
```yaml
# app.yaml
runtime: python311

env_variables:
  ENVIRONMENT: production
  OPENAI_API_KEY: your_key_here

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6

handlers:
  - url: /.*
    script: auto
```

### Azure Deployment

#### Container Instances
```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name langchain-playbook \
  --image myregistry.azurecr.io/langchain-playbook:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production \
  --secure-environment-variables OPENAI_API_KEY=your_key_here
```

#### App Service
```yaml
# azure-pipelines.yml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
  azureServiceConnectionId: 'your-service-connection'
  webAppName: 'langchain-playbook'
  
stages:
- stage: Build
  displayName: Build stage
  jobs:
  - job: BuildJob
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: Docker@2
      displayName: Build and push image
      inputs:
        containerRegistry: 'your-registry'
        repository: 'langchain-playbook'
        command: 'buildAndPush'
        Dockerfile: '**/Dockerfile'
```

---

## 🎛️ Kubernetes Deployment

### Basic Kubernetes Manifests

#### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-playbook
  labels:
    app: langchain-playbook
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langchain-playbook
  template:
    metadata:
      labels:
        app: langchain-playbook
    spec:
      containers:
      - name: langchain-api
        image: your-registry/langchain-playbook:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: langchain-playbook-service
spec:
  selector:
    app: langchain-playbook
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

#### Ingress
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: langchain-playbook-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - yourdomain.com
    secretName: langchain-tls
  rules:
  - host: yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: langchain-playbook-service
            port:
              number: 80
```

#### Secrets
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  anthropic-api-key: <base64-encoded-key>
```

### Helm Chart
```yaml
# helm/Chart.yaml
apiVersion: v2
name: langchain-playbook
description: LangChain Playbook Helm Chart
version: 0.1.0
appVersion: "1.0"

# helm/values.yaml
replicaCount: 3

image:
  repository: your-registry/langchain-playbook
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: langchain-tls
      hosts:
        - yourdomain.com

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

---

## 📊 Monitoring and Observability

### Application Monitoring
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # Increment request counter
            REQUEST_COUNT.labels(
                method=scope["method"],
                endpoint=scope["path"]
            ).inc()
            
            # Track active connections
            ACTIVE_CONNECTIONS.inc()
            
            try:
                await self.app(scope, receive, send)
            finally:
                # Record duration
                REQUEST_DURATION.observe(time.time() - start_time)
                ACTIVE_CONNECTIONS.dec()
```

### Logging Configuration
```python
# logging_config.py
import structlog
import logging.config

def configure_logging():
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=False),
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "json",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
        }
    })

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

---

## 🔒 Security Deployment

### Security Checklist
- [ ] **API Keys**: Stored in secure vaults, not in code
- [ ] **HTTPS**: TLS/SSL enabled for all endpoints
- [ ] **Authentication**: API authentication implemented
- [ ] **Rate Limiting**: Request throttling in place
- [ ] **Input Validation**: All inputs validated and sanitized
- [ ] **CORS**: Cross-origin requests properly configured
- [ ] **Secrets Management**: Production secrets in vault systems

### Security Headers
```python
# security/middleware.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

def add_security_middleware(app: FastAPI):
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Trusted Host
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )
    
    # Security Headers
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
```

---

## 📈 Scaling Strategies

### Horizontal Scaling
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langchain-playbook-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langchain-playbook
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Balancing
```nginx
# nginx-lb.conf
upstream langchain_backend {
    least_conn;
    server 10.0.1.10:8000 weight=3;
    server 10.0.1.11:8000 weight=3;
    server 10.0.1.12:8000 weight=2;
    
    # Health checks
    health_check interval=30s fails=3 passes=2;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://langchain_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

---

## 🚀 Deployment Automation

### CI/CD Pipeline (GitHub Actions)
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python tests/test_suite.py --mode full

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: |
        docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} .
    - name: Push to registry
      run: |
        echo ${{ secrets.GITHUB_TOKEN }} | docker login ${{ env.REGISTRY }} -u ${{ github.actor }} --password-stdin
        docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
    - name: Deploy to production
      run: |
        # Deploy commands here
        kubectl set image deployment/langchain-playbook langchain-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
```

### Deployment Scripts
```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Starting deployment..."

# Build and push image
docker build -t langchain-playbook:latest .
docker tag langchain-playbook:latest your-registry/langchain-playbook:$(git rev-parse --short HEAD)
docker push your-registry/langchain-playbook:$(git rev-parse --short HEAD)

# Update Kubernetes deployment
kubectl set image deployment/langchain-playbook langchain-api=your-registry/langchain-playbook:$(git rev-parse --short HEAD)

# Wait for rollout
kubectl rollout status deployment/langchain-playbook --timeout=300s

# Verify deployment
kubectl get pods -l app=langchain-playbook

echo "✅ Deployment completed successfully!"
```

---

**Ready to deploy? Choose your deployment strategy and follow the appropriate guide. Remember to test thoroughly in staging before production deployment! 🎯**