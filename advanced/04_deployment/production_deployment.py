#!/usr/bin/env python3
"""
LangChain Production Deployment

This module demonstrates:
1. Production deployment architectures
2. Containerization with Docker
3. Kubernetes orchestration
4. CI/CD pipelines
5. Monitoring and observability

Key concepts:
- Microservices architecture
- Container orchestration
- Infrastructure as Code
- Auto-scaling and load balancing
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger


@dataclass
class DeploymentConfig:
    """Configuration for deployment setup."""
    app_name: str
    environment: str
    replicas: int = 3
    cpu_limit: str = "500m"
    memory_limit: str = "1Gi"
    port: int = 8000


class DockerfileGenerator:
    """Generate Dockerfiles for LangChain applications."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_fastapi_dockerfile(self) -> str:
        """Generate Dockerfile for FastAPI applications."""
        return """# Multi-stage build for LangChain FastAPI application
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Install runtime dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""


class KubernetesGenerator:
    """Generate Kubernetes manifests for LangChain applications."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_deployment(self, config: DeploymentConfig, image: str) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{config.app_name}-deployment",
                "namespace": config.environment,
                "labels": {"app": config.app_name}
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {"matchLabels": {"app": config.app_name}},
                "template": {
                    "metadata": {"labels": {"app": config.app_name}},
                    "spec": {
                        "containers": [{
                            "name": config.app_name,
                            "image": image,
                            "ports": [{"containerPort": config.port}],
                            "resources": {
                                "limits": {"cpu": config.cpu_limit, "memory": config.memory_limit},
                                "requests": {"cpu": "100m", "memory": "256Mi"}
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": config.port},
                                "initialDelaySeconds": 30, "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": config.port},
                                "initialDelaySeconds": 5, "periodSeconds": 5
                            },
                            "env": [{
                                "name": "OPENAI_API_KEY",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "langchain-secrets",
                                        "key": "openai-api-key"
                                    }
                                }
                            }]
                        }]
                    }
                }
            }
        }
    
    def generate_service(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Kubernetes service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.app_name}-service",
                "namespace": config.environment
            },
            "spec": {
                "selector": {"app": config.app_name},
                "ports": [{"protocol": "TCP", "port": 80, "targetPort": config.port}],
                "type": "ClusterIP"
            }
        }
    
    def generate_hpa(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{config.app_name}-hpa",
                "namespace": config.environment
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{config.app_name}-deployment"
                },
                "minReplicas": 2,
                "maxReplicas": 10,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {"type": "Utilization", "averageUtilization": 70}
                    }
                }]
            }
        }


class CICDGenerator:
    """Generate CI/CD pipeline configurations."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_github_actions(self, app_name: str) -> str:
        """Generate GitHub Actions workflow."""
        return f"""name: Build and Deploy {app_name}

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{{{ github.repository }}}}/{app_name}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: pytest tests/
      env:
        OPENAI_API_KEY: ${{{{ secrets.OPENAI_API_KEY }}}}

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
    - uses: actions/checkout@v4
    - uses: docker/setup-buildx-action@v3
    - uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    - name: Deploy to Kubernetes
      run: |
        echo "${{{{ secrets.KUBE_CONFIG }}}}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/{app_name}-deployment {app_name}=${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}:latest
        kubectl rollout status deployment/{app_name}-deployment
"""


class DeploymentOrchestrator:
    """Main deployment orchestrator."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.docker_gen = DockerfileGenerator()
        self.k8s_gen = KubernetesGenerator()
        self.cicd_gen = CICDGenerator()
    
    def create_deployment_package(self, config: DeploymentConfig, 
                                output_dir: Path = Path("./deployment")):
        """Create complete deployment package."""
        output_dir.mkdir(exist_ok=True)
        
        # Generate Dockerfile
        dockerfile_content = self.docker_gen.generate_fastapi_dockerfile()
        (output_dir / "Dockerfile").write_text(dockerfile_content)
        
        # Generate Kubernetes manifests
        k8s_dir = output_dir / "k8s"
        k8s_dir.mkdir(exist_ok=True)
        
        deployment = self.k8s_gen.generate_deployment(config, f"{config.app_name}:latest")
        service = self.k8s_gen.generate_service(config)
        hpa = self.k8s_gen.generate_hpa(config)
        
        with open(k8s_dir / "deployment.yaml", "w") as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        with open(k8s_dir / "service.yaml", "w") as f:
            yaml.dump(service, f, default_flow_style=False)
        
        with open(k8s_dir / "hpa.yaml", "w") as f:
            yaml.dump(hpa, f, default_flow_style=False)
        
        # Generate CI/CD workflow
        cicd_dir = output_dir / ".github" / "workflows"
        cicd_dir.mkdir(parents=True, exist_ok=True)
        
        github_workflow = self.cicd_gen.generate_github_actions(config.app_name)
        (cicd_dir / "deploy.yml").write_text(github_workflow)
        
        self.logger.info(f"Deployment package created in {output_dir}")
        return output_dir


def create_production_deployment_guide() -> str:
    """Create production deployment guide."""
    return """# Production Deployment Guide

## 1. Prerequisites
- Docker installed
- Kubernetes cluster access
- CI/CD platform (GitHub Actions, GitLab CI)
- Container registry access

## 2. Deployment Steps

### Step 1: Containerize Application
```bash
# Build Docker image
docker build -t your-app:latest .

# Test locally
docker run -p 8000:8000 your-app:latest
```

### Step 2: Deploy to Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

### Step 3: Configure Monitoring
```bash
# Install monitoring stack
helm install prometheus prometheus-community/kube-prometheus-stack
```

## 3. Best Practices
- Use multi-stage Docker builds
- Implement health checks
- Configure resource limits
- Set up auto-scaling
- Monitor application metrics
"""


def main():
    """Demonstrate deployment patterns."""
    setup_logging()
    logger = get_logger("Deployment-Demo")
    
    print("🚀 LangChain Production Deployment Demo")
    print("=" * 50)
    
    # Create deployment configuration
    config = DeploymentConfig(
        app_name="langchain-api",
        environment="production",
        replicas=3,
        cpu_limit="500m",
        memory_limit="1Gi",
        port=8000
    )
    
    # Create deployment orchestrator
    orchestrator = DeploymentOrchestrator()
    
    # Generate deployment package
    output_dir = Path(__file__).parent / "deployment-package"
    package_dir = orchestrator.create_deployment_package(config, output_dir)
    
    print(f"📦 Deployment package created: {package_dir}")
    print("\nGenerated files:")
    for file_path in package_dir.rglob("*"):
        if file_path.is_file():
            print(f"  {file_path.relative_to(package_dir)}")
    
    # Create deployment guide
    guide_content = create_production_deployment_guide()
    guide_file = package_dir / "DEPLOYMENT_GUIDE.md"
    guide_file.write_text(guide_content)
    
    print(f"\n📖 Deployment guide: {guide_file}")
    print("\n✅ Production deployment setup completed!")
    print("\nNext steps:")
    print("1. Customize the generated manifests for your environment")
    print("2. Set up your container registry")
    print("3. Configure your Kubernetes cluster")
    print("4. Set up CI/CD secrets and variables")


if __name__ == "__main__":
    main()