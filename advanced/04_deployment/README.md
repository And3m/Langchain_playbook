# Production Deployment 🚀

Complete guide for deploying LangChain applications to production environments with modern DevOps practices.

## 📋 Overview

This module covers:
- **Containerization**: Docker best practices for LangChain apps
- **Orchestration**: Kubernetes deployment patterns
- **CI/CD**: Automated deployment pipelines
- **Monitoring**: Observability and health checking
- **Scaling**: Auto-scaling and load balancing

## 🐳 Containerization

### Docker Best Practices

1. **Multi-stage builds** for smaller images
2. **Non-root user** for security
3. **Health checks** for reliability
4. **Layer caching** for faster builds

### Example Dockerfile
```dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.11-slim
RUN useradd --create-home app
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
USER app
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

## ☸️ Kubernetes Deployment

### Core Components

1. **Deployment**: Application pods management
2. **Service**: Internal load balancing
3. **Ingress**: External traffic routing
4. **HPA**: Horizontal pod autoscaling

### Quick Start
```bash
# Generate deployment package
python production_deployment.py

# Apply to cluster
kubectl apply -f deployment-package/k8s/

# Check status
kubectl get pods
kubectl get services
```

## 🔄 CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Deploy LangChain App
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run tests
      run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to Kubernetes
      run: kubectl apply -f k8s/
```

## 📊 Monitoring & Observability

### Health Checks
- **Liveness probe**: Container restart if unhealthy
- **Readiness probe**: Traffic routing control
- **Startup probe**: Initial startup time handling

### Metrics Collection
- Application metrics via Prometheus
- Custom LangChain metrics
- Cost and usage tracking

## ⚡ Performance & Scaling

### Auto-scaling Configuration
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

### Load Balancing
- Service-level load balancing
- Ingress controller configuration
- Regional traffic distribution

## 🔒 Security

### Best Practices
1. **Secret management**: Use Kubernetes secrets
2. **Network policies**: Control pod communication
3. **RBAC**: Role-based access control
4. **Image scanning**: Vulnerability detection

### Secret Configuration
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: langchain-secrets
data:
  openai-api-key: <base64-encoded-key>
```

## 🌍 Multi-Environment Setup

### Environment Structure
- **Development**: Local testing
- **Staging**: Pre-production validation
- **Production**: Live environment

### Configuration Management
```bash
# Deploy to staging
kubectl apply -f k8s/ --namespace=staging

# Deploy to production
kubectl apply -f k8s/ --namespace=production
```

## 📈 Cost Optimization

### Resource Management
1. **Right-sizing**: Appropriate CPU/memory limits
2. **Spot instances**: Cost-effective compute
3. **Cluster autoscaling**: Dynamic node management
4. **Resource monitoring**: Usage tracking

### Example Resource Limits
```yaml
resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 100m
    memory: 256Mi
```

## 🛠️ Troubleshooting

### Common Issues

1. **Pod not starting**
   ```bash
   kubectl describe pod <pod-name>
   kubectl logs <pod-name>
   ```

2. **Service unreachable**
   ```bash
   kubectl get endpoints
   kubectl port-forward service/<service-name> 8080:80
   ```

3. **Resource limits exceeded**
   ```bash
   kubectl top pods
   kubectl describe hpa
   ```

## 📚 Advanced Patterns

### Blue-Green Deployment
```bash
# Deploy new version
kubectl apply -f k8s-green/

# Switch traffic
kubectl patch service app-service -p '{"spec":{"selector":{"version":"green"}}}'
```

### Canary Deployment
```yaml
# Istio virtual service for canary
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: app-service
        subset: v2
  - route:
    - destination:
        host: app-service
        subset: v1
      weight: 90
    - destination:
        host: app-service
        subset: v2
      weight: 10
```

## 🚀 Getting Started

1. **Run the deployment generator**:
   ```bash
   cd advanced/04_deployment
   python production_deployment.py
   ```

2. **Customize the generated files** for your environment

3. **Set up your infrastructure**:
   - Container registry
   - Kubernetes cluster
   - CI/CD platform

4. **Deploy your application**:
   ```bash
   kubectl apply -f deployment-package/k8s/
   ```

## 📖 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [CI/CD with GitHub Actions](https://docs.github.com/en/actions)

---

**Deploy LangChain Applications at Scale! 🌟**