# Build and push images
docker buildx build --platform linux/amd64 -t gcr.io/artisk-photo/api-service:v1 -f Dockerfile.api .

docker push gcr.io/artisk-photo/api-service:v1

docker buildx build --platform linux/amd64 -t gcr.io/artisk-photo/worker-service:v1 -f Dockerfile.worker .

docker push gcr.io/artisk-photo/worker-service:v1`

# Set new image in deployment
kubectl set image deployment/api-service api=gcr.io/artisk-photo/api-service:v1

kubectl set image deployment/worker-service worker=gcr.io/artisk-photo/worker-service:v1

# Restart deployments
kubectl rollout restart deployment/api-service

kubectl rollout restart deployment/worker-service

# Apply deployments
kubectl apply -f api-deployment.yaml

kubectl apply -f worker-deployment.yaml

kubectl apply -f redis-deployment.yaml

# Set new image in deployment
kubectl set image deployment/api-service api=gcr.io/artisk-photo/api-service:v1

kubectl set image deployment/worker-service worker=gcr.io/artisk-photo/worker-service:v1

# Restart deployments
kubectl rollout restart deployment/api-service

kubectl rollout restart deployment/worker-service