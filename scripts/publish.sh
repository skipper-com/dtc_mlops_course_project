#!/usr/bin/env bash
echo "publishing image ${LOCAL_IMAGE_NAME} to ECR..."
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 898314662918.dkr.ecr.eu-central-1.amazonaws.com
docker tag ${LOCAL_IMAGE_NAME} 898314662918.dkr.ecr.eu-central-1.amazonaws.com/${ECR_REGISTRY}:latest
docker push 898314662918.dkr.ecr.eu-central-1.amazonaws.com/${ECR_REGISTRY}:latest
