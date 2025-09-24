#!/bin/bash
# Script to set up required GitHub secrets

echo "Setting up GitHub repository secrets..."

# GitHub CLI commands to set secrets
gh secret set DOCKER_USERNAME --body="your-docker-username"
gh secret set DOCKER_PASSWORD --body="your-docker-password"

# For AWS (if using ECS)
gh secret set AWS_ACCESS_KEY_ID --body="your-aws-access-key"
gh secret set AWS_SECRET_ACCESS_KEY --body="your-aws-secret-key"

echo "Secrets configured successfully!"