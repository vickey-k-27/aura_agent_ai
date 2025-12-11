#!/bin/bash
# ==============================================================================
# PolicyVoice - Deploy to Cloud Run
# ==============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-wide-decoder-477801-u8}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
SERVICE_NAME="policyvoice-agent"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=============================================="
echo "PolicyVoice Cloud Run Deployment"
echo "=============================================="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "Image:    ${IMAGE_NAME}"
echo "=============================================="

# Step 1: Enable required APIs
echo ""
echo "📦 Step 1: Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    firestore.googleapis.com \
    bigquery.googleapis.com \
    discoveryengine.googleapis.com \
    --project=${PROJECT_ID}

# Step 2: Build and push Docker image
echo ""
echo "🐳 Step 2: Building Docker image..."
gcloud builds submit . \
    --tag ${IMAGE_NAME} \
    --project=${PROJECT_ID} \
    --timeout=15m

# Step 3: Deploy to Cloud Run
echo ""
echo "🚀 Step 3: Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 60 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},GCP_PROJECT_ID=${PROJECT_ID},GCP_LOCATION=${REGION},VERTEX_SEARCH_DATASTORE_ID=lumen-policy-store_1765096191975,GOOGLE_GENAI_USE_VERTEXAI=TRUE"

# Step 4: Get the service URL
echo ""
echo "🔗 Step 4: Getting service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --format 'value(status.url)')

echo ""
echo "=============================================="
echo "✅ Deployment Complete!"
echo "=============================================="
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Webhook URL for Dialogflow CX:"
echo "  ${SERVICE_URL}/webhook"
echo ""
echo "Health check:"
echo "  curl ${SERVICE_URL}/health"
echo ""
echo "Test webhook manually:"
echo '  curl -X POST ${SERVICE_URL}/webhook \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"text": "What does home insurance cover?", "sessionInfo": {"session": "test-123"}}'"'"
echo ""
echo "=============================================="
echo "Next Steps:"
echo "=============================================="
echo "1. Go to Dialogflow CX Console"
echo "2. Open your agent"
echo "3. Go to Manage > Webhooks"
echo "4. Create webhook with URL: ${SERVICE_URL}/webhook"
echo "5. In your flow, add webhook fulfillment"
echo "=============================================="