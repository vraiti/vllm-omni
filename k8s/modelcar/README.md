# vLLM-Omni ModelCar Deployment

This directory contains Kubernetes manifests for deploying the Z-Image-Turbo diffusion model using the **ModelCar deployment pattern** on OpenShift AI.

## What is ModelCar?

ModelCar is an OCI-compliant container image that packages ML model files in a standardized format. In this deployment:
- The ModelCar image (`quay.io/vraiti/z-image-turbo:v1`) contains only the model files at `/models`
- KServe's storage initializer extracts the model from the OCI image to `/mnt/models`
- The runtime environment (`docker.io/vllm/vllm-omni:0.14.0`) runs separately and serves the extracted model

This provides a standardized, portable way to deploy models across environments.

## Architecture

```
External Client
    |
    | HTTPS + Bearer Token
    v
data-science-gateway (OpenShift Service Mesh Gateway)
    |
    v
HTTPRoute (URL: /z-image-turbo)
    |
    | URL Rewrite: /z-image-turbo/* -> /*
    v
z-image-turbo-gateway-svc (ClusterIP Service)
    |
    v
z-image-turbo-predictor Pods (1-5 replicas, KEDA autoscaled)
    |
    | Runtime Container: docker.io/vllm/vllm-omni:0.14.0
    | Model Source: quay.io/vraiti/z-image-turbo:v1 (extracted to /mnt/models)
    v
vLLM-Omni Runtime serving model from /mnt/models
```

## ModelCar Image

**ModelCar Image:** `quay.io/vraiti/z-image-turbo:v1` (contains model files only)

**Runtime Image:** `docker.io/vllm/vllm-omni:0.14.0`

**Model Location:** `/models` in the ModelCar image, extracted to `/mnt/models` by KServe storage initializer

## Components

### Core Deployment
- `00-namespace.yaml` - omni-demo namespace
- `01-servingruntime.yaml` - ServingRuntime using the ModelCar image
- `02-inferenceservice.yaml` - InferenceService with `storageUri: oci://...`

### KEDA Autoscaling
- `03-keda-rbac.yaml` - RBAC for KEDA to query Prometheus
- `04-keda-auth.yaml` - TriggerAuthentication for Thanos Querier
- `05-scaledobject.yaml` - Autoscaling based on HTTP request rate

### External Access (Gateway)
- `06-gateway-service.yaml` - ClusterIP service for Gateway routing
- `07-referencegrant.yaml` - Allows HTTPRoute to reference service across namespaces
- `08-httproute.yaml` - Gateway route configuration (deployed in openshift-ingress namespace)

## Prerequisites

1. **OpenShift Cluster** with:
   - OpenShift AI / RHODS installed
   - KServe configured
   - KEDA operator installed
   - Service Mesh with data-science-gateway configured
   - GPU nodes available

2. **Container Image Access**:
   ```bash
   podman pull quay.io/vraiti/z-image-turbo:v1
   ```

3. **API Token**:
   ```bash
   export NERC_API_TOKEN="<your-token>"
   ```

## Deployment

```bash
# Deploy all resources
oc apply -k .

# Deploy HTTPRoute (in openshift-ingress namespace, separate from kustomization)
oc apply -f 08-httproute.yaml

# Configure KEDA authentication
SA_TOKEN=$(oc create token keda-prometheus-sa -n omni-demo --duration=8760h)
CA_CERT=$(oc get configmap -n openshift-service-ca kube-root-ca.crt -o jsonpath='{.data.ca\.crt}')

oc patch secret keda-prometheus-secret -n omni-demo \
  --type merge \
  -p "{\"stringData\":{\"bearerToken\":\"$SA_TOKEN\",\"ca\":\"$CA_CERT\"}}"
```

**Note:** The HTTPRoute must be deployed separately because it lives in the `openshift-ingress` namespace, while kustomization.yaml targets `omni-demo`.

## Access URL

**Gateway Endpoint:**
```
https://data-science-gateway.apps.ocp-beta-test.nerc.mghpcc.org/z-image-turbo
```

**Endpoints:**
- Health: `/z-image-turbo/health`
- Image generation: `/z-image-turbo/v1/images/generations`
- Models: `/z-image-turbo/v1/models`

## Testing

### Health Check
```bash
curl -k https://data-science-gateway.apps.ocp-beta-test.nerc.mghpcc.org/z-image-turbo/health \
  -H "Authorization: Bearer $NERC_API_TOKEN"
```

### Generate Image
```bash
curl -k https://data-science-gateway.apps.ocp-beta-test.nerc.mghpcc.org/z-image-turbo/v1/images/generations \
  -H "Authorization: Bearer $NERC_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Tongyi-MAI/Z-Image-Turbo",
    "prompt": "A sunset over mountains",
    "n": 1,
    "size": "512x512"
  }'
```

## Autoscaling Configuration

**KEDA ScaledObject:**
- **Metric:** HTTP request rate from Prometheus
- **Query:** `sum(rate(http_requests_total{namespace="omni-demo",pod=~"z-image-turbo-predictor.*",handler="/v1/images/generations"}[30s]))`
- **Threshold:** 0.2 req/sec per replica
- **Min Replicas:** 1
- **Max Replicas:** 5
- **Polling Interval:** 5 seconds
- **Cooldown Period:** 60 seconds

**Scaling Behavior:**
- Traffic < 0.2 req/sec → 1 replica
- Traffic 0.4 req/sec → 2 replicas
- Traffic 0.6 req/sec → 3 replicas
- Traffic 0.8 req/sec → 4 replicas
- Traffic ≥ 1.0 req/sec → 5 replicas (max)

## Monitoring

### Check Deployment Status
```bash
# Pods
oc get pods -n omni-demo

# InferenceService
oc get inferenceservice -n omni-demo

# HPA (created by KEDA)
oc get hpa -n omni-demo

# HTTPRoute
oc get httproute -n openshift-ingress omni-demo-route
```

### View Metrics
```bash
# Current request rate
oc exec -n omni-demo <pod-name> -c kserve-container -- \
  curl -s localhost:8000/metrics | grep http_requests_total

# KEDA scaling metrics
oc get scaledobject -n omni-demo z-image-turbo-scaler -o yaml
```

## Benefits of ModelCar Approach

### vs. Traditional Storage URI (HuggingFace/S3)

| Aspect | ModelCar | Traditional |
|--------|----------|-------------|
| **Pod Startup** | Fast (local extraction) | Slow (downloads model) |
| **Network Usage** | Low (one-time pull) | High (30GB+ per pod) |
| **Consistency** | Guaranteed | Depends on network |
| **Version Control** | Container tags | Model path/version |
| **Portability** | Full (OCI standard) | Limited |
| **Offline Support** | Yes (after initial pull) | No |

### vs. HostPath/PVC Approach

| Aspect | ModelCar | HostPath/PVC |
|--------|----------|--------------|
| **Setup Complexity** | Low | High |
| **Cross-Node Scaling** | Yes | Limited (RWO) or Complex (RWX) |
| **Storage Management** | Container registry | PV provisioning |
| **Updates** | Push new image | Update files on storage |
| **Cleanup** | Automatic | Manual |

## Troubleshooting

### Pod stuck in ImagePullBackOff
```bash
# Check image accessibility
podman pull quay.io/vraiti/z-image-turbo:v1

# Verify image pull secret if using private registry
oc get secrets -n omni-demo
```

### HTTPRoute returns 503
```bash
# Check service endpoints
oc get endpoints -n omni-demo z-image-turbo-gateway-svc

# Check pod status
oc get pods -n omni-demo

# Check HTTPRoute status
oc describe httproute -n openshift-ingress omni-demo-route
```

### Autoscaling not working
```bash
# Check HPA status
oc get hpa -n omni-demo

# Check KEDA authentication
oc describe triggerauthentication -n omni-demo keda-trigger-auth-prometheus

# Verify Prometheus query
# (Run query in Thanos UI to verify it returns data)
```

### Model fails to load
```bash
# Check container logs
oc logs -n omni-demo <pod-name> -c kserve-container

# Verify model location in container
oc exec -n omni-demo <pod-name> -c kserve-container -- ls -la /models
```

## Cleanup

```bash
# Delete HTTPRoute from openshift-ingress
oc delete -f 08-httproute.yaml

# Delete all resources from omni-demo namespace
oc delete -k .

# Or delete the entire namespace
oc delete namespace omni-demo

# Clean up cluster-scoped resources
oc delete clusterrolebinding keda-prometheus-metrics-reader-binding
oc delete clusterrolebinding keda-prometheus-monitoring-view
oc delete clusterrole keda-prometheus-metrics-reader
```

## Customization

### Change ModelCar Image

Edit `01-servingruntime.yaml` to change the runtime image:
```yaml
image: docker.io/vllm/vllm-omni:0.15.0
```

Edit `02-inferenceservice.yaml` to change the model source:
```yaml
storageUri: oci://quay.io/your-org/your-modelcar-image:tag
```

### Adjust Autoscaling

Edit `05-scaledobject.yaml`:
```yaml
minReplicaCount: 2        # Minimum replicas
maxReplicaCount: 10       # Maximum replicas
threshold: '0.5'          # Requests/sec per replica
pollingInterval: 10       # Check every 10 seconds
cooldownPeriod: 120       # Wait 2 minutes before scaling down
```

### Change Gateway Path

Edit `08-httproute.yaml`:
```yaml
- path:
    type: PathPrefix
    value: /my-custom-path
```

## References

- [Red Hat OpenShift AI Documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai)
- [Build and deploy a ModelCar container](https://developers.redhat.com/articles/2025/01/30/build-and-deploy-modelcar-container-openshift-ai)
- [KEDA Documentation](https://keda.sh/)
- [KServe Documentation](https://kserve.github.io/website/)
- [Gateway API Documentation](https://gateway-api.sigs.k8s.io/)
