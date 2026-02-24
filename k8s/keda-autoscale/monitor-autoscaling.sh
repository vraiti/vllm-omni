#!/bin/bash
# Monitor KEDA autoscaling during stress test

NAMESPACE="omni-demo"
DEPLOYMENT="omni-keda-demo-predictor"
SCALEDOBJECT="omni-keda-demo-scaler"

# Monitor in a loop
while true; do
    # Collect all output into a variable
    OUTPUT=""

    OUTPUT+="==========================================\n"
    OUTPUT+="KEDA Autoscaling Status - $(date '+%Y-%m-%d %H:%M:%S')\n"
    OUTPUT+="==========================================\n"
    OUTPUT+="\n"

    OUTPUT+="--- ScaledObject Status ---\n"
    OUTPUT+="$(oc get scaledobject $SCALEDOBJECT -n $NAMESPACE 2>/dev/null || echo "ScaledObject not found")\n"
    OUTPUT+="\n"

    OUTPUT+="--- HPA Status (managed by KEDA) ---\n"
    OUTPUT+="$(oc get hpa -n $NAMESPACE -l scaledobject.keda.sh/name=$SCALEDOBJECT 2>/dev/null || echo "No HPA found yet")\n"
    OUTPUT+="\n"

    OUTPUT+="--- Deployment Replicas ---\n"
    DESIRED=$(oc get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    READY=$(oc get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    AVAILABLE=$(oc get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.status.availableReplicas}' 2>/dev/null || echo "0")
    OUTPUT+="Desired: $DESIRED | Ready: $READY | Available: $AVAILABLE\n"
    OUTPUT+="\n"

    OUTPUT+="--- Pods ---\n"
    OUTPUT+="$(oc get pods -n $NAMESPACE -l serving.kserve.io/inferenceservice=omni-keda-demo -o wide 2>/dev/null)\n"
    OUTPUT+="\n"

    OUTPUT+="--- KEDA Scaler Events (last 5) ---\n"
    OUTPUT+="$(oc get events -n $NAMESPACE --field-selector involvedObject.name=$SCALEDOBJECT \
        --sort-by='.lastTimestamp' 2>/dev/null | tail -6)\n"
    OUTPUT+="\n"

    # Display all collected output at once
    clear
    echo -e "$OUTPUT"

    sleep 5
done
