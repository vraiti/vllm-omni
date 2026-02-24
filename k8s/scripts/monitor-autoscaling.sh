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
    OUTPUT+="$(oc get scaledobject)\n"
    OUTPUT+="\n"

    OUTPUT+="--- HPA Status (managed by KEDA) ---\n"
    OUTPUT+="$(oc get hpa)\n"
    OUTPUT+="\n"

    OUTPUT+="--- Deployment Replicas ---\n"
    OUTPUT+=$(oc get deployment)
    OUTPUT+="\n\n"

    OUTPUT+="--- Pods ---\n"
    OUTPUT+="$(oc get pods -o wide)\n"
    OUTPUT+="\n"

    # Display all collected output at once
    clear
    echo -e "$OUTPUT"

    sleep 5
done
