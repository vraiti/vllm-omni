PROMPT="A llama eating gpus"

curl -k -X POST "$SERVICE_URL/v1/images/generations" \
  --max-time 120 \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Tongyi-MAI/Z-Image-Turbo\",
    \"prompt\": \"$PROMPT\",
    \"n\": 1,
    \"size\": \"1024x1024\"
  }" | jq -r '.data[0].b64_json' | base64 -d > image.png
