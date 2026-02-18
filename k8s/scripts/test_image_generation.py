#!/usr/bin/env python3
"""
Test script for vLLM-Omni Image Generation via KServe InferenceService

This script tests the deployed vLLM-Omni service by:
1. Checking the health endpoint
2. Listing available models
3. Generating an image from a text prompt
"""
import requests
import json
import sys
import base64
import subprocess
from datetime import datetime

def get_service_url():
    """Get the service URL from OpenShift route"""
    try:
        result = subprocess.run(
            ["oc", "get", "route", "vllm-omni-turbo", "--output", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        route_data = json.loads(result.stdout)
        host = route_data.get("spec", {}).get("host", "")
        if host:
            return f"https://{host}"
        else:
            raise ValueError("No host found in route")
    except Exception as e:
        print(f"Error getting service URL: {e}")
        sys.exit(1)

SERVICE_URL = get_service_url()
MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"

def test_health():
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/health", verify=False, timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def list_models():
    """List available models"""
    print("\nListing available models...")
    try:
        response = requests.get(f"{SERVICE_URL}/v1/models", verify=False, timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Available models: {json.dumps(models, indent=2)}")
            return True
        else:
            print(f"❌ Failed to list models: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

def generate_image(prompt, output_file="generated_image.png"):
    """Generate an image from a text prompt"""
    print(f"\nGenerating image with prompt: '{prompt}'")

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }

    try:
        print("Sending request to /v1/images/generations...")
        response = requests.post(
            f"{SERVICE_URL}/v1/images/generations",
            json=payload,
            verify=False,
            timeout=120  # Image generation can take time
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Image generation successful!")
            print(f"Response: {json.dumps(result, indent=2)[:500]}...")

            # Save image if base64 encoded
            if "data" in result and len(result["data"]) > 0:
                image_data = result["data"][0]
                if "b64_json" in image_data:
                    print(f"Saving image to {output_file}...")
                    image_bytes = base64.b64decode(image_data["b64_json"])
                    with open(output_file, "wb") as f:
                        f.write(image_bytes)
                    print(f"✅ Image saved to {output_file}")
                elif "url" in image_data:
                    print(f"Image URL: {image_data['url']}")

            return True
        else:
            print(f"❌ Image generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return False

def main():
    print("=" * 60)
    print("vLLM-Omni Image Generation Test")
    print(f"Service: {SERVICE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Suppress SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Test health
    if not test_health():
        print("\n❌ Service is not healthy, aborting tests")
        sys.exit(1)

    # List models
    list_models()

    # Generate image
    prompt = "A serene mountain landscape at sunset with a lake reflecting the orange sky and a dragon"
    success = generate_image(prompt, "/tmp/vllm_omni_test_output.png")

    print("\n" + "=" * 60)
    if success:
        print("✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
