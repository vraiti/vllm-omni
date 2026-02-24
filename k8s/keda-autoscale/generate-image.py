#!/usr/bin/env python3
"""
Simple script to test image generation and save the result.
"""

import requests
import base64
import subprocess
import os
import argparse
import urllib3

# Disable SSL warnings when verify=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def discover_gateway_endpoint(namespace: str = "openshift-ingress",
                             gateway_name: str = "data-science-gateway",
                             httproute_name: str = "omni-keda-demo-route") -> str:
    """
    Discover the Gateway endpoint URL programmatically.

    Returns the full HTTPS URL to the service, or None if discovery fails.
    """
    try:
        # Get Gateway Route hostname
        result = subprocess.run(
            ["oc", "get", "route", "-n", namespace, gateway_name,
             "-o", "jsonpath={.spec.host}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"Warning: Could not get Gateway route: {result.stderr}")
            return None

        gateway_host = result.stdout.strip()
        if not gateway_host:
            print("Warning: Gateway route hostname is empty")
            return None

        # Get HTTPRoute path prefix
        result = subprocess.run(
            ["oc", "get", "httproute", "-n", namespace, httproute_name,
             "-o", "jsonpath={.spec.rules[0].matches[0].path.value}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"Warning: Could not get HTTPRoute: {result.stderr}")
            return None

        path_prefix = result.stdout.strip()
        if not path_prefix:
            print("Warning: HTTPRoute path prefix is empty")
            return None

        # Combine into full URL
        endpoint = f"https://{gateway_host}{path_prefix}"
        print(f"Discovered Gateway endpoint: {endpoint}")
        return endpoint

    except subprocess.TimeoutExpired:
        print("Warning: Timeout while discovering Gateway endpoint")
        return None
    except FileNotFoundError:
        print("Warning: 'oc' command not found, cannot discover endpoint")
        return None
    except Exception as e:
        print(f"Warning: Error discovering Gateway endpoint: {e}")
        return None


def generate_image(service_url: str, prompt: str, output_path: str = "image.png"):
    """Generate a single image and save it to a file."""

    url = f"{service_url}/v1/images/generations"
    payload = {
        "model": "Tongyi-MAI/Z-Image-Turbo",
        "prompt": prompt,
        "n": 1,
        "size": "512x512"
    }

    # Get API token from environment if available
    headers = {}
    api_token = os.environ.get('NERC_API_TOKEN')
    if api_token:
        headers['Authorization'] = f'Bearer {api_token}'
        print(f"Using API token from NERC_API_TOKEN environment variable")

    print(f"Requesting image generation...")
    print(f"URL: {url}")
    print(f"Prompt: {prompt}")

    response = requests.post(url, json=payload, headers=headers, timeout=300, verify=False)

    if response.status_code == 200:
        result = response.json()

        # Extract image data from response
        # OpenAI-compatible API returns: {"data": [{"b64_json": "..." or "url": "..."}]}
        if "data" in result and len(result["data"]) > 0:
            image_data = result["data"][0]

            # Handle base64 encoded image
            if "b64_json" in image_data:
                image_bytes = base64.b64decode(image_data["b64_json"])
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                print(f"Image saved to {output_path}")

            # Handle URL-based image
            elif "url" in image_data:
                img_response = requests.get(image_data["url"])
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
                print(f"Image saved to {output_path}")

            else:
                print(f"Unexpected response format: {result}")
        else:
            print(f"No image data in response: {result}")
    else:
        print(f"Error: HTTP {response.status_code}")
        print(f"Response: {response.text}")


def main():
    parser = argparse.ArgumentParser(description='Generate a single image')
    parser.add_argument('--url',
                       default=None,
                       help='Service URL (default: auto-discover from Gateway)')
    parser.add_argument('--prompt',
                       default='A beautiful landscape with mountains and lakes',
                       help='Image generation prompt')
    parser.add_argument('--output',
                       default='image.png',
                       help='Output file path (default: image.png)')
    parser.add_argument('--namespace',
                       default='openshift-ingress',
                       help='Namespace where Gateway is deployed (default: openshift-ingress)')
    parser.add_argument('--gateway-name',
                       default='data-science-gateway',
                       help='Gateway name (default: data-science-gateway)')
    parser.add_argument('--httproute-name',
                       default='omni-keda-demo-route',
                       help='HTTPRoute name (default: omni-keda-demo-route)')

    args = parser.parse_args()

    # Discover endpoint if not provided
    service_url = args.url
    if not service_url:
        service_url = discover_gateway_endpoint(
            namespace=args.namespace,
            gateway_name=args.gateway_name,
            httproute_name=args.httproute_name
        )
        if not service_url:
            print("Error: Could not discover Gateway endpoint and --url not provided")
            print("Please specify --url manually or ensure oc is configured correctly")
            return 1

    generate_image(service_url, args.prompt, args.output)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
