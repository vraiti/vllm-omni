#!/usr/bin/env python3
"""
Stress test script for vLLM-Omni deployment.
Sends requests at a controlled frequency.
"""

import asyncio
import aiohttp
import time
import os
import ssl
import subprocess
import json
from datetime import datetime
import argparse


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


async def generate_image(session: aiohttp.ClientSession, service_url: str, request_id: int, headers: dict = None):
    """Send a single image generation request."""
    url = f"{service_url}/v1/images/generations"
    payload = {
        "model": "Tongyi-MAI/Z-Image-Turbo",
        "prompt": f"A beautiful landscape with mountains and lakes, request {request_id}",
        "n": 1,
        "size": "512x512"
    }

    request_start = time.time()
    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as response:
            if response.status == 200:
                await response.json()
                elapsed = time.time() - request_start
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id:3d} completed in {elapsed:.2f}s")
                return True
            else:
                error_text = await response.text()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id:3d} failed: HTTP {response.status}")
                return False
    except asyncio.TimeoutError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id:3d} timed out")
        return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id:3d} failed: {str(e)[:100]}")
        return False


async def run_stress_test(service_url: str, num_requests: int, frequency: float):
    """Send requests at controlled frequency."""
    # Get API token from environment
    api_token = os.environ.get('NERC_API_TOKEN')
    headers = {}
    if api_token:
        headers['Authorization'] = f'Bearer {api_token}'
        print(f"Using API token from NERC_API_TOKEN environment variable")
    else:
        print(f"Warning: NERC_API_TOKEN not set, requests may fail if authentication is required")

    print(f"\n{'='*60}")
    print(f"Stress test started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Target: {service_url}")
    print(f"Requests: {num_requests}")
    print(f"Frequency: {frequency}s between requests ({1/frequency:.2f} req/sec)")
    print(f"{'='*60}\n")

    start_time = time.time()
    tasks = []

    # Create SSL context that doesn't verify certificates (like curl -k)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Launch requests at fixed intervals
        for i in range(num_requests):
            task = asyncio.create_task(generate_image(session, service_url, i, headers))
            tasks.append(task)

            # Wait before launching next request (except after the last one)
            if i < num_requests - 1:
                await asyncio.sleep(frequency)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    completed = sum(results)
    failed = len(results) - completed

    print(f"\n{'='*60}")
    print(f"Stress test completed at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Completed: {completed}/{num_requests}")
    print(f"Failed: {failed}/{num_requests}")
    print(f"Average rate: {completed/total_time:.2f} requests/sec")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Stress test vLLM-Omni deployment with controlled request frequency',
        epilog='Example: %(prog)s --url https://... --requests 50 --frequency 0.5',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--url',
                       default=None,
                       help='Service URL (default: auto-discover from Gateway)')
    parser.add_argument('--requests', type=int, required=True,
                       help='Total number of requests to send')
    parser.add_argument('--frequency', type=float, required=True,
                       help='Seconds to wait between requests (e.g., 0.5 = 2 req/sec)')
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

    asyncio.run(run_stress_test(service_url, args.requests, args.frequency))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
