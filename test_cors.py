#!/usr/bin/env python3
"""
Simple CORS test script to verify the backend is responding correctly
"""

import requests
import json

def test_cors():
    """Test CORS configuration"""
    backend_url = "https://backend.quickcap.pro"
    origin = "https://quickcap.pro"
    
    print("Testing CORS configuration...")
    print(f"Backend URL: {backend_url}")
    print(f"Origin: {origin}")
    
    # Test 1: OPTIONS preflight request
    print("\n1. Testing OPTIONS preflight request...")
    try:
        response = requests.options(
            f"{backend_url}/api/status",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            },
            timeout=10
        )
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        
        if "Access-Control-Allow-Origin" in response.headers:
            print("   ✅ CORS headers present")
        else:
            print("   ❌ CORS headers missing")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 2: GET request to status endpoint
    print("\n2. Testing GET request to /api/status...")
    try:
        response = requests.get(
            f"{backend_url}/api/status",
            headers={"Origin": origin},
            timeout=10
        )
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("   ✅ Status endpoint working")
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
        else:
            print(f"   ❌ Status endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 3: Check if backend is reachable
    print("\n3. Testing basic connectivity...")
    try:
        response = requests.get(f"{backend_url}/api/status", timeout=5)
        print(f"   Backend is reachable: {response.status_code == 200}")
    except Exception as e:
        print(f"   ❌ Backend not reachable: {str(e)}")

if __name__ == "__main__":
    test_cors()