import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_expected_ranges():
    print("=== Testing Expected Ranges ===")
    response = requests.get(f"{BASE_URL}/expected_ranges")
    print(f"Status: {response.status_code}")
    print("Expected input ranges:")
    for sensor, ranges in response.json().items():
        print(f"  {sensor}: [{ranges['min']:.2f}, {ranges['max']:.2f}]")
    print()

def test_single_prediction():
    print("=== Testing Single Prediction ===")
    data = {
        "vibration": 3.0,
        "temperature": 310,
        "pressure": 150,
        "current": 40,
        "rpm": 1500,
        "tool_wear": 100
    }
    response = requests.post(f"{BASE_URL}/predict?equipment_id=EQ_TEST_001", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_batch_prediction():
    print("=== Testing Batch Prediction ===")
    data = [
        {
            "equipment_id": "EQ_001",
            "vibration": 3.0,
            "temperature": 310,
            "pressure": 150,
            "current": 40,
            "rpm": 1500,
            "tool_wear": 100
        },
        {
            "equipment_id": "EQ_002",
            "vibration": 3.02,
            "temperature": 311,
            "pressure": 160,
            "current": 45,
            "rpm": 1600,
            "tool_wear": 120
        }
    ]
    response = requests.post(f"{BASE_URL}/batch_predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_out_of_range_prediction():
    print("=== Testing Out-of-Range Prediction (should show warnings) ===")
    data = {
        "vibration": 0.5,  # Too low (expected: ~2.94-3.06)
        "temperature": 70,  # Too low (expected: ~305-314)
        "pressure": 100,
        "current": 10,     # Too low (expected: ~20-60)
        "rpm": 3000,       # Too high (expected: ~1200-1876)
        "tool_wear": 200
    }
    response = requests.post(f"{BASE_URL}/predict?equipment_id=EQ_OUT_OF_RANGE", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    test_health()
    test_expected_ranges()
    test_single_prediction()
    test_batch_prediction()
    test_out_of_range_prediction()