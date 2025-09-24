#!/usr/bin/env python3
"""
Test script to verify AI Meeting Assistant setup
"""

import requests
import time
import sys
from typing import Dict, List

# Service endpoints
SERVICES = {
    "ingestion": "http://localhost:8001",
    "asr": "http://localhost:8002", 
    "nlu": "http://localhost:8003",
    "rag": "http://localhost:8004",
    "agent": "http://localhost:8005",
    "integrations": "http://localhost:8006",
    "ui": "http://localhost:3000"
}

def test_service_health(service_name: str, base_url: str) -> bool:
    """Test if a service is healthy"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} service is healthy")
            return True
        else:
            print(f"❌ {service_name} service returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {service_name} service is not responding: {e}")
        return False

def test_database_connection() -> bool:
    """Test database connection"""
    try:
        import asyncpg
        import asyncio
        
        async def test_db():
            conn = await asyncpg.connect("postgresql://postgres:postgres@localhost:5432/meeting_assistant")
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            return result == 1
        
        result = asyncio.run(test_db())
        if result:
            print("✅ Database connection is healthy")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def test_redis_connection() -> bool:
    """Test Redis connection"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection is healthy")
        return True
    except Exception as e:
        print(f"❌ Redis connection error: {e}")
        return False

def test_meeting_workflow() -> bool:
    """Test basic meeting workflow"""
    try:
        # Test meeting start
        meeting_data = {
            "meeting_id": "test-meeting-123",
            "title": "Test Meeting",
            "platform": "web",
            "start_time": time.time(),
            "privacy_mode": "transcript+notes",
            "participants": ["test@example.com"]
        }
        
        response = requests.post(
            f"{SERVICES['ingestion']}/meetings/start",
            json=meeting_data,
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ Meeting start workflow works")
            
            # Test meeting end
            end_response = requests.post(
                f"{SERVICES['ingestion']}/meetings/{meeting_data['meeting_id']}/end",
                timeout=5
            )
            
            if end_response.status_code == 200:
                print("✅ Meeting end workflow works")
                return True
            else:
                print("❌ Meeting end workflow failed")
                return False
        else:
            print("❌ Meeting start workflow failed")
            return False
            
    except Exception as e:
        print(f"❌ Meeting workflow error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing AI Meeting Assistant Setup")
    print("=" * 50)
    
    # Test database and Redis
    db_ok = test_database_connection()
    redis_ok = test_redis_connection()
    
    if not db_ok or not redis_ok:
        print("\n❌ Infrastructure tests failed. Please check your setup.")
        sys.exit(1)
    
    # Test services
    print("\n🔍 Testing Services")
    print("-" * 30)
    
    healthy_services = 0
    for service_name, base_url in SERVICES.items():
        if test_service_health(service_name, base_url):
            healthy_services += 1
    
    # Test workflow
    print("\n🔄 Testing Workflow")
    print("-" * 30)
    workflow_ok = test_meeting_workflow()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"Healthy Services: {healthy_services}/{len(SERVICES)}")
    print(f"Database: {'✅' if db_ok else '❌'}")
    print(f"Redis: {'✅' if redis_ok else '❌'}")
    print(f"Workflow: {'✅' if workflow_ok else '❌'}")
    
    if healthy_services == len(SERVICES) and db_ok and redis_ok and workflow_ok:
        print("\n🎉 All tests passed! AI Meeting Assistant is ready to use.")
        print("\n📱 Access the application at: http://localhost:3000")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the logs and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
