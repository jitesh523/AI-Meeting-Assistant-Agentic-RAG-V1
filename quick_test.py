#!/usr/bin/env python3
"""
Quick test to verify the AI Meeting Assistant is working
"""
import requests
import json

def test_server():
    print("🧪 Quick Server Test")
    print("=" * 30)
    
    # Test 1: Health check
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health Check: {health}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Start a meeting
    try:
        meeting_data = {
            "meeting_id": "test_meeting_quick",
            "title": "Quick Test Meeting",
            "platform": "AI Meeting Assistant",
            "start_time": 1234567890,
            "privacy_mode": "private",
            "participants": ["Test User"]
        }
        
        response = requests.post("http://localhost:8002/meetings/start", 
                               json=meeting_data, timeout=5)
        if response.status_code == 200:
            print("✅ Meeting started successfully")
        else:
            print(f"❌ Failed to start meeting: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error starting meeting: {e}")
        return False
    
    # Test 3: Test Groq AI directly
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello! This is a test."}],
            max_tokens=50
        )
        
        ai_response = response.choices[0].message.content.strip()
        print(f"✅ Groq AI working: {ai_response[:100]}...")
        
    except Exception as e:
        print(f"❌ Groq AI test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The server is working correctly.")
    print("📱 Open your browser to: http://localhost:8002")
    return True

if __name__ == "__main__":
    test_server()
