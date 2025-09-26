#!/usr/bin/env python3
"""
Diagnostic script to identify what's not working
"""
import requests
import json
import time
import subprocess
import sys

def check_port():
    """Check if port 8002 is accessible"""
    print("🔍 Checking port 8002...")
    try:
        result = subprocess.run(['lsof', '-i', ':8002'], capture_output=True, text=True)
        if result.returncode == 0 and 'Python' in result.stdout:
            print("✅ Port 8002 is in use by Python process")
            return True
        else:
            print("❌ Port 8002 is not in use")
            return False
    except Exception as e:
        print(f"❌ Error checking port: {e}")
        return False

def check_server_response():
    """Check if server responds to HTTP requests"""
    print("\n🌐 Checking server HTTP response...")
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server responding: {data}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server - connection refused")
        return False
    except requests.exceptions.Timeout:
        print("❌ Server timeout - not responding")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False

def check_web_interface():
    """Check if web interface loads"""
    print("\n🖥️ Checking web interface...")
    try:
        response = requests.get("http://localhost:8002/", timeout=10)
        if response.status_code == 200:
            if "AI Meeting Assistant" in response.text:
                print("✅ Web interface loads correctly")
                return True
            else:
                print("❌ Web interface loads but content seems wrong")
                return False
        else:
            print(f"❌ Web interface returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error loading web interface: {e}")
        return False

def check_groq_api():
    """Check if Groq API is working"""
    print("\n🤖 Checking Groq API...")
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )
        
        ai_response = response.choices[0].message.content.strip()
        print(f"✅ Groq API working: {ai_response[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return False

def test_meeting_api():
    """Test meeting creation API"""
    print("\n📅 Testing meeting API...")
    try:
        meeting_data = {
            "meeting_id": "diagnostic_test",
            "title": "Diagnostic Test Meeting",
            "platform": "AI Meeting Assistant",
            "start_time": time.time(),
            "privacy_mode": "private",
            "participants": ["Test User"]
        }
        
        response = requests.post("http://localhost:8002/meetings/start", 
                               json=meeting_data, timeout=5)
        if response.status_code == 200:
            print("✅ Meeting API working")
            return True
        else:
            print(f"❌ Meeting API failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Meeting API error: {e}")
        return False

def main():
    print("🔧 AI Meeting Assistant Diagnostic Tool")
    print("=" * 50)
    
    tests = [
        ("Port Check", check_port),
        ("Server Response", check_server_response),
        ("Web Interface", check_web_interface),
        ("Groq API", check_groq_api),
        ("Meeting API", test_meeting_api)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n📊 Diagnostic Summary:")
    print("=" * 30)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! The server should be working.")
        print("📱 Try opening: http://localhost:8002")
    else:
        print("\n⚠️ Some tests failed. Here's what to check:")
        if not results.get("Port Check"):
            print("  • Server is not running - start it with: python3 demo.py")
        if not results.get("Server Response"):
            print("  • Server is not responding - check if it's stuck in a loop")
        if not results.get("Web Interface"):
            print("  • Web interface issue - check browser console for errors")
        if not results.get("Groq API"):
            print("  • Groq API issue - check API key and internet connection")
        if not results.get("Meeting API"):
            print("  • Meeting API issue - check server logs")

if __name__ == "__main__":
    main()
