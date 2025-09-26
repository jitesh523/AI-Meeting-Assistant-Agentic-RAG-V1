#!/usr/bin/env python3
"""
Test script for Groq integration in the AI Meeting Assistant
"""
import requests
import json
import time
import websocket
import threading

def test_groq_via_api():
    """Test Groq integration through the demo API"""
    print("🧪 Testing Groq Integration via Demo API")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8002/health")
        health_data = response.json()
        print(f"✅ Health Check: {health_data}")
        
        if not health_data.get("groq_available"):
            print("❌ Groq not available in demo")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Start a test meeting
    meeting_id = f"test_meeting_{int(time.time())}"
    try:
        response = requests.post("http://localhost:8002/meetings/start", json={
            "meeting_id": meeting_id,
            "title": "Groq Integration Test",
            "platform": "AI Meeting Assistant",
            "start_time": time.time(),
            "privacy_mode": "private",
            "participants": ["Test User"]
        })
        
        if response.status_code == 200:
            print(f"✅ Meeting started: {meeting_id}")
        else:
            print(f"❌ Failed to start meeting: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting meeting: {e}")
        return False
    
    return meeting_id

def test_websocket_groq(meeting_id):
    """Test Groq via WebSocket connection"""
    print(f"\n🔌 Testing WebSocket connection for meeting: {meeting_id}")
    
    received_suggestions = []
    
    def on_message(ws, message):
        try:
            data = json.loads(message)
            print(f"📨 Received: {data.get('type', 'unknown')}")
            
            if data.get('type') == 'suggestion':
                suggestion = data.get('suggestion', {})
                received_suggestions.append(suggestion)
                print(f"🤖 AI Suggestion: {suggestion.get('text', 'No text')[:100]}...")
                print(f"   Kind: {suggestion.get('kind')}, Confidence: {suggestion.get('confidence')}")
                
        except Exception as e:
            print(f"❌ Error parsing message: {e}")
    
    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print("🔌 WebSocket connection closed")
    
    def on_open(ws):
        print("✅ WebSocket connected")
        
        # Send test messages to trigger Groq AI processing
        test_messages = [
            "We need to discuss the Q4 budget allocation and resource planning.",
            "What are the key milestones for our product launch timeline?",
            "I think we should schedule a follow-up meeting to review the action items."
        ]
        
        for i, message in enumerate(test_messages):
            time.sleep(2)  # Wait between messages
            ws.send(json.dumps({
                "type": "chat",
                "text": message
            }))
            print(f"📤 Sent message {i+1}: {message[:50]}...")
    
    # Connect to WebSocket
    ws_url = f"ws://localhost:8002/ws/audio/{meeting_id}"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Run WebSocket in a separate thread
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    
    # Wait for messages
    print("⏳ Waiting for AI suggestions (10 seconds)...")
    time.sleep(10)
    
    # Close connection
    ws.close()
    
    return received_suggestions

def test_direct_groq():
    """Test Groq API directly"""
    print("\n🔬 Testing Groq API Directly")
    print("=" * 30)
    
    try:
        from groq import Groq
        
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Test with meeting context
        prompt = """You are an AI meeting assistant. Based on this meeting transcript:
Speaker: Sarah Chen
Text: We need to finalize the Q4 budget allocation and ensure we have enough resources for the product launch.
Timestamp: 2024-01-15T10:30:00

Generate a helpful suggestion for the meeting. Consider:
- If it's a question, suggest follow-up questions
- If it's a decision point, suggest next steps
- If it's a problem, suggest solutions
- If it's a task, suggest action items

Respond with a concise suggestion (max 150 words) that would be helpful for this meeting context."""
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content.strip()
        print(f"✅ Direct Groq Response:")
        print(f"📝 {ai_response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct Groq test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Groq Integration Test Suite")
    print("=" * 40)
    
    # Test 1: Direct Groq API
    direct_success = test_direct_groq()
    
    # Test 2: Demo API health
    meeting_id = test_groq_via_api()
    
    if meeting_id:
        # Test 3: WebSocket integration
        suggestions = test_websocket_groq(meeting_id)
        
        print(f"\n📊 Test Results:")
        print(f"  Direct Groq API: {'✅' if direct_success else '❌'}")
        print(f"  Demo Health Check: {'✅' if meeting_id else '❌'}")
        print(f"  WebSocket Suggestions: {len(suggestions)} received")
        
        if suggestions:
            print(f"\n🎯 Sample AI Suggestions:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"  {i}. {suggestion.get('text', 'No text')[:100]}...")
        
        if direct_success and meeting_id and suggestions:
            print(f"\n🎉 All tests passed! Groq integration is working perfectly!")
        else:
            print(f"\n⚠️ Some tests failed. Check the error messages above.")
    else:
        print(f"\n❌ Cannot proceed with WebSocket test - meeting setup failed")
