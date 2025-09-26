#!/usr/bin/env python3
"""
Test script to simulate audio data being sent to Redis for ASR processing
"""
import redis
import json
import numpy as np
import time
import uuid

def generate_test_audio():
    """Generate test audio data"""
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # Hz
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, audio.shape)
    audio = audio + noise
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()

def send_audio_to_redis():
    """Send test audio data to Redis"""
    print("🎤 Sending test audio to Redis...")
    
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Generate test audio
        audio_data = generate_test_audio()
        audio_hex = audio_data.hex()
        
        # Create meeting ID
        meeting_id = str(uuid.uuid4())
        
        # Create audio chunk data
        audio_chunk = {
            "meeting_id": meeting_id,
            "data": audio_hex,
            "timestamp": time.time(),
            "sample_rate": 16000,
            "channels": 1
        }
        
        # Send to Redis queue
        queue_name = f"meeting:{meeting_id}:audio"
        r.lpush(queue_name, json.dumps(audio_chunk))
        
        print(f"✅ Audio chunk sent to Redis queue: {queue_name}")
        print(f"📊 Audio data size: {len(audio_data)} bytes")
        print(f"🆔 Meeting ID: {meeting_id}")
        
        # Also publish to NLU for testing
        nlu_data = {
            "meeting_id": meeting_id,
            "text": "This is a test message from Redis",
            "timestamp": time.time(),
            "confidence": 0.9
        }
        
        r.publish("nlu_process", json.dumps(nlu_data))
        print("✅ Test message published to NLU channel")
        
        return True
        
    except Exception as e:
        print(f"❌ Error sending audio to Redis: {e}")
        return False

def monitor_nlu_channel():
    """Monitor NLU channel for responses"""
    print("\n👂 Monitoring NLU channel for responses...")
    
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        pubsub = r.pubsub()
        pubsub.subscribe("nlu_process")
        
        print("Waiting for messages (press Ctrl+C to stop)...")
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                print(f"📨 Received NLU message: {data}")
                
    except KeyboardInterrupt:
        print("\n👋 Stopping monitor...")
    except Exception as e:
        print(f"❌ Error monitoring NLU channel: {e}")

if __name__ == "__main__":
    print("🚀 Redis Audio Test")
    print("=" * 30)
    
    # Send test audio
    success = send_audio_to_redis()
    
    if success:
        print("\n💡 To test the full pipeline:")
        print("1. Start the ASR service: python services/asr/main.py")
        print("2. Start the NLU service: python services/nlu/main.py")
        print("3. Run this script again to send audio data")
        print("4. Check the services for processing logs")
        
        # Ask if user wants to monitor
        try:
            response = input("\nMonitor NLU channel? (y/n): ").lower()
            if response == 'y':
                monitor_nlu_channel()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print("\n❌ Failed to send audio data. Check Redis connection.")
