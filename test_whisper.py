#!/usr/bin/env python3
"""
Test script for Whisper integration
"""
import whisper
import numpy as np
import time

def test_whisper():
    """Test Whisper model loading and basic transcription"""
    print("🎤 Testing Whisper integration...")
    
    try:
        # Load Whisper model
        print("Loading Whisper model (small)...")
        model = whisper.load_model("small")
        print("✅ Whisper model loaded successfully!")
        
        # Create a simple test audio (sine wave)
        print("Generating test audio...")
        sample_rate = 16000
        duration = 2  # seconds
        frequency = 440  # Hz (A note)
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.1, audio.shape)
        audio = audio + noise
        
        # Normalize
        audio = audio.astype(np.float32)
        
        print("Transcribing test audio...")
        start_time = time.time()
        
        # Transcribe
        result = model.transcribe(audio, fp16=False)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Transcription completed in {processing_time:.2f} seconds")
        print(f"📝 Result: '{result['text']}'")
        print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
        
        if result['segments']:
            print("\n📊 Segments:")
            for i, segment in enumerate(result['segments']):
                print(f"  {i+1}. [{segment['start']:.2f}s - {segment['end']:.2f}s]: '{segment['text']}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Whisper: {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    print("\n🔗 Testing Redis connection...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test connection
        r.ping()
        print("✅ Redis connection successful!")
        
        # Test pub/sub
        r.publish("test_channel", "Hello Redis!")
        print("✅ Redis pub/sub working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Make sure Redis is running: redis-server")
        return False

def test_database_connection():
    """Test PostgreSQL connection"""
    print("\n🗄️ Testing PostgreSQL connection...")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            dbname="meetings",
            user="admin", 
            password="secret",
            host="localhost"
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()
            print(f"✅ PostgreSQL connection successful! Version: {version[0]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("💡 Make sure PostgreSQL is running with the correct credentials")
        return False

if __name__ == "__main__":
    print("🚀 AI Meeting Assistant - Whisper Integration Test")
    print("=" * 50)
    
    # Test Whisper
    whisper_ok = test_whisper()
    
    # Test Redis
    redis_ok = test_redis_connection()
    
    # Test Database
    db_ok = test_database_connection()
    
    print("\n📋 Test Summary:")
    print(f"  Whisper: {'✅' if whisper_ok else '❌'}")
    print(f"  Redis: {'✅' if redis_ok else '❌'}")
    print(f"  Database: {'✅' if db_ok else '❌'}")
    
    if whisper_ok and redis_ok and db_ok:
        print("\n🎉 All tests passed! Whisper integration is ready.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
