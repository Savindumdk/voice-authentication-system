#!/usr/bin/env python3
"""
Test MongoDB connection to diagnose the issue
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
import dns.resolver

# Load environment variables
load_dotenv()

def test_dns_resolution():
    """Test DNS resolution for MongoDB Atlas"""
    print("🔍 Testing DNS resolution...")
    try:
        # Test basic DNS resolution
        result = dns.resolver.resolve('cluster0.7ebo7.mongodb.net', 'A')
        print(f"✅ DNS A record resolution successful: {[str(r) for r in result]}")
        
        # Test SRV record resolution (what MongoDB Atlas uses)
        srv_result = dns.resolver.resolve('_mongodb._tcp.cluster0.7ebo7.mongodb.net', 'SRV')
        print(f"✅ DNS SRV record resolution successful: {[str(r) for r in srv_result]}")
        return True
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}")
        return False

def test_mongodb_connection():
    """Test MongoDB connection"""
    print("\n🔍 Testing MongoDB connection...")
    
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb+srv://savindumdk:mazner2002@cluster0.7ebo7.mongodb.net/")
    print(f"📡 Connecting to: {mongodb_uri}")
    
    try:
        # Create client with timeout
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        print("📡 Testing connection...")
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # Test database access
        db_name = os.getenv("DATABASE_NAME", "voice_auth")
        db = client[db_name]
        collection_name = os.getenv("COLLECTION_NAME", "user_data")
        collection = db[collection_name]
        
        print(f"✅ Database '{db_name}' accessible")
        print(f"✅ Collection '{collection_name}' accessible")
        
        # Close connection
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def test_alternative_connection():
    """Test alternative connection methods"""
    print("\n🔍 Testing alternative connection...")
    
    # Try without SRV (using direct host)
    try:
        # Extract cluster details and try direct connection
        alt_uri = "mongodb://cluster0-shard-00-00.7ebo7.mongodb.net:27017,cluster0-shard-00-01.7ebo7.mongodb.net:27017,cluster0-shard-00-02.7ebo7.mongodb.net:27017/voice_auth?ssl=true&replicaSet=atlas-123abc-shard-0&authSource=admin&retryWrites=true&w=majority"
        print(f"📡 Trying alternative connection...")
        
        client = MongoClient(alt_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ Alternative connection successful!")
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Alternative connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 MongoDB Connection Diagnostic Test")
    print("=" * 50)
    
    # Test DNS
    dns_ok = test_dns_resolution()
    
    # Test MongoDB connection
    mongo_ok = test_mongodb_connection()
    
    # If primary fails, try alternative
    if not mongo_ok:
        alt_ok = test_alternative_connection()
    
    print("\n📋 Test Summary:")
    print(f"DNS Resolution: {'✅' if dns_ok else '❌'}")
    print(f"MongoDB Connection: {'✅' if mongo_ok else '❌'}")
    
    if not mongo_ok and not dns_ok:
        print("\n💡 Suggestions:")
        print("1. Check internet connection")
        print("2. Check firewall settings")
        print("3. Verify MongoDB Atlas cluster is running")
        print("4. Check MongoDB credentials")
