import os
import torch
from datetime import datetime
from typing import Optional, Dict
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

class DatabaseConfig:
    """Database configuration class"""
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://savindumdk:mazner2002@cluster0.7ebo7.mongodb.net/")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "voice_auth")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "user_data")

# Global database variables
client = None
db = None
collection = None

def initialize_database():
    """Initialize MongoDB connection"""
    global client, db, collection
    
    try:
        config = DatabaseConfig()
        client = MongoClient(config.MONGODB_URI)
        db = client[config.DATABASE_NAME]
        collection = db[config.COLLECTION_NAME]
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB connection established successfully.")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        client = None
        db = None
        collection = None
        return False

def save_user_embedding(user_id: str, embedding: torch.Tensor, samples_info: Optional[Dict] = None) -> bool:
    """Save user embedding to MongoDB with optional sample information."""
    if collection is None:
        return False
    
    try:
        embedding_list = embedding.squeeze().tolist()
        
        # Prepare document
        doc = {
            "user_id": user_id,
            "embedding": embedding_list,
            "updated_at": datetime.now().isoformat()
        }
        
        # Add sample information if provided
        if samples_info:
            doc.update(samples_info)
        
        collection.update_one(
            {"user_id": user_id},
            {"$set": doc},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"Error saving user embedding: {e}")
        return False

def update_user_embedding_ewma(user_id: str, new_embedding: torch.Tensor, alpha: float = 0.1) -> bool:
    """
    Update user embedding using Exponentially Weighted Moving Average (EWMA)
    
    Formula: E_new = α * E_incoming + (1-α) * E_current
    
    Args:
        user_id: User identifier
        new_embedding: New embedding from successful verification (score > 0.65)
        alpha: Learning rate (0.05-0.2), controls how much the new sample influences the master voiceprint
    
    Returns:
        bool: Success status
    """
    if collection is None:
        return False
        
    try:
        # Get current embedding
        current_embedding = get_user_embedding(user_id)
        if current_embedding is None:
            print(f"⚠️ No existing embedding found for {user_id}, saving new embedding directly")
            return save_user_embedding(user_id, new_embedding)
        
        # Convert to same device and ensure shapes match
        current_embedding = current_embedding.to(new_embedding.device)
        
        # Apply EWMA formula: E_new = α * E_incoming + (1-α) * E_current
        updated_embedding = alpha * new_embedding + (1 - alpha) * current_embedding
        
        # Save the updated embedding
        embedding_list = updated_embedding.squeeze().tolist()
        
        doc = {
            "user_id": user_id,
            "embedding": embedding_list,
            "updated_at": datetime.now().isoformat(),
            "last_ewma_update": datetime.now().isoformat(),
            "ewma_alpha": alpha
        }
        
        collection.update_one(
            {"user_id": user_id},
            {"$set": doc},
            upsert=True
        )
        
        print(f"🔄 EWMA Update: {user_id} voiceprint adapted (α={alpha:.3f})")
        return True
        
    except Exception as e:
        print(f"❌ Error updating embedding with EWMA for {user_id}: {e}")
        return False

def save_user_samples(user_id: str, sample_embeddings: list, averaged_embedding: torch.Tensor) -> bool:
    """Save multiple sample embeddings and their average for a user."""
    if collection is None:
        return False
    
    try:
        # Convert embeddings to lists
        samples_list = [emb.squeeze().tolist() for emb in sample_embeddings]
        averaged_list = averaged_embedding.squeeze().tolist()
        
        # Prepare comprehensive document
        doc = {
            "user_id": user_id,
            "embedding": averaged_list,  # Main embedding for verification
            "sample_embeddings": samples_list,  # Individual samples
            "num_samples": len(sample_embeddings),
            "enrollment_type": "multi_sample",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        collection.update_one(
            {"user_id": user_id},
            {"$set": doc},
            upsert=True
        )
        print(f"✅ Saved {len(sample_embeddings)} samples and averaged embedding for user '{user_id}'")
        return True
    except Exception as e:
        print(f"❌ Error saving user samples: {e}")
        return False

def get_user_enrollment_info(user_id: str) -> Optional[Dict]:
    """Retrieve detailed enrollment information for a user."""
    if collection is None:
        return None
    
    try:
        user_doc = collection.find_one({"user_id": user_id})
        if user_doc:
            # Remove MongoDB _id field for cleaner output
            user_doc.pop('_id', None)
            return user_doc
        return None
    except Exception as e:
        print(f"Error retrieving user enrollment info: {e}")
        return None

def get_user_embedding(user_id: str) -> Optional[torch.Tensor]:
    """Retrieve user embedding from MongoDB."""
    if collection is None:
        return None
    
    try:
        user_doc = collection.find_one({"user_id": user_id})
        if user_doc and "embedding" in user_doc:
            embedding_list = user_doc["embedding"]
            return torch.tensor(embedding_list).unsqueeze(0)
        return None
    except Exception as e:
        print(f"Error retrieving user embedding: {e}")
        return None

def get_all_user_embeddings() -> Dict[str, torch.Tensor]:
    """
    Retrieve all user embeddings from MongoDB for 1-to-many comparison.
    """
    if collection is None:
        return {}
    
    try:
        all_users = collection.find({})
        embeddings_dict = {}
        for user_doc in all_users:
            if "user_id" in user_doc and "embedding" in user_doc:
                user_id = user_doc["user_id"]
                embedding_list = user_doc["embedding"]
                embeddings_dict[user_id] = torch.tensor(embedding_list).unsqueeze(0)
        return embeddings_dict
    except Exception as e:
        print(f"Error retrieving all user embeddings: {e}")
        return {}

def user_exists(user_id: str) -> bool:
    """Check if user exists in MongoDB."""
    if collection is None:
        return False
    
    try:
        return collection.find_one({"user_id": user_id}) is not None
    except Exception as e:
        print(f"Error checking user existence: {e}")
        return False

def update_voiceprint_with_ewma(user_id: str, new_embedding: torch.Tensor, alpha: float = 0.1) -> bool:
    """
    Update user's voiceprint using Exponentially Weighted Moving Average (EWMA).
    
    Formula: E_new = α * E_incoming + (1-α) * E_current
    
    Args:
        user_id: User identifier
        new_embedding: New high-confidence embedding to blend in
        alpha: Learning rate (0.05-0.2 recommended). Lower = more conservative
    
    Returns:
        bool: Success status
    """
    if collection is None:
        return False
    
    try:
        # Get current embedding
        current_doc = collection.find_one({"user_id": user_id})
        if not current_doc or "embedding" not in current_doc:
            print(f"⚠️ No existing embedding found for user '{user_id}' - skipping EWMA update")
            return False
        
        # Convert current embedding to tensor
        current_embedding = torch.tensor(current_doc["embedding"]).unsqueeze(0)
        
        # Ensure both embeddings are the same shape
        if current_embedding.shape != new_embedding.shape:
            print(f"⚠️ Shape mismatch: current {current_embedding.shape} vs new {new_embedding.shape}")
            return False
        
        # Apply EWMA formula: E_new = α * E_incoming + (1-α) * E_current
        updated_embedding = alpha * new_embedding + (1 - alpha) * current_embedding
        
        # Normalize the updated embedding to maintain unit length (important for cosine similarity)
        updated_embedding = updated_embedding / updated_embedding.norm(dim=-1, keepdim=True)
        
        # Convert back to list for storage
        updated_embedding_list = updated_embedding.squeeze().tolist()
        
        # Update the document with EWMA-blended embedding
        update_doc = {
            "embedding": updated_embedding_list,
            "updated_at": datetime.now().isoformat(),
            "last_ewma_update": datetime.now().isoformat(),
            "ewma_alpha": alpha
        }
        
        # Increment adaptation counter if it exists
        if "adaptation_count" in current_doc:
            update_doc["adaptation_count"] = current_doc["adaptation_count"] + 1
        else:
            update_doc["adaptation_count"] = 1
        
        collection.update_one(
            {"user_id": user_id},
            {"$set": update_doc}
        )
        
        print(f"🔄 EWMA voiceprint update for '{user_id}': α={alpha:.3f}, adaptation #{update_doc['adaptation_count']}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating voiceprint with EWMA: {e}")
        return False

def store_voice_sample(user_id: str, embedding: torch.Tensor, similarity_score: float, 
                      audio_duration: float = None, audio_info: Dict = None) -> bool:
    """
    Store voice samples in 'voice_data' collection for analysis and training.
    All authentication attempts are stored regardless of similarity score.
    
    Args:
        user_id: User identifier
        embedding: Voice embedding tensor
        similarity_score: Verification similarity score
        audio_duration: Duration of audio sample in seconds
        audio_info: Additional audio metadata (sample_rate, energy, etc.)
    
    Returns:
        bool: Success status
    """
    if db is None:
        print("❌ Database not initialized for voice sample storage")
        return False
        
    try:
        # Get the voice_data collection
        voice_data_collection = db["voice_data"]
        
        # Convert embedding to list for MongoDB storage
        embedding_list = embedding.squeeze().tolist()
        
        # Prepare the voice sample document
        voice_sample = {
            "user_id": user_id,
            "embedding": embedding_list,
            "similarity_score": similarity_score,
            "timestamp": datetime.now().isoformat(),
            "audio_duration": audio_duration,
            "embedding_dim": len(embedding_list)
        }
        
        # Add audio information if provided
        if audio_info:
            voice_sample.update(audio_info)
        
        # Insert the voice sample
        result = voice_data_collection.insert_one(voice_sample)
        
        print(f"💾 Voice sample stored: {user_id} (score: {similarity_score:.4f}, sample_id: {result.inserted_id})")
        return True
        
    except Exception as e:
        print(f"❌ Error storing voice sample for {user_id}: {e}")
        return False

def store_labeled_voice_sample(user_id: str, embedding: torch.Tensor, similarity_score: float, 
                              is_actual_speaker: bool, audio_duration: float = None, 
                              audio_info: Dict = None) -> bool:
    """
    Store labeled voice samples in 'sample_voice_data' collection with ground truth labels.
    This function stores voice samples along with whether they are from the actual speaker or not.
    
    Args:
        user_id: User identifier
        embedding: Voice embedding tensor
        similarity_score: Verification similarity score
        is_actual_speaker: Ground truth label - True if this is the actual speaker, False otherwise
        audio_duration: Duration of audio sample in seconds
        audio_info: Additional audio metadata (sample_rate, energy, etc.)
    
    Returns:
        bool: Success status
    """
    if db is None:
        print("❌ Database not initialized for labeled voice sample storage")
        return False
        
    try:
        # Get the sample_voice_data collection
        sample_voice_data_collection = db["sample_voice_data"]
        
        # Convert embedding to list for MongoDB storage
        embedding_list = embedding.squeeze().tolist()
        
        # Prepare the labeled voice sample document
        voice_sample = {
            "user_id": user_id,
            "embedding": embedding_list,
            "similarity_score": similarity_score,
            "is_actual_speaker": is_actual_speaker,  # Ground truth label
            "timestamp": datetime.now().isoformat(),
            "audio_duration": audio_duration,
            "embedding_dim": len(embedding_list),
            "sample_type": "labeled"  # Mark as labeled sample for analysis
        }
        
        # Add audio information if provided
        if audio_info:
            voice_sample.update(audio_info)
        
        # Insert the labeled voice sample
        result = sample_voice_data_collection.insert_one(voice_sample)
        
        label_str = "ACTUAL" if is_actual_speaker else "IMPOSTER"
        print(f"💾 Labeled voice sample stored: {user_id} [{label_str}] (score: {similarity_score:.4f}, sample_id: {result.inserted_id})")
        return True
        
    except Exception as e:
        print(f"❌ Error storing labeled voice sample for {user_id}: {e}")
        return False

def get_user_voice_samples(user_id: str, limit: int = None, min_score: float = 0.5) -> list:
    """
    Retrieve stored voice samples for a specific user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of samples to return (None for all)
        min_score: Minimum similarity score filter
    
    Returns:
        list: List of voice sample documents
    """
    if db is None:
        print("❌ Database not initialized")
        return []
        
    try:
        sample_voice_data_collection = db["sample_voice_data"]
        
        # Build query
        query = {
            "user_id": user_id,
            "similarity_score": {"$gte": min_score}
        }
        
        # Execute query with optional limit
        cursor = sample_voice_data_collection.find(query).sort("timestamp", -1)  # Most recent first
        if limit:
            cursor = cursor.limit(limit)
        
        samples = list(cursor)
        print(f"📊 Retrieved {len(samples)} voice samples for {user_id} (score ≥ {min_score})")
        return samples
        
    except Exception as e:
        print(f"❌ Error retrieving voice samples for {user_id}: {e}")
        return []

def get_voice_data_statistics() -> Dict:
    """
    Get statistics about stored voice samples.
    
    Returns:
        dict: Statistics including total samples, users, score distribution
    """
    if db is None:
        print("❌ Database not initialized")
        return {}
        
    try:
        sample_voice_data_collection = db["sample_voice_data"]
        
        # Basic statistics
        total_samples = sample_voice_data_collection.count_documents({})
        unique_users = len(sample_voice_data_collection.distinct("user_id"))
        
        # Score distribution
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$similarity_score"},
                    "max_score": {"$max": "$similarity_score"},
                    "min_score": {"$min": "$similarity_score"}
                }
            }
        ]
        
        score_stats = list(sample_voice_data_collection.aggregate(pipeline))
        
        stats = {
            "total_samples": total_samples,
            "unique_users": unique_users,
            "score_statistics": score_stats[0] if score_stats else {}
        }
        
        print(f"📈 Voice Data Stats: {total_samples} samples from {unique_users} users")
        return stats
        
    except Exception as e:
        print(f"❌ Error getting voice data statistics: {e}")
        return {}

def close_database_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        print("📫 MongoDB connection closed.")