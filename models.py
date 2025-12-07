"""
Global Model Management for Voice Authentication System
Handles caching and initialization of all ML models
"""

import torch
import os
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.inference.VAD import VAD
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.enhancement import SpectralMaskEnhancement

# Configuration
class AppConfig:
    SPEAKER_VERIFIER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    VERIFICATION_THRESHOLD = float(os.environ.get('VERIFICATION_THRESHOLD', '0.50'))
    
    # EWMA Configuration
    EWMA_ENABLED = os.environ.get('EWMA_ENABLED', 'true').lower() == 'true'
    EWMA_ADAPTATION_THRESHOLD = float(os.environ.get('EWMA_ADAPTATION_THRESHOLD', '0.70'))
    EWMA_LEARNING_RATE = float(os.environ.get('EWMA_LEARNING_RATE', '0.1'))

CONFIG = AppConfig()

# Device detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Models module using device: {DEVICE}")

# ----------------------------
# Global Model Manager
# ----------------------------

class ModelManager:
    """Global model manager to cache frequently used models and avoid reloading."""
    
    def __init__(self):
        self.vad_model = None
        self.separator_2mix = None
        self.separator_3mix = None
        self.enhancer = None
        self.pyannote_vad = None
        self.pyannote_diarization = None
        
    def load_models(self, device="cpu"):
        """Load all models at startup."""
        print("🔄 Loading pipeline models for caching...")
        
        # Load VAD model
        try:
            print("📡 Loading VAD model...")
            self.vad_model = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                savedir="pretrained_models/vad",
                run_opts={"device": device}
            )
            print("✅ VAD model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load VAD model: {e}")
            
        # Load Sepformer models
        try:
            print("📡 Loading Sepformer 2-speaker model...")
            self.separator_2mix = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir="pretrained_models/sepformer_wsj02mix",
                run_opts={"device": device}
            )
            print("✅ Sepformer 2-speaker model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load Sepformer 2-speaker model: {e}")
            
        try:
            print("📡 Loading Sepformer 3-speaker model...")
            self.separator_3mix = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj03mix",
                savedir="pretrained_models/sepformer_wsj03mix",
                run_opts={"device": device}
            )
            print("✅ Sepformer 3-speaker model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load Sepformer 3-speaker model: {e}")
            
        # Load Enhancement model
        try:
            print("📡 Loading Enhancement model...")
            # Try different enhancement models in order of preference
            enhancement_models = [
                ("speechbrain/metricgan-plus-voicebank", "pretrained_models/metricgan-plus-voicebank"),
                ("speechbrain/mtl-mimic-voicebank", "pretrained_models/mtl-mimic-voicebank"),
                ("speechbrain/sepformer-wham-enhancement", "pretrained_models/sepformer-wham-enhancement")
            ]
            
            for model_source, save_dir in enhancement_models:
                try:
                    print(f"   Trying {model_source}...")
                    self.enhancer = SpectralMaskEnhancement.from_hparams(
                        source=model_source,
                        savedir=save_dir,
                        run_opts={"device": device}
                    )
                    print(f"✅ Enhancement model loaded: {model_source}")
                    break
                except Exception as model_error:
                    print(f"   Failed to load {model_source}: {model_error}")
                    continue
            else:
                # If all models fail, set enhancer to None
                print("⚠️ All enhancement models failed to load, enhancement will be disabled")
                self.enhancer = None
                
        except Exception as e:
            print(f"⚠️ Failed to load Enhancement model: {e}")
            self.enhancer = None
            
        # Load PyAnnote models (optional, may require auth)
        # First, try to login to HuggingFace if token is available
        hf_token = os.getenv('HF_AUTH_TOKEN', None)
        if hf_token:
            try:
                from huggingface_hub import login
                print("� Logging in to HuggingFace...")
                login(token=hf_token)
                print("✅ HuggingFace login successful")
            except Exception as e:
                print(f"⚠️ HuggingFace login failed: {e}")
        
        try:
            from pyannote.audio import Pipeline
            print("📡 Loading PyAnnote VAD pipeline...")
            if hf_token:
                print("🔑 Using HuggingFace authentication token for VAD...")
                # Try both old and new parameter names for compatibility
                try:
                    self.pyannote_vad = Pipeline.from_pretrained(
                        "pyannote/voice-activity-detection",
                        token=hf_token  # New HF API
                    )
                except TypeError:
                    self.pyannote_vad = Pipeline.from_pretrained(
                        "pyannote/voice-activity-detection",
                        use_auth_token=hf_token  # Old HF API
                    )
            else:
                print("⚠️ No HF_AUTH_TOKEN found, trying without authentication...")
                self.pyannote_vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
            print("✅ PyAnnote VAD pipeline loaded")
        except Exception as e:
            print(f"⚠️ Failed to load PyAnnote VAD: {e}")
            
        try:
            from pyannote.audio import Pipeline
            print("📡 Loading PyAnnote Speaker Diarization pipeline...")
            if hf_token:
                print("🔑 Using HuggingFace authentication token for Diarization...")
                # Try both old and new parameter names for compatibility
                try:
                    self.pyannote_diarization = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization", 
                        token=hf_token  # New HF API
                    )
                except TypeError:
                    self.pyannote_diarization = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization", 
                        use_auth_token=hf_token  # Old HF API
                    )
            else:
                print("⚠️ No HF_AUTH_TOKEN found, trying without authentication...")
                self.pyannote_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization")
            print("✅ PyAnnote Speaker Diarization pipeline loaded")
        except Exception as e:
            print(f"⚠️ Failed to load PyAnnote Diarization: {e}")
            
        print("🎯 Model loading complete!")
        
    def get_vad_model(self):
        """Get cached VAD model."""
        return self.vad_model
        
    def get_separator(self, num_speakers=2):
        """Get cached Sepformer model for specified number of speakers."""
        if num_speakers <= 2:
            return self.separator_2mix
        else:
            return self.separator_3mix
            
    def get_enhancer(self):
        """Get cached Enhancement model."""
        return self.enhancer
        
    def get_pyannote_vad(self):
        """Get cached PyAnnote VAD pipeline."""
        return self.pyannote_vad
        
    def get_pyannote_diarization(self):
        """Get cached PyAnnote Diarization pipeline."""
        return self.pyannote_diarization

# Create global model manager instance
model_manager = ModelManager()

# Initialize core speaker models globally
speaker_encoder = None
speaker_verifier = None

def load_core_models():
    """Load core speaker models."""
    global speaker_encoder, speaker_verifier
    
    try:
        print(f"Loading Speaker Models from source: {CONFIG.SPEAKER_VERIFIER_MODEL}")
        
        # EncoderClassifier for extracting embeddings
        speaker_encoder = EncoderClassifier.from_hparams(
            source=CONFIG.SPEAKER_VERIFIER_MODEL,
            run_opts={"device": DEVICE}
        )
        
        # SpeakerRecognition for verification tasks
        speaker_verifier = SpeakerRecognition.from_hparams(
            source=CONFIG.SPEAKER_VERIFIER_MODEL,
            run_opts={"device": DEVICE}
        )
        
        print("✅ Speaker Models Loaded Successfully.")
        
        # Load pipeline models for caching
        model_manager.load_models(DEVICE)
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        speaker_encoder = None
        speaker_verifier = None

# Auto-load models when module is imported
print("🔧 Initializing global models...")
load_core_models()
