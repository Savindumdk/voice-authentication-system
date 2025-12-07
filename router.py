import torch
import torchaudio
import tempfile
import os
import subprocess
import shutil
import warnings
import logging
import sys
import contextlib
from io import StringIO
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import asyncio

# Suppress all warnings before any other imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="pytorch_lightning")
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="speechbrain")
warnings.filterwarnings("ignore", module="pyannote")

# Suppress specific warning messages
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*ModelCheckpoint.*callback states.*")
warnings.filterwarnings("ignore", message=".*Model was trained with.*")
warnings.filterwarnings("ignore", message=".*Bad things might happen.*")
warnings.filterwarnings("ignore", message=".*symlinks on Windows.*")
warnings.filterwarnings("ignore", message=".*Xet Storage is enabled.*")
warnings.filterwarnings("ignore", message=".*hf_xet.*package is not installed.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")

# Suppress PyTorch Lightning logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr during model loading"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Load environment variables
load_dotenv()

# Get HF auth token from environment
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN', None)

# FORCE all models to use local cache - set before any AI library imports
local_cache_base = os.path.abspath('./pretrained_models')
os.environ['HF_HOME'] = f'{local_cache_base}/huggingface'
os.environ['TORCH_HOME'] = f'{local_cache_base}/torch'
os.environ['TRANSFORMERS_CACHE'] = f'{local_cache_base}/huggingface'
os.environ['HF_HUB_CACHE'] = f'{local_cache_base}/huggingface'
os.environ['PYANNOTE_CACHE'] = f'{local_cache_base}/pyannote'

# Create cache directories immediately
cache_dirs = [
    f'{local_cache_base}/huggingface',
    f'{local_cache_base}/torch', 
    f'{local_cache_base}/pyannote',
    f'{local_cache_base}/speechbrain'
]

for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)

# GPU Device Configuration
def get_optimal_device():
    """Detect and configure the best available device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        print(f"🔥 Router using GPU acceleration: {device}")
        return device
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available in router, using CPU")
        return device

# Set global device
DEVICE = get_optimal_device()

# Configure PyTorch for optimal GPU usage
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed

# Import database functions
from database import save_user_embedding, get_all_user_embeddings, user_exists, save_user_samples, get_user_enrollment_info, update_user_embedding_ewma, store_voice_sample, store_labeled_voice_sample, get_voice_data_statistics, get_user_voice_samples

# Import global models
from models import speaker_encoder, speaker_verifier, model_manager, DEVICE, CONFIG

# Create router
router = APIRouter()

# Get HF auth token from environment
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN', None)

def apply_advanced_vad_diarization_pipeline(signal: torch.Tensor, enrolled_embeddings: dict = None, sample_rate: int = 16000) -> torch.Tensor:
    """
    Apply the complete advanced pipeline:
    Mic Input → pyannote VAD → Speaker Diarization → Speech Separation (Sepformer) → Enhancement → Clean Audio
    
    This comprehensive pipeline can:
    1. Detect voice activity using pyannote VAD
    2. Perform speaker diarization to identify different speakers
    3. Separate overlapping speech using Sepformer
    4. Select the best audio source matching enrolled users
    5. Apply enhancement for optimal quality
    
    Args:
        signal: Input audio tensor of shape [batch_size, samples] or [samples]
        enrolled_embeddings: Dict of user_id -> embedding for source selection
        sample_rate: Sample rate of the audio (default: 16000)
    
    Returns:
        Enhanced audio tensor with the same shape as input
    """
    try:
        # Import required modules
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import SpeakerRecognition
        
        # Try to import pyannote modules (may require auth token)
        try:
            from pyannote.audio import Pipeline
            pyannote_available = True
        except ImportError:
            print("⚠️ pyannote.audio not available, skipping VAD/diarization")
            pyannote_available = False
        
        original_shape = signal.shape
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)
        
        # Check if signal is too short for meaningful processing
        min_length = sample_rate * 1.0  # At least 1 second for diarization
        if signal.shape[1] < min_length:
            print(f"⚠️ Signal too short ({signal.shape[1]} samples) for advanced processing, returning original")
            if len(original_shape) == 1:
                return signal.squeeze(0)
            return signal
        
        enhanced_batch = []
        
        for i in range(signal.shape[0]):
            audio = signal[i]
            print(f"🎙️ Processing audio sample {i+1}/{signal.shape[0]} with advanced VAD+Diarization pipeline")
            
            # Step 1: Voice Activity Detection using pyannote
            vad_segments = None
            if pyannote_available:
                print("🔍 Step 1: Applying pyannote Voice Activity Detection...")
                try:
                    # Use cached VAD pipeline
                    vad_pipeline = model_manager.get_pyannote_vad()
                    if vad_pipeline is None:
                        print("⚠️ PyAnnote VAD model not cached, loading on-demand...")
                        # Initialize VAD pipeline with forced local cache and GPU support
                        local_pyannote_cache = os.path.abspath('./pretrained_models/pyannote')
                        with suppress_stdout_stderr():
                            vad_pipeline = Pipeline.from_pretrained(
                                "pyannote/voice-activity-detection",
                                use_auth_token=HF_AUTH_TOKEN,
                                cache_dir=local_pyannote_cache
                            )
                            # Move VAD pipeline to GPU if available
                            if DEVICE.type == "cuda":
                                vad_pipeline = vad_pipeline.to(DEVICE)
                    else:
                        print("✅ Using cached PyAnnote VAD pipeline")
                    
                    # Create temporary file for pyannote processing
                    import tempfile
                    import torchaudio
                    temp_file = None
                    
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                            temp_file = f.name
                        
                        # Save audio to temporary file (move to CPU first)
                        audio_for_save = audio.cpu() if audio.is_cuda else audio
                        torchaudio.save(temp_file, audio_for_save.unsqueeze(0), sample_rate)
                        
                        # Apply VAD
                        vad_result = vad_pipeline(temp_file)
                        
                        # Extract speech segments
                        vad_segments = []
                        for segment in vad_result.get_timeline():
                            start_sample = int(segment.start * sample_rate)
                            end_sample = int(segment.end * sample_rate)
                            if end_sample > start_sample and end_sample <= len(audio):
                                vad_segments.append((start_sample, end_sample))
                        
                        if vad_segments:
                            print(f"✅ VAD: Found {len(vad_segments)} speech segments")
                        else:
                            print("⚠️ VAD: No speech segments detected")
                    
                    finally:
                        if temp_file and os.path.exists(temp_file):
                            try:
                                os.unlink(temp_file)
                            except Exception:
                                pass
                
                except Exception as vad_error:
                    print(f"⚠️ VAD failed: {vad_error}")
                    vad_segments = None
            
            # Step 2: Speaker Diarization using pyannote
            diarization_segments = None
            if pyannote_available and vad_segments:
                print("👥 Step 2: Applying Speaker Diarization...")
                try:
                    # Use cached diarization pipeline
                    diarization_pipeline = model_manager.get_pyannote_diarization()
                    if diarization_pipeline is None:
                        print("⚠️ PyAnnote Diarization model not cached, loading on-demand...")
                        # Initialize speaker diarization pipeline with forced local cache and GPU support
                        local_pyannote_cache = os.path.abspath('./pretrained_models/pyannote')
                        with suppress_stdout_stderr():
                            diarization_pipeline = Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.1",
                                use_auth_token=HF_AUTH_TOKEN,
                                cache_dir=local_pyannote_cache
                            )
                            # Move diarization pipeline to GPU if available
                            if DEVICE.type == "cuda":
                                diarization_pipeline = diarization_pipeline.to(DEVICE)
                    else:
                        print("✅ Using cached PyAnnote Diarization pipeline")
                    
                    # Create temporary file for diarization
                    temp_file = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                            temp_file = f.name
                        
                        # Save audio to temporary file (move to CPU first)
                        audio_for_save = audio.cpu() if audio.is_cuda else audio
                        torchaudio.save(temp_file, audio_for_save.unsqueeze(0), sample_rate)
                        
                        # Apply diarization
                        diarization_result = diarization_pipeline(temp_file)
                        
                        # Extract speaker segments
                        diarization_segments = {}
                        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                            if speaker not in diarization_segments:
                                diarization_segments[speaker] = []
                            
                            start_sample = int(turn.start * sample_rate)
                            end_sample = int(turn.end * sample_rate)
                            if end_sample > start_sample and end_sample <= len(audio):
                                diarization_segments[speaker].append((start_sample, end_sample))
                        
                        if diarization_segments:
                            print(f"✅ Diarization: Found {len(diarization_segments)} speakers")
                            for speaker, segments in diarization_segments.items():
                                total_duration = sum([(end - start) / sample_rate for start, end in segments])
                                print(f"   👤 {speaker}: {len(segments)} segments, {total_duration:.2f}s total")
                        else:
                            print("⚠️ Diarization: No speakers detected")
                            diarization_result = None  # Clear result if no speakers detected
                    
                    finally:
                        if temp_file and os.path.exists(temp_file):
                            try:
                                os.unlink(temp_file)
                            except Exception:
                                pass
                
                except Exception as diar_error:
                    print(f"⚠️ Diarization failed: {diar_error}")
                    diarization_segments = None
            
            # Step 3: Dynamic Speech Separation based on detected speakers
            print("🔄 Step 3: Applying Dynamic Speech Separation...")
            separated_sources = None
            detected_speakers = 1  # Default fallback
            
            # Determine number of speakers from diarization
            if diarization_segments:
                # Count speakers from our processed segments dictionary
                detected_speakers = len(diarization_segments)
                print(f"🎭 Detected {detected_speakers} speakers from diarization")
            
            try:
                # Choose appropriate separation model based on speaker count
                if detected_speakers <= 2:
                    # Use WSJ 2-mix model for 2 speakers or less
                    print("📻 Using 2-speaker separation model (Sepformer WSJ02mix)")
                    separator = model_manager.get_separator(num_speakers=2)
                    if separator is None:
                        print("⚠️ 2-speaker Sepformer model not cached, loading on-demand...")
                        separator = SepformerSeparation.from_hparams(
                            source="speechbrain/sepformer-wsj02mix", 
                            savedir='pretrained_models/sepformer-wsj02mix',
                            run_opts={"device": DEVICE}
                        )
                elif detected_speakers <= 3:
                    # Use WSJ 3-mix model for 3 speakers
                    print("📻 Using 3-speaker separation model (Sepformer WSJ03mix)")
                    separator = model_manager.get_separator(num_speakers=3)
                    if separator is None:
                        print("⚠️ 3-speaker Sepformer model not cached, loading on-demand...")
                        try:
                            separator = SepformerSeparation.from_hparams(
                                source="speechbrain/sepformer-wsj03mix", 
                                savedir='pretrained_models/sepformer-wsj03mix',
                                run_opts={"device": DEVICE}
                            )
                        except Exception:
                            print("⚠️ 3-speaker model not available, falling back to 2-speaker model")
                            separator = model_manager.get_separator(num_speakers=2)
                            if separator is None:
                                separator = SepformerSeparation.from_hparams(
                                    source="speechbrain/sepformer-wsj02mix", 
                                    savedir='pretrained_models/sepformer-wsj02mix',
                                    run_opts={"device": DEVICE}
                                )
                else:
                    # For more speakers, use the most capable model available
                    print(f"📻 Many speakers detected ({detected_speakers}), using best available separation model")
                    try:
                        # Try WHAM! model which can handle more complex scenarios
                        separator = SepformerSeparation.from_hparams(
                            source="speechbrain/sepformer-wham", 
                            savedir='pretrained_models/sepformer-wham',
                            run_opts={"device": DEVICE}
                        )
                    except:
                        print("⚠️ Advanced model not available, falling back to 2-speaker model")
                        separator = SepformerSeparation.from_hparams(
                            source="speechbrain/sepformer-wsj02mix", 
                            savedir='pretrained_models/sepformer-wsj02mix',
                            run_opts={"device": DEVICE}
                        )
                
                # Prepare audio for separation and move to GPU if available
                if len(audio.shape) == 1:
                    separation_input = audio.unsqueeze(0)  # Add batch dimension
                else:
                    separation_input = audio
                
                # Move audio to GPU if available
                if DEVICE.type == "cuda":
                    separation_input = separation_input.to(DEVICE)
                
                # Apply separation (handles overlapping speakers)
                separated_sources = separator.separate_batch(separation_input)
                
                # Move results back to CPU for processing
                if DEVICE.type == "cuda":
                    separated_sources = separated_sources.cpu()
                
                actual_sources = separated_sources.shape[2]
                print(f"📊 Successfully separated into {actual_sources} sources")
                
                # Calculate energy for each source for debugging
                for src_idx in range(actual_sources):
                    source_energy = torch.sum(separated_sources[0, :, src_idx] ** 2).item()
                    print(f"   🔊 Source {src_idx+1} energy: {source_energy:.6f}")
                
            except Exception as separation_error:
                print(f"⚠️ Separation failed: {separation_error}")
                separated_sources = None
            
            # Step 4: Source Selection with multi-modal information
            selected_audio = audio  # Default fallback
            
            if separated_sources is not None:
                print(f"🎯 Step 4: Evaluating all {separated_sources.shape[2]} separated sources...")
                
                # If we have enrolled embeddings, use them for source selection
                if enrolled_embeddings and len(enrolled_embeddings) > 0:
                    try:
                        # Initialize speaker recognition for source evaluation with GPU support
                        speaker_encoder = SpeakerRecognition.from_hparams(
                            source="speechbrain/spkrec-ecapa-voxceleb",
                            savedir="pretrained_models/spkrec-ecapa-voxceleb",
                            run_opts={"device": DEVICE}
                        )
                        
                        # Store all source evaluations for comprehensive analysis
                        source_evaluations = []
                        
                        # Evaluate each separated source against enrolled users
                        for src_idx in range(separated_sources.shape[2]):
                            source_audio = separated_sources[0, :, src_idx].unsqueeze(0)
                            
                            # Calculate source statistics
                            source_energy = torch.sum(source_audio ** 2).item()
                            source_duration = len(source_audio[0]) / 16000  # Assuming 16kHz
                            
                            print(f"   🔍 Evaluating Source {src_idx+1}: energy={source_energy:.6f}, duration={source_duration:.2f}s")
                            
                            # Skip very short or silent sources
                            if source_energy < 0.001 or source_duration < 0.5:
                                print(f"   ⏭️ Skipping Source {src_idx+1} (too short/silent)")
                                continue
                            
                            try:
                                # Move source audio to GPU if available for embedding extraction
                                if DEVICE.type == "cuda":
                                    source_audio_gpu = source_audio.to(DEVICE)
                                    source_embedding = speaker_encoder.encode_batch(source_audio_gpu)
                                    # Move embedding back to CPU for similarity computation
                                    source_embedding = source_embedding.cpu()
                                else:
                                    source_embedding = speaker_encoder.encode_batch(source_audio)
                                
                                # Compare against all enrolled users
                                best_user_for_source = None
                                best_similarity_for_source = -1.0
                                
                                for user_id, user_embeddings in enrolled_embeddings.items():
                                    # Handle both single embeddings and lists of embeddings
                                    if isinstance(user_embeddings, list):
                                        # Average similarity across all enrolled samples for this user
                                        similarities = []
                                        for enrolled_embedding in user_embeddings:
                                            if len(enrolled_embedding.shape) > 1:
                                                enrolled_embedding = enrolled_embedding.squeeze()
                                            
                                            # Ensure both tensors are on the same device
                                            enrolled_embedding = enrolled_embedding.to(DEVICE)
                                            source_embedding_gpu = source_embedding.to(DEVICE)
                                            
                                            similarity = torch.nn.functional.cosine_similarity(
                                                source_embedding_gpu.squeeze(), 
                                                enrolled_embedding, 
                                                dim=0
                                            ).item()
                                            similarities.append(similarity)
                                        
                                        # Use average similarity for robustness
                                        avg_similarity = sum(similarities) / len(similarities)
                                    else:
                                        # Single embedding - ensure both are on same device
                                        user_embeddings = user_embeddings.to(DEVICE)
                                        source_embedding_gpu = source_embedding.to(DEVICE)
                                        
                                        avg_similarity = torch.nn.functional.cosine_similarity(
                                            source_embedding_gpu.squeeze(), 
                                            user_embeddings.squeeze(), 
                                            dim=0
                                        ).item()
                                    
                                    if avg_similarity > best_similarity_for_source:
                                        best_similarity_for_source = avg_similarity
                                        best_user_for_source = user_id
                                
                                # Store evaluation results
                                source_evaluations.append({
                                    'source_idx': src_idx,
                                    'user_id': best_user_for_source,
                                    'similarity': best_similarity_for_source,
                                    'energy': source_energy,
                                    'duration': source_duration
                                })
                                
                                print(f"   📊 Source {src_idx+1} best match: {best_user_for_source} (similarity: {best_similarity_for_source:.4f})")
                            
                            except Exception as embed_error:
                                print(f"   ⚠️ Failed to extract embedding from source {src_idx+1}: {embed_error}")
                        
                        # Select the best source based on similarity score
                        if source_evaluations:
                            # Sort by similarity score (descending)
                            source_evaluations.sort(key=lambda x: x['similarity'], reverse=True)
                            
                            best_evaluation = source_evaluations[0]
                            best_source = best_evaluation['source_idx']
                            selected_audio = separated_sources[0, :, best_source]
                            
                            print(f"✅ Selected Source {best_source+1} (similarity {best_evaluation['similarity']:.4f} with {best_evaluation['user_id']})")
                            print(f"   📊 Source rankings:")
                            for i, eval_data in enumerate(source_evaluations[:3]):  # Show top 3
                                print(f"   {i+1}. Source {eval_data['source_idx']+1}: {eval_data['user_id']} ({eval_data['similarity']:.4f})")
                        else:
                            # Fallback to energy-based selection
                            source_energies = [torch.sum(separated_sources[0, :, i] ** 2).item() 
                                             for i in range(separated_sources.shape[2])]
                            best_source = torch.argmax(torch.tensor(source_energies)).item()
                            selected_audio = separated_sources[0, :, best_source]
                            print(f"✅ Fallback: Selected Source {best_source+1} (highest energy: {source_energies[best_source]:.6f})")
                    
                    except Exception as selection_error:
                        print(f"⚠️ Source selection failed: {selection_error}")
                        # Use energy-based fallback
                        source_energies = [torch.sum(separated_sources[0, :, i] ** 2).item() 
                                         for i in range(separated_sources.shape[2])]
                        best_source = torch.argmax(torch.tensor(source_energies)).item()
                        selected_audio = separated_sources[0, :, best_source]
                        print(f"✅ Energy fallback: Selected source {best_source+1}")
                
                else:
                    # No enrolled embeddings: comprehensive energy and quality analysis
                    print("🔍 No enrolled embeddings, using comprehensive source analysis...")
                    source_evaluations = []
                    
                    for i in range(separated_sources.shape[2]):
                        source_audio = separated_sources[0, :, i]
                        energy = torch.sum(source_audio ** 2).item()
                        duration = len(source_audio) / 16000
                        
                        # Calculate additional quality metrics
                        rms_energy = torch.sqrt(torch.mean(source_audio ** 2)).item()
                        peak_amplitude = torch.max(torch.abs(source_audio)).item()
                        
                        # Simple voice activity estimation (energy above threshold)
                        voice_frames = (torch.abs(source_audio) > 0.01).float()
                        voice_activity_ratio = torch.mean(voice_frames).item()
                        
                        source_evaluations.append({
                            'source_idx': i,
                            'energy': energy,
                            'rms_energy': rms_energy,
                            'peak_amplitude': peak_amplitude,
                            'duration': duration,
                            'voice_activity_ratio': voice_activity_ratio,
                            'quality_score': energy * voice_activity_ratio * duration  # Combined quality metric
                        })
                        
                        print(f"   🔊 Source {i+1}: energy={energy:.6f}, rms={rms_energy:.4f}, voice_activity={voice_activity_ratio:.2f}, duration={duration:.2f}s")
                    
                    # Sort by quality score (combination of energy, voice activity, and duration)
                    source_evaluations.sort(key=lambda x: x['quality_score'], reverse=True)
                    
                    best_evaluation = source_evaluations[0]
                    best_source = best_evaluation['source_idx']
                    selected_audio = separated_sources[0, :, best_source]
                    
                    print(f"✅ Selected Source {best_source+1} (quality score: {best_evaluation['quality_score']:.6f})")
                    print(f"   📊 Source quality rankings:")
                    for i, eval_data in enumerate(source_evaluations):
                        print(f"   {i+1}. Source {eval_data['source_idx']+1}: quality={eval_data['quality_score']:.6f}")
                    
                    # Use diarization info for additional context if available
                    if diarization_segments:
                        speaker_durations = {}
                        for speaker, segments in diarization_segments.items():
                            total_duration = sum([(end - start) / sample_rate for start, end in segments])
                            speaker_durations[speaker] = total_duration
                        
                        if speaker_durations:
                            dominant_speaker = max(speaker_durations, key=speaker_durations.get)
                            print(f"   🎭 Diarization context: dominant speaker {dominant_speaker} ({speaker_durations[dominant_speaker]:.2f}s)")
                            print(f"   📊 All speakers: {dict(speaker_durations)}")
            
            # Step 5: Apply VAD filtering to selected audio with quality consistency check
            if vad_segments and len(vad_segments) > 0:
                print("✂️ Step 5: Applying VAD filtering to selected audio...")
                try:
                    # Store original audio for comparison
                    original_selected_audio = selected_audio.clone()
                    
                    # Extract only speech segments from selected audio
                    speech_parts = []
                    for start_sample, end_sample in vad_segments:
                        if end_sample <= len(selected_audio):
                            speech_parts.append(selected_audio[start_sample:end_sample])
                    
                    if speech_parts:
                        # Concatenate speech segments
                        filtered_audio = torch.cat(speech_parts, dim=0)
                        
                        # Ensure we have enough audio for embedding generation (minimum 1 second)
                        min_length = 16000  # 1 second at 16kHz
                        if len(filtered_audio) < min_length:
                            print(f"⚠️ VAD filtered audio too short ({len(filtered_audio)/16000:.2f}s), keeping original")
                            selected_audio = original_selected_audio
                        else:
                            # Quality consistency check: Compare embeddings before and after VAD
                            try:
                                from main import speaker_verifier
                                
                                # Prepare audio for embedding extraction (ensure proper dimensions)
                                original_batch = original_selected_audio.unsqueeze(0).to(DEVICE)
                                filtered_batch = filtered_audio.unsqueeze(0).to(DEVICE)
                                
                                # Generate embeddings
                                with torch.no_grad():
                                    original_embedding = speaker_verifier.encode_batch(original_batch)
                                    filtered_embedding = speaker_verifier.encode_batch(filtered_batch)
                                
                                # Calculate cosine similarity between embeddings
                                orig_norm = original_embedding / original_embedding.norm(dim=-1, keepdim=True)
                                filt_norm = filtered_embedding / filtered_embedding.norm(dim=-1, keepdim=True)
                                consistency = torch.cosine_similarity(orig_norm.squeeze(), filt_norm.squeeze(), dim=0).item()
                                
                                print(f"🔍 VAD Quality Check: embedding consistency = {consistency:.4f}")
                                
                                # If VAD filtering significantly degrades the embedding, use original
                                if consistency < 0.85:  # If consistency drops below 85%
                                    print(f"⚠️ VAD filtering degraded embedding quality (consistency: {consistency:.4f}), using original audio")
                                    selected_audio = original_selected_audio
                                else:
                                    print(f"✅ VAD filtering maintained embedding quality (consistency: {consistency:.4f})")
                                    # Pad or trim to maintain reasonable length (don't force original length)
                                    target_length = min(len(original_selected_audio), len(filtered_audio) + 8000)  # Allow some flexibility
                                    if len(filtered_audio) > target_length:
                                        selected_audio = filtered_audio[:target_length]
                                    else:
                                        selected_audio = filtered_audio
                                        
                                print(f"✅ VAD filtering applied, processed {len(speech_parts)} speech segments")
                                    
                            except Exception as embedding_error:
                                print(f"⚠️ Embedding consistency check failed: {embedding_error}, using original audio")
                                selected_audio = original_selected_audio
                    else:
                        print("⚠️ VAD filtering: No speech parts found, keeping original")
                        selected_audio = original_selected_audio
                
                except Exception as filter_error:
                    print(f"⚠️ VAD filtering failed: {filter_error}")
            else:
                print("✂️ Step 5: No VAD segments available, skipping VAD filtering")
            
            # Step 6: Final length adjustment
            target_length = audio.shape[0]
            if len(selected_audio) > target_length:
                selected_audio = selected_audio[:target_length]
            elif len(selected_audio) < target_length:
                padding = target_length - len(selected_audio)
                selected_audio = torch.nn.functional.pad(selected_audio, (0, padding))
            
            enhanced_batch.append(selected_audio)
        
        # Stack results
        result = torch.stack(enhanced_batch)
        
        # Return with original shape
        if len(original_shape) == 1:
            result = result.squeeze(0)
            
        print("🎯 Advanced pipeline completed: VAD → Diarization → Separation → Source Selection → Clean Audio")
        return result
        
    except Exception as e:
        print(f"⚠️ Advanced VAD+Diarization pipeline failed, returning original signal: {e}")
        return signal

# ----------------------------
# Pydantic Schemas
# ----------------------------

class EnrollmentResponse(BaseModel):
    status: str
    user_id: str
    message: str

class MultiSampleEnrollmentResponse(BaseModel):
    status: str
    user_id: str
    message: str
    samples_collected: int
    total_samples_needed: int
    enrollment_complete: bool

class EnrollmentStatusResponse(BaseModel):
    user_id: str
    enrollment_type: str
    num_samples: int
    created_at: str
    embedding_quality: str

class IdentificationResponse(BaseModel):
    status: str
    identified_user_id: Optional[str] = Field(None, description="The ID of the recognized user, or null if not identified.")
    message: str
    highest_score: float = Field(..., description="The best similarity score found among all users.")

class VerificationResponse(BaseModel):
    status: str
    target_user_id: str = Field(..., description="The user ID being verified against.")
    similarity_score: float = Field(..., description="The similarity score for the target user.")
    verified: bool = Field(..., description="Whether the user was successfully verified (score above threshold).")
    message: str

class SmartAuthResponse(BaseModel):
    action: str = Field(..., description="The action performed: 'enrolled' or 'verified'")
    status: str = Field(..., description="The status of the operation: 'success' or 'failed'")
    user_id: str = Field(..., description="The user ID processed")
    message: str = Field(..., description="Descriptive message about the operation")
    similarity_score: Optional[float] = Field(None, description="Similarity score (only for verification)")
    verified: Optional[bool] = Field(None, description="Verification result (only for verification)")

class LabeledSmartAuthResponse(BaseModel):
    action: str = Field(..., description="The action performed: 'enrolled' or 'verified'")
    status: str = Field(..., description="The status of the operation: 'success' or 'failed'")
    user_id: str = Field(..., description="The user ID processed")
    is_actual_speaker: bool = Field(..., description="Ground truth label indicating if this is the actual speaker")
    message: str = Field(..., description="Descriptive message about the operation")
    similarity_score: Optional[float] = Field(None, description="Similarity score (only for verification)")
    verified: Optional[bool] = Field(None, description="Verification result (only for verification)")
    data_stored: bool = Field(..., description="Whether the labeled data was successfully stored in the database")

# ----------------------------
# Helper Functions
# ----------------------------

async def validate_file_size(file: UploadFile, max_size_mb: float = 2.0) -> None:
    """
    Validate uploaded file size to prevent abuse and server overload.
    
    Args:
        file: The uploaded file to validate
        max_size_mb: Maximum allowed file size in megabytes (default: 2.0 MB)
    
    Raises:
        HTTPException: If file exceeds the size limit
    """
    # Read file content to check size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large: {file_size_mb:.2f}MB. Maximum allowed: {max_size_mb}MB"
        )
    
    # Reset file pointer for further processing
    await file.seek(0)
    print(f"📏 File size validation passed: {file_size_mb:.2f}MB (limit: {max_size_mb}MB)")

def compute_rms_energy(signal: torch.Tensor) -> float:
    """Compute RMS energy of the (mono or batched mono) signal."""
    try:
        if signal.dim() > 1:
            # Collapse batch/channel to mono energy
            mono = signal.mean(dim=0)
        else:
            mono = signal
        rms = torch.sqrt(torch.mean(mono.float() ** 2)).item()
        return float(rms)
    except Exception:
        return None

def estimate_snr_db(signal: torch.Tensor, sample_rate: int) -> float:
    """Estimate SNR in dB using frame energy percentiles (90th/10th)."""
    try:
        if signal.dim() > 1:
            mono = signal.mean(dim=0)
        else:
            mono = signal
        frame_size = max(1, sample_rate // 10)  # 100 ms
        hop_size = max(1, frame_size // 2)
        if mono.shape[-1] < frame_size * 2:
            # Too short; fallback using whole-signal vs. small epsilon
            rms = torch.sqrt(torch.mean(mono.float() ** 2)) + 1e-9
            noise = max(1e-9, (rms.item() * 0.1))
            import math
            return float(20.0 * math.log10((rms.item() + 1e-9) / noise))
        frames = mono.unfold(-1, frame_size, hop_size)
        frame_rms = torch.sqrt(torch.mean(frames.float() ** 2, dim=-1))
        low = torch.quantile(frame_rms, 0.10).item() + 1e-9
        high = torch.quantile(frame_rms, 0.90).item() + 1e-9
        import math
        snr_db = 20.0 * math.log10(high / low)
        return float(snr_db)
    except Exception:
        return None

def detect_number_of_speakers(
    signal: torch.Tensor,
    sample_rate: int,
    diarization: Optional[object] = None,
    diarization_count: Optional[int] = None,
) -> int:
    """
    Return the number of speakers without instantiating heavy/gated pipelines.

    Preferred usage: pass precomputed diarization info from the main pipeline.
    - If `diarization_count` is provided, it's used directly.
    - If `diarization` is provided, the function tries to infer unique labels from it.
    - Otherwise, returns 1 as a safe fallback.

    Note: This function intentionally avoids calling pyannote Pipeline to prevent
    gated downloads and repeated model loads. Compute diarization upstream and
    pass it here instead.
    """
    try:
        # 1) Direct count provided
        if diarization_count is not None:
            try:
                return max(1, int(diarization_count))
            except Exception:
                return 1

        # 2) Try to infer from provided diarization object/structure
        if diarization is not None:
            # pyannote Annotation-like: has itertracks(yield_label=True)
            if hasattr(diarization, "itertracks"):
                try:
                    labels = set()
                    for _, _, label in diarization.itertracks(yield_label=True):
                        labels.add(label)
                    return max(1, len(labels)) if labels else 1
                except Exception:
                    pass

            # Dictionary structure: {"segments": [{"label": "SPEAKER_00"}, ...]}
            if isinstance(diarization, dict):
                try:
                    segs = diarization.get("segments") or []
                    labels = set()
                    for seg in segs:
                        if isinstance(seg, dict) and "label" in seg:
                            labels.add(seg["label"]) 
                    if labels:
                        return max(1, len(labels))
                except Exception:
                    pass

            # List/tuple/set of labels or dicts with label
            if isinstance(diarization, (list, tuple, set)):
                try:
                    labels = set()
                    for item in diarization:
                        if isinstance(item, str):
                            labels.add(item)
                        elif isinstance(item, dict) and "label" in item:
                            labels.add(item["label"]) 
                    if labels:
                        return max(1, len(labels))
                except Exception:
                    pass

        # 3) Fallback: unknown diarization → default to 1
        return 1
    except Exception:
        return 1

async def save_upload_to_temp_file(file: UploadFile) -> str:
    """Save uploaded file to a temporary file and return the path."""
    try:
        # Create a temporary file with original extension
        file_extension = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Read the uploaded file content
            content = await file.read()
            # Write to temporary file
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"📁 Saved uploaded file to: {temp_file_path}")
        return temp_file_path
    
    except Exception as e:
        print(f"❌ Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {str(e)}")

def convert_audio_to_wav(input_path: str) -> str:
    """Convert audio file to WAV format that torchaudio can handle."""
    try:
        # First, try to load directly with torchaudio
        try:
            signal, fs = torchaudio.load(input_path)
            print(f"✅ Audio loaded directly: shape={signal.shape}, sample_rate={fs}")
            return input_path
        except Exception as direct_load_error:
            print(f"⚠️ Direct torchaudio load failed: {direct_load_error}")
            
            # Create output path
            output_path = input_path.replace(os.path.splitext(input_path)[1], "_converted.wav")
            
            # Try with pydub first (best for WebM and browser formats)
            try:
                from pydub import AudioSegment
                
                print("🔄 Trying pydub conversion...")
                # Load with pydub (handles WebM, MP4, OGG, etc.)
                audio = AudioSegment.from_file(input_path)
                
                print(f"📊 Pydub loaded: channels={audio.channels}, frame_rate={audio.frame_rate}, duration={len(audio)}ms")
                
                # Convert to mono and set sample rate to 16kHz (SpeechBrain standard)
                audio = audio.set_channels(1)  # Mono
                audio = audio.set_frame_rate(16000)  # 16kHz
                
                # Export as WAV
                audio.export(output_path, format="wav")
                print(f"✅ Audio converted with pydub: {input_path} -> {output_path}")
                
                # Verify the converted file can be loaded by torchaudio
                signal, fs = torchaudio.load(output_path)
                print(f"✅ Converted audio verified: shape={signal.shape}, sample_rate={fs}")
                
                return output_path
                
            except Exception as pydub_error:
                print(f"⚠️ Pydub conversion failed: {pydub_error}")
                
                # Try with librosa (robust for various formats)
                try:
                    import librosa
                    import numpy as np
                    
                    print("🔄 Trying librosa conversion...")
                    # Load with librosa (handles many formats)
                    data, samplerate = librosa.load(input_path, sr=None, mono=True)
                    
                    print(f"📊 Librosa loaded: shape={data.shape}, sample_rate={samplerate}")
                    
                    # Resample to 16kHz if needed (SpeechBrain standard)
                    if samplerate != 16000:
                        print(f"🔄 Resampling from {samplerate}Hz to 16000Hz...")
                        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                        samplerate = 16000
                    
                    # Convert to tensor
                    signal = torch.from_numpy(data).float().unsqueeze(0)
                    
                    # Save as WAV
                    torchaudio.save(output_path, signal, samplerate)
                    print(f"✅ Audio converted with librosa: {input_path} -> {output_path}")
                    return output_path
                    
                except Exception as librosa_error:
                    print(f"⚠️ Librosa conversion failed: {librosa_error}")
                    
                    # Final fallback to soundfile
                    try:
                        import soundfile as sf
                        import numpy as np
                        
                        print("🔄 Trying soundfile conversion...")
                        # Read with soundfile
                        data, samplerate = sf.read(input_path)
                        
                        print(f"📊 Soundfile loaded: shape={data.shape}, sample_rate={samplerate}")
                        
                        # Ensure it's mono
                        if len(data.shape) > 1:
                            data = np.mean(data, axis=1)
                        
                        # Convert to tensor
                        signal = torch.from_numpy(data).float().unsqueeze(0)
                        
                        # Resample to 16kHz if needed (SpeechBrain standard)
                        if samplerate != 16000:
                            print(f"🔄 Resampling from {samplerate}Hz to 16000Hz...")
                            resampler = torchaudio.transforms.Resample(samplerate, 16000)
                            signal = resampler(signal)
                            samplerate = 16000
                        
                        # Save as WAV
                        torchaudio.save(output_path, signal, samplerate)
                        print(f"✅ Audio converted with soundfile: {input_path} -> {output_path}")
                        return output_path
                        
                    except Exception as sf_error:
                        print("❌ All conversion methods failed!")
                        print(f"   - Direct torchaudio: {direct_load_error}")
                        print(f"   - Pydub: {pydub_error}")
                        print(f"   - Librosa: {librosa_error}")
                        print(f"   - Soundfile: {sf_error}")
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Unable to process audio format. File type: {os.path.splitext(input_path)[1]}. Please try recording again or use a different browser."
                        )
    
    except Exception as e:
        print(f"❌ Audio conversion error: {e}")
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")

async def background_retrain_and_store(
    user_id: str, 
    verification_embedding: torch.Tensor, 
    best_score: float,
    verification_signal: torch.Tensor = None,
    fs: int = 16000
):
    """
    Background task to handle EWMA adaptation and voice sample storage
    after sending response to frontend for optimal performance.
    """
    try:
        from main import CONFIG
        
        # Apply EWMA adaptation if score is above adaptation threshold
        if (CONFIG.EWMA_ENABLED and 
            best_score >= CONFIG.EWMA_ADAPTATION_THRESHOLD and 
            user_id):
            
            print(f"🔄 Applying EWMA adaptation to {user_id}'s voiceprint...")
            
            try:
                success = update_user_embedding_ewma(
                    user_id=user_id,
                    new_embedding=verification_embedding,
                    alpha=CONFIG.EWMA_LEARNING_RATE
                )
                if success:
                    print(f"🔄 EWMA Update: {user_id} voiceprint adapted (α={CONFIG.EWMA_LEARNING_RATE:.3f})")
                else:
                    print(f"⚠️ EWMA update failed for {user_id}")
            except Exception as e:
                print(f"❌ EWMA update error: {e}")
        
        # Store voice sample regardless of score in voice_data collection
        if user_id:
            try:
                # Calculate audio duration
                audio_duration = verification_signal.shape[-1] / fs if verification_signal is not None else None
                
                # Prepare audio metadata
                audio_info = {
                    "sample_rate": int(fs),
                    "audio_shape": list(verification_signal.shape) if verification_signal is not None else None,
                    "processing_pipeline": "VAD+Diarization+Separation",
                    "verification_type": "identification"
                }
                
                success = store_voice_sample(
                    user_id=user_id,
                    embedding=verification_embedding,
                    similarity_score=best_score,
                    audio_duration=audio_duration,
                    audio_info=audio_info
                )
                
                if success:
                    # Generate a sample_id for logging (similar to what was in the original)
                    import hashlib
                    import time
                    sample_id = hashlib.md5(f"{user_id}_{time.time()}".encode()).hexdigest()[:24]
                    print(f"💾 Voice sample stored: {user_id} (score: {best_score:.4f}, sample_id: {sample_id})")
                else:
                    print(f"⚠️ Voice sample storage failed for {user_id}")
                    
            except Exception as e:
                print(f"❌ Voice sample storage error: {e}")
                
    except Exception as e:
        print(f"❌ Background retrain/store task error: {e}")

def cleanup_temp_file(file_path: str):
    """Clean up temporary file."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            print(f"🗑️ Cleaned up temporary file: {file_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not clean up temporary file {file_path}: {e}")

def apply_advanced_enhancement_pipeline(signal: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Apply advanced audio processing pipeline:
    Mic Input → VAD → Speech Separation (Sepformer) → Enhancement → Clean Audio
    
    This function implements a comprehensive audio processing pipeline:
    1. Voice Activity Detection (VAD) using SpeechBrain
    2. Speech Separation using SpeechBrain Sepformer (handles overlapping speakers)
    3. Optional Enhancement for noise reduction
    4. Audio quality improvement for better embedding extraction
    
    Args:
        signal: Input audio tensor of shape [batch_size, samples] or [samples]
        sample_rate: Sample rate of the audio (default: 16000)
    
    Returns:
        Enhanced audio tensor with the same shape as input
    """
    try:
        # Import SpeechBrain modules
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.VAD import VAD
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        
        original_shape = signal.shape
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)
        
        # Check if signal is too short for meaningful processing
        min_length = sample_rate * 0.5  # At least 0.5 seconds
        if signal.shape[1] < min_length:
            print(f"⚠️ Signal too short ({signal.shape[1]} samples) for advanced processing, returning original")
            if len(original_shape) == 1:
                return signal.squeeze(0)
            return signal
        
        enhanced_batch = []
        
        for i in range(signal.shape[0]):
            audio = signal[i]
            print(f"🎙️ Processing audio sample {i+1}/{signal.shape[0]}")
            
            # Step 1: Voice Activity Detection
            print("🔍 Step 1: Applying Voice Activity Detection...")
            try:
                # Use cached VAD model
                vad_model = model_manager.get_vad_model()
                if vad_model is None:
                    print("⚠️ VAD model not cached, loading on-demand...")
                    vad_model = VAD.from_hparams(
                        source="speechbrain/vad-crdnn-libriparty",
                        savedir="pretrained_models/vad"
                    )
                
                # Create temporary file for VAD processing
                import tempfile
                import torchaudio
                temp_file = None
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        temp_file = f.name
                    
                    # Save audio to temporary file (move to CPU first)
                    audio_for_save = audio.cpu() if audio.is_cuda else audio
                    torchaudio.save(temp_file, audio_for_save.unsqueeze(0), sample_rate)
                    
                    # Get speech segments using VAD
                    speech_segments = vad_model.get_speech_segments(temp_file)
                    
                    if len(speech_segments) > 0:
                        # Extract speech regions
                        speech_audio_parts = []
                        for segment in speech_segments:
                            start_sample = int(segment[0] * sample_rate)
                            end_sample = int(segment[1] * sample_rate)
                            if end_sample > start_sample and end_sample <= len(audio):
                                speech_audio_parts.append(audio[start_sample:end_sample])
                        
                        if speech_audio_parts:
                            vad_audio = torch.cat(speech_audio_parts, dim=0)
                            print(f"✅ VAD: Extracted {len(speech_audio_parts)} speech segments, duration: {len(vad_audio)/sample_rate:.2f}s")
                        else:
                            print("⚠️ VAD: No valid speech segments, using original audio")
                            vad_audio = audio
                    else:
                        print("⚠️ VAD: No speech detected, using original audio")
                        vad_audio = audio
                
                finally:
                    # Cleanup temporary file
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.unlink(temp_file)
                        except Exception:
                            pass
                    
            except Exception as vad_error:
                print(f"⚠️ VAD failed: {vad_error}, using original audio")
                vad_audio = audio
            
            # Step 2: Speech Separation using Sepformer
            print("🔄 Step 2: Applying Speech Separation (Sepformer)...")
            try:
                # Use cached Sepformer separation model
                separator = model_manager.get_separator(num_speakers=2)
                if separator is None:
                    print("⚠️ Sepformer model not cached, loading on-demand...")
                    separator = SepformerSeparation.from_hparams(
                        source="speechbrain/sepformer-wsj02mix", 
                        savedir='pretrained_models/sepformer-wsj02mix'
                    )
                
                # Prepare audio for separation
                if len(vad_audio.shape) == 1:
                    separation_input = vad_audio.unsqueeze(0)  # Add batch dimension
                else:
                    separation_input = vad_audio
                
                # Apply separation (separates overlapping speakers)
                separated_sources = separator.separate_batch(separation_input)
                print(f"📊 Separation output shape: {separated_sources.shape}")
                
                # separated_sources shape: [batch, samples, num_sources]
                # Choose the best source (highest energy or best quality)
                if separated_sources.shape[2] > 1:
                    # Calculate energy for each source
                    source_energies = []
                    for src_idx in range(separated_sources.shape[2]):
                        source = separated_sources[0, :, src_idx]
                        energy = torch.sum(source ** 2).item()
                        source_energies.append(energy)
                    
                    # Select source with highest energy (likely the main speaker)
                    best_source_idx = torch.argmax(torch.tensor(source_energies)).item()
                    separated_audio = separated_sources[0, :, best_source_idx]
                    print(f"✅ Separation: Selected source {best_source_idx+1} with highest energy")
                else:
                    separated_audio = separated_sources[0, :, 0]
                    print("✅ Separation: Using single separated source")
                
            except Exception as separation_error:
                print(f"⚠️ Separation failed: {separation_error}, using VAD output")
                separated_audio = vad_audio
            
            # Step 3: Optional Enhancement for noise reduction
            print("🔧 Step 3: Applying Enhancement...")
            try:
                # Use cached enhancement model
                enhancer = model_manager.get_enhancer()
                if enhancer is None:
                    print("⚠️ Enhancement model not cached, trying to load on-demand...")
                    try:
                        enhancer = SpectralMaskEnhancement.from_hparams(
                            source="speechbrain/metricgan-plus-voicebank",
                            savedir="pretrained_models/metricgan-plus",
                            run_opts={"device": DEVICE}
                        )
                    except Exception as fallback_error:
                        print(f"⚠️ On-demand enhancement loading also failed: {fallback_error}")
                        print("🔄 Skipping enhancement, using separated audio as-is")
                        enhanced_audio = separated_audio
                        enhancer = None
                
                if enhancer is not None:
                    # Prepare audio for enhancement and move to GPU if available
                    if len(separated_audio.shape) == 1:
                        enhancement_input = separated_audio.unsqueeze(0)
                    else:
                        enhancement_input = separated_audio
                    
                    # Move audio to GPU if available
                    if DEVICE.type == "cuda":
                        enhancement_input = enhancement_input.to(DEVICE)
                    
                    # Apply enhancement
                    enhanced_audio = enhancer.enhance_batch(enhancement_input)
                    
                    # Move results back to CPU and remove batch dimension if added
                    if DEVICE.type == "cuda":
                        enhanced_audio = enhanced_audio.cpu()
                    
                    if enhanced_audio.shape[0] == 1:
                        enhanced_audio = enhanced_audio.squeeze(0)
                    
                    print(f"✅ Enhancement: Processed audio shape={enhanced_audio.shape}")
                else:
                    # Enhancement model not available, use separated audio as-is
                    enhanced_audio = separated_audio
                    print("🔄 Enhancement skipped - using separated audio directly")
                
            except Exception as enhancement_error:
                print(f"⚠️ Enhancement failed: {enhancement_error}, using separated audio")
                enhanced_audio = separated_audio
            
            # Step 4: Ensure correct length and format
            target_length = audio.shape[0]
            if len(enhanced_audio) > target_length:
                # Trim to original length
                enhanced_audio = enhanced_audio[:target_length]
            elif len(enhanced_audio) < target_length:
                # Pad with zeros to original length
                padding = target_length - len(enhanced_audio)
                enhanced_audio = torch.nn.functional.pad(enhanced_audio, (0, padding))
            
            enhanced_batch.append(enhanced_audio)
        
        # Stack results
        result = torch.stack(enhanced_batch)
        
        # Return with original shape
        if len(original_shape) == 1:
            result = result.squeeze(0)
            
        print("🎯 Advanced pipeline completed: VAD → Separation → Enhancement → Clean Audio")
        return result
        
    except Exception as e:
        print(f"⚠️ Advanced pipeline failed, returning original signal: {e}")
        return signal


def apply_speaker_aware_enhancement_pipeline(signal: torch.Tensor, enrolled_embeddings: dict = None, sample_rate: int = 16000) -> torch.Tensor:
    """
    Apply speaker-aware enhancement pipeline with source selection:
    Mic Input → VAD → Speech Separation → Source Selection → Enhancement → Clean Audio
    
    This advanced pipeline can:
    1. Detect and separate multiple speakers
    2. Select the audio source that best matches enrolled users
    3. Apply targeted enhancement
    
    Args:
        signal: Input audio tensor of shape [batch_size, samples] or [samples]
        enrolled_embeddings: Dict of user_id -> embedding for source selection
        sample_rate: Sample rate of the audio (default: 16000)
    
    Returns:
        Enhanced audio tensor with the same shape as input
    """
    try:
        # Import SpeechBrain modules
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import SpeakerRecognition
        
        original_shape = signal.shape
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)
        
        # Check if signal is too short for meaningful processing
        min_length = sample_rate * 0.5  # At least 0.5 seconds
        if signal.shape[1] < min_length:
            print(f"⚠️ Signal too short ({signal.shape[1]} samples) for advanced processing, returning original")
            if len(original_shape) == 1:
                return signal.squeeze(0)
            return signal
        
        enhanced_batch = []
        
        for i in range(signal.shape[0]):
            audio = signal[i]
            print(f"🎙️ Processing audio sample {i+1}/{signal.shape[0]}")
            
            # Step 1: Speech Separation using Sepformer
            print("🔄 Step 1: Applying Speech Separation (Multi-Speaker)...")
            try:
                # Initialize Sepformer separation model with GPU support
                separator = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-wsj02mix", 
                    savedir='pretrained_models/sepformer-wsj02mix',
                    run_opts={"device": DEVICE}
                )
                
                # Prepare audio for separation
                if len(audio.shape) == 1:
                    separation_input = audio.unsqueeze(0)  # Add batch dimension
                else:
                    separation_input = audio
                
                # Apply separation (handles overlapping speakers)
                separated_sources = separator.separate_batch(separation_input)
                print(f"📊 Separated into {separated_sources.shape[2]} sources")
                
                # Step 2: Source Selection based on enrolled users
                if enrolled_embeddings and len(enrolled_embeddings) > 0:
                    print("🎯 Step 2: Selecting best source using enrolled embeddings...")
                    
                    # Initialize speaker recognition for source evaluation with GPU support
                    speaker_encoder = SpeakerRecognition.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="pretrained_models/spkrec-ecapa-voxceleb",
                        run_opts={"device": DEVICE}
                    )
                    
                    best_source = None
                    best_similarity = -1.0
                    best_user = None
                    
                    # Evaluate each separated source against enrolled users
                    for src_idx in range(separated_sources.shape[2]):
                        source_audio = separated_sources[0, :, src_idx].unsqueeze(0)
                        
                        try:
                            # Extract embedding from this source
                            source_embedding = speaker_encoder.encode_batch(source_audio)
                            
                            # Compare against all enrolled users
                            for user_id, enrolled_embedding in enrolled_embeddings.items():
                                similarity = torch.nn.functional.cosine_similarity(
                                    source_embedding.squeeze(), 
                                    enrolled_embedding.squeeze(), 
                                    dim=0
                                ).item()
                                
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_source = src_idx
                                    best_user = user_id
                        
                        except Exception as embed_error:
                            print(f"⚠️ Failed to extract embedding from source {src_idx}: {embed_error}")
                    
                    if best_source is not None:
                        selected_audio = separated_sources[0, :, best_source]
                        print(f"✅ Selected source {best_source+1} (similarity {best_similarity:.3f} with {best_user})")
                    else:
                        # Fallback: use highest energy source
                        source_energies = [torch.sum(separated_sources[0, :, i] ** 2).item() 
                                         for i in range(separated_sources.shape[2])]
                        best_source = torch.argmax(torch.tensor(source_energies)).item()
                        selected_audio = separated_sources[0, :, best_source]
                        print(f"✅ Fallback: Selected source {best_source+1} (highest energy)")
                
                else:
                    # No enrolled embeddings: select highest energy source
                    print("🔧 Step 2: Selecting highest energy source...")
                    source_energies = [torch.sum(separated_sources[0, :, i] ** 2).item() 
                                     for i in range(separated_sources.shape[2])]
                    best_source = torch.argmax(torch.tensor(source_energies)).item()
                    selected_audio = separated_sources[0, :, best_source]
                    print(f"✅ Selected source {best_source+1} with highest energy")
                
            except Exception as separation_error:
                print(f"⚠️ Separation failed: {separation_error}, using original audio")
                selected_audio = audio
            
            # Step 3: Final length adjustment
            target_length = audio.shape[0]
            if len(selected_audio) > target_length:
                selected_audio = selected_audio[:target_length]
            elif len(selected_audio) < target_length:
                padding = target_length - len(selected_audio)
                selected_audio = torch.nn.functional.pad(selected_audio, (0, padding))
            
            enhanced_batch.append(selected_audio)
        
        # Stack results
        result = torch.stack(enhanced_batch)
        
        # Return with original shape
        if len(original_shape) == 1:
            result = result.squeeze(0)
            
        print("🎯 Speaker-aware pipeline completed: Separation → Source Selection → Clean Audio")
        return result
        
    except Exception as e:
        print(f"⚠️ Speaker-aware pipeline failed, returning original signal: {e}")
        return signal


def calculate_embedding_quality(embeddings: list) -> tuple:
    """Calculate embedding quality metrics from multiple samples."""
    if len(embeddings) < 2:
        return "single_sample", 0.0
    
    import torch
    
    # Calculate pairwise similarities between samples
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            emb1 = embeddings[i] / embeddings[i].norm(dim=-1, keepdim=True)
            emb2 = embeddings[j] / embeddings[j].norm(dim=-1, keepdim=True)
            sim = torch.cosine_similarity(emb1.squeeze(), emb2.squeeze(), dim=0)
            similarities.append(sim.item())
    
    avg_similarity = sum(similarities) / len(similarities)
    
    # Determine quality based on consistency
    if avg_similarity >= 0.85:
        quality = "excellent"
    elif avg_similarity >= 0.75:
        quality = "good"
    elif avg_similarity >= 0.65:
        quality = "fair"
    else:
        quality = "poor"
    
    return quality, avg_similarity

def check_volume(signal: torch.Tensor, energy_threshold: float = 0.0005) -> bool:
    """Checks if the audio signal has sufficient energy (not silent/too quiet)."""
    rms_energy = torch.sqrt(torch.mean(signal**2))
    print(f"🎤 RMS Energy: {rms_energy.item():.6f}")
    return rms_energy.item() > energy_threshold

def check_speech_presence(signal: torch.Tensor, sample_rate: int = 16000, vad_ratio_threshold: float = 0.15, min_speech_duration: float = 4.0) -> bool:
    """
    Checks if the audio contains enough speech using VAD ratio and minimum duration requirements.
    
    Args:
        signal: Input audio tensor
        sample_rate: Sample rate of the audio
        vad_ratio_threshold: Minimum ratio of speech to total audio (default: 0.15)
        min_speech_duration: Minimum total speech duration in seconds (default: 4.0)
    
    Returns:
        bool: True if audio meets both ratio and duration requirements
    """
    try:
        from speechbrain.inference.VAD import VAD
        
        # Initialize VAD model with GPU support
        vad_model = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad",
            run_opts={"device": DEVICE}
        )
        
        # Create temporary file for VAD processing
        import tempfile
        import torchaudio
        temp_file = None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                temp_file = f.name
            
            # Save audio to temporary file for VAD processing
            if len(signal.shape) > 1:
                audio_for_save = signal.cpu()
            else:
                audio_for_save = signal.unsqueeze(0).cpu()
            
            torchaudio.save(temp_file, audio_for_save, sample_rate)
            
            # Get speech segments using VAD
            speech_segments = vad_model.get_speech_segments(temp_file)
            
            # Calculate total speech duration
            total_speech_duration = sum([end - start for start, end in speech_segments])
            total_duration = signal.shape[-1] / sample_rate
            
            vad_ratio = total_speech_duration / total_duration if total_duration > 0 else 0
            
            # Check both ratio and minimum duration requirements
            ratio_check = vad_ratio > vad_ratio_threshold
            duration_check = total_speech_duration >= min_speech_duration
            
            print(f"🗣️ Speech Analysis: {total_speech_duration:.2f}s speech / {total_duration:.2f}s total (ratio: {vad_ratio:.3f})")
            print(f"📊 Requirements: Ratio ≥ {vad_ratio_threshold:.3f} {'✅' if ratio_check else '❌'}, Duration ≥ {min_speech_duration:.1f}s {'✅' if duration_check else '❌'}")
            
            return ratio_check and duration_check
        
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
    
    except Exception as e:
        print(f"⚠️ VAD ratio check failed: {e}")
        # Fallback to basic energy-based speech detection
        frame_size = sample_rate // 10  # 100ms frames
        if signal.shape[-1] < frame_size * 2:
            return True  # Too short to analyze, assume valid
        
        frames = signal.unfold(-1, frame_size, frame_size // 2)
        frame_energies = torch.sqrt(torch.mean(frames**2, dim=-1))
        energy_variance = torch.var(frame_energies).item()
        
        # If energy varies enough, assume speech is present
        speech_detected = energy_variance > 0.0001
        
        # Check minimum duration (fallback method)
        total_duration = signal.shape[-1] / sample_rate
        duration_check = total_duration >= min_speech_duration
        
        print(f"🗣️ Fallback speech detection: {'✅ Speech detected' if speech_detected else '❌ No speech'} (energy variance: {energy_variance:.6f})")
        print(f"📊 Duration check: {total_duration:.2f}s ≥ {min_speech_duration:.1f}s {'✅' if duration_check else '❌'}")
        
        return speech_detected and duration_check

def audio_quality_gatekeeper(signal: torch.Tensor, sample_rate: int = 16000) -> tuple[bool, str]:
    """
    Comprehensive audio quality gatekeeper that validates audio before processing.
    
    Returns:
        (is_valid, error_message): Tuple indicating if audio passes quality checks
    """
    print("🚦 Running Audio Quality Gatekeeper...")
    
    # Check 1: Volume/Energy Check
    print("🔍 Check 1: Volume and Energy Analysis...")
    if not check_volume(signal):
        return False, "Audio too quiet or silent. Please speak clearly and closer to the microphone."
    print("✅ Volume check passed")
    
    # Check 2: Speech Presence Check
    print("🔍 Check 2: Speech Presence Analysis...")
    if not check_speech_presence(signal, sample_rate, min_speech_duration=4.0):
        return False, "Insufficient speech content. Please record at least 4 seconds of clear speech."
    print("✅ Speech presence check passed")
    
    print("🎉 Audio quality validation passed! Audio is ready for processing.")
    return True, "Audio quality validation successful"

def find_optimal_speech_segment(signal: torch.Tensor, sample_rate: int = 16000, target_duration: float = 12.0, min_duration: float = 4.0) -> torch.Tensor:
    """
    Find the optimal 10-12 second segment with highest speech density using sliding window analysis.
    
    Args:
        signal: Input audio tensor
        sample_rate: Audio sample rate
        target_duration: Target segment duration in seconds (default 12s)
        min_duration: Minimum required duration in seconds (default 4s)
    
    Returns:
        Optimal audio segment tensor
    """
    print(f"🎯 Finding optimal {target_duration}s speech segment from {signal.shape[-1]/sample_rate:.1f}s audio...")
    
    audio_duration = signal.shape[-1] / sample_rate
    
    # If audio is already shorter than target, return as-is (if above minimum)
    if audio_duration <= target_duration:
        if audio_duration >= min_duration:
            print(f"✅ Audio duration ({audio_duration:.1f}s) is within target range, using full audio")
            return signal
        else:
            print(f"⚠️ Audio too short ({audio_duration:.1f}s < {min_duration}s minimum)")
            return signal
    
    try:
        # Step 1: Run VAD to find speech segments
        print("🔍 Step 1: Running VAD analysis for speech detection...")
        speech_segments = []
        
        # Use existing VAD functionality
        try:
            import tempfile
            import os
            
            # Save audio to temporary file for VAD processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                temp_file = f.name
            
            # Ensure audio is in correct format for saving
            if len(signal.shape) > 1:
                audio_to_save = signal.cpu()
            else:
                audio_to_save = signal.unsqueeze(0).cpu()
            
            torchaudio.save(temp_file, audio_to_save, sample_rate)
            
            # Try to use pyannote VAD if available
            try:
                from pyannote.audio import Pipeline
                
                # Load VAD pipeline
                if not hasattr(find_optimal_speech_segment, 'vad_pipeline'):
                    vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
                    vad_pipeline = vad_pipeline.to(DEVICE)
                    find_optimal_speech_segment.vad_pipeline = vad_pipeline
                
                vad_output = find_optimal_speech_segment.vad_pipeline(temp_file)
                for speech in vad_output.get_timeline():
                    speech_segments.append((speech.start, speech.end))
                
                total_speech = sum([end - start for start, end in speech_segments])
                print(f"✅ VAD found {len(speech_segments)} speech segments, {total_speech:.2f}s total speech")
                
            except Exception as vad_error:
                print(f"⚠️ Pyannote VAD failed: {vad_error}")
                # Fallback: assume entire audio is speech
                speech_segments = [(0, audio_duration)]
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except Exception:
                pass
                
        except Exception as e:
            print(f"⚠️ VAD processing failed: {e}, assuming entire audio is speech")
            speech_segments = [(0, audio_duration)]
        
        # Step 2: Sliding window analysis
        print(f"🔍 Step 2: Sliding window analysis (window size: {target_duration}s)...")
        
        window_size_samples = int(target_duration * sample_rate)
        step_size_samples = int(0.5 * sample_rate)  # 0.5 second steps
        
        best_window_start = 0
        best_speech_density = 0
        best_speech_duration = 0
        best_quality_score = 0
        
        windows_analyzed = 0
        
        # Slide window across the audio
        for start_sample in range(0, signal.shape[-1] - window_size_samples + 1, step_size_samples):
            end_sample = start_sample + window_size_samples
            window_start_time = start_sample / sample_rate
            window_end_time = end_sample / sample_rate
            
            # Calculate speech coverage in this window
            speech_in_window = 0
            for seg_start, seg_end in speech_segments:
                # Find overlap between speech segment and current window
                overlap_start = max(seg_start, window_start_time)
                overlap_end = min(seg_end, window_end_time)
                
                if overlap_end > overlap_start:
                    speech_in_window += overlap_end - overlap_start
            
            # Calculate speech density (percentage of window that contains speech)
            speech_density = speech_in_window / target_duration
            
            # Quality scoring: prioritize both density and absolute speech content
            quality_score = speech_density * 1000 + speech_in_window * 10
            
            windows_analyzed += 1
            
            if quality_score > best_quality_score:
                best_quality_score = quality_score
                best_speech_density = speech_density
                best_speech_duration = speech_in_window
                best_window_start = start_sample
        
        print(f"📊 Analyzed {windows_analyzed} windows")
        
        # Step 3: Extract the optimal segment
        optimal_start_sample = best_window_start
        optimal_end_sample = min(best_window_start + window_size_samples, signal.shape[-1])
        optimal_segment = signal[..., optimal_start_sample:optimal_end_sample]
        
        optimal_start_time = optimal_start_sample / sample_rate
        optimal_end_time = optimal_end_sample / sample_rate
        optimal_duration = (optimal_end_sample - optimal_start_sample) / sample_rate
        
        print("🏆 Optimal segment found:")
        print(f"   📍 Time range: {optimal_start_time:.1f}s - {optimal_end_time:.1f}s")
        print(f"   ⏱️ Duration: {optimal_duration:.1f}s")
        print(f"   🗣️ Speech density: {best_speech_density:.1%}")
        print(f"   📊 Speech content: {best_speech_duration:.1f}s")
        print(f"   🎯 Quality score: {best_quality_score:.1f}")
        
        return optimal_segment
        
    except Exception as e:
        print(f"⚠️ Optimal segment selection failed: {e}")
        print(f"📋 Fallback: Using first {target_duration}s of audio")
        
        # Fallback: Use first N seconds
        fallback_samples = min(int(target_duration * sample_rate), signal.shape[-1])
        return signal[..., :fallback_samples]

def enhanced_pause_robust_processing(signal: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Enhanced processing specifically designed to handle speech with long pauses.
    
    Strategy:
    1. Detect all speech segments using VAD
    2. Filter out very short segments (< 0.5s)
    3. Concatenate only substantial speech segments
    4. Ensure minimum continuous speech duration
    5. Optimize length for best embedding quality
    
    Args:
        signal: Input audio tensor
        sample_rate: Audio sample rate (default: 16000)
    
    Returns:
        Pause-robust audio tensor with continuous speech
    """
    print("🔇 Applying enhanced pause-robust speech processing...")
    
    try:
        audio_duration = signal.shape[-1] / sample_rate
        print(f"📊 Original audio: {audio_duration:.1f}s")
        
        # If audio is very short, return as-is
        if audio_duration < 2.0:
            print(f"⚠️ Audio too short ({audio_duration:.1f}s) for pause processing")
            return signal
        
        # Step 1: Voice Activity Detection using SpeechBrain VAD
        print("🔍 Step 1: Running VAD for speech segment detection...")
        speech_segments = []
        
        try:
            from speechbrain.inference.VAD import VAD
            
            # Initialize VAD model with GPU support
            vad_model = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                savedir="pretrained_models/vad",
                run_opts={"device": DEVICE}
            )
            
            # Create temporary file for VAD processing
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                temp_file = f.name
            
            # Save audio for VAD processing
            if len(signal.shape) > 1:
                audio_to_save = signal.cpu()
            else:
                audio_to_save = signal.unsqueeze(0).cpu()
            
            torchaudio.save(temp_file, audio_to_save, sample_rate)
            
            # Get speech segments
            speech_segments_raw = vad_model.get_speech_segments(temp_file)
            
            # Convert to our format
            for start_time, end_time in speech_segments_raw:
                speech_segments.append((start_time, end_time))
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except Exception:
                pass
                
            print(f"✅ VAD detected {len(speech_segments)} speech segments")
            
        except Exception as vad_error:
            print(f"⚠️ SpeechBrain VAD failed: {vad_error}")
            
            # Fallback: Try pyannote VAD if available
            try:
                from pyannote.audio import Pipeline
                
                # Load VAD pipeline
                if not hasattr(enhanced_pause_robust_processing, 'vad_pipeline'):
                    vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
                    vad_pipeline = vad_pipeline.to(DEVICE)
                    enhanced_pause_robust_processing.vad_pipeline = vad_pipeline
                
                # Create temp file for pyannote
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    temp_file = f.name
                
                if len(signal.shape) > 1:
                    audio_to_save = signal.cpu()
                else:
                    audio_to_save = signal.unsqueeze(0).cpu()
                
                torchaudio.save(temp_file, audio_to_save, sample_rate)
                
                vad_output = enhanced_pause_robust_processing.vad_pipeline(temp_file)
                for speech in vad_output.get_timeline():
                    speech_segments.append((speech.start, speech.end))
                
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
                    
                print(f"✅ Pyannote VAD detected {len(speech_segments)} speech segments")
                
            except Exception as pyannote_error:
                print(f"⚠️ Pyannote VAD also failed: {pyannote_error}")
                # Ultimate fallback: assume entire audio is speech
                speech_segments = [(0, audio_duration)]
                print("📋 Using fallback: treating entire audio as speech")
        
        if not speech_segments:
            print("⚠️ No speech segments detected")
            return signal
        
        # Step 2: Filter and process speech segments
        print("🔍 Step 2: Filtering and processing speech segments...")
        filtered_segments = []
        total_speech_duration = 0
        min_segment_duration = 0.5  # Minimum 0.5 seconds for substantial speech
        
        for i, (start_time, end_time) in enumerate(speech_segments):
            duration = end_time - start_time
            
            # Only keep substantial speech segments
            if duration >= min_segment_duration:
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                start_sample = max(0, start_sample)
                end_sample = min(signal.shape[-1], end_sample)
                
                if end_sample > start_sample:
                    segment_audio = signal[..., start_sample:end_sample]
                    filtered_segments.append(segment_audio)
                    total_speech_duration += duration
                    print(f"   ✅ Segment {i+1}: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s)")
                else:
                    print(f"   ⏭️ Skipped invalid segment {i+1}: {start_time:.1f}s-{end_time:.1f}s")
            else:
                print(f"   ⏭️ Skipped short segment {i+1}: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s < {min_segment_duration}s)")
        
        if not filtered_segments:
            print("⚠️ No substantial speech segments found after filtering")
            return signal
        
        # Step 3: Concatenate speech segments (removes all pauses)
        print("🔗 Step 3: Concatenating speech segments...")
        continuous_speech = torch.cat(filtered_segments, dim=-1)
        
        print(f"🎯 Concatenated speech: {total_speech_duration:.1f}s of continuous speech from {audio_duration:.1f}s original")
        print(f"📊 Speech efficiency: {total_speech_duration/audio_duration:.1%} (removed {audio_duration-total_speech_duration:.1f}s of pauses)")
        
        # Step 4: Quality validation
        print("🔍 Step 4: Validating continuous speech quality...")
        min_required_speech = 8.0  # seconds
        if total_speech_duration < min_required_speech:
            print(f"⚠️ Insufficient continuous speech: {total_speech_duration:.1f}s < {min_required_speech}s required")
            print("📋 Returning original audio (may contain pauses but meets minimum duration)")
            return signal
        
        # Step 5: Optimize length for embedding quality
        print("🎯 Step 5: Optimizing length for embedding quality...")
        target_samples_min = int(10.0 * sample_rate)  # 10 seconds minimum
        target_samples_max = int(12.0 * sample_rate)  # 12 seconds maximum
        
        current_samples = continuous_speech.shape[-1]
        
        if current_samples < target_samples_min:
            # If we have less than 10s, use what we have (it's at least 8s from step 4)
            print(f"📋 Using all {total_speech_duration:.1f}s of continuous speech (less than 10s target)")
            final_speech = continuous_speech
            
        elif current_samples <= target_samples_max:
            # Perfect range: 10-12 seconds
            print(f"✅ Perfect length: {total_speech_duration:.1f}s of continuous speech")
            final_speech = continuous_speech
            
        else:
            # More than 12s: intelligently select the best 12s
            print(f"✂️ Selecting optimal 12s from {total_speech_duration:.1f}s of continuous speech...")
            
            # Strategy: Take from the middle-to-end (often contains the most natural speech)
            # Skip the first 1-2 seconds (potential startup artifacts) but keep the substantial middle part
            skip_start_samples = min(int(1.0 * sample_rate), current_samples // 10)  # Skip max 1s or 10% of audio
            optimal_start = skip_start_samples
            optimal_end = optimal_start + target_samples_max
            
            if optimal_end <= current_samples:
                final_speech = continuous_speech[..., optimal_start:optimal_end]
                final_duration = target_samples_max / sample_rate
                print(f"✅ Selected 12.0s segment from continuous speech (skipped first {skip_start_samples/sample_rate:.1f}s)")
            else:
                # Take the last 12 seconds
                final_speech = continuous_speech[..., -target_samples_max:]
                final_duration = target_samples_max / sample_rate
                print("✅ Selected final 12.0s from continuous speech")
        
        final_duration = final_speech.shape[-1] / sample_rate
        
        # Step 6: Final quality check
        print("🔍 Step 6: Final quality validation...")
        
        # Check RMS energy to ensure we didn't accidentally select silence
        rms_energy = torch.sqrt(torch.mean(final_speech**2))
        energy_threshold = 0.001  # Minimum energy for valid speech
        
        if rms_energy.item() < energy_threshold:
            print(f"⚠️ Final segment has low energy ({rms_energy.item():.6f}), using original audio")
            return signal
        
        print("✅ Enhanced pause-robust processing complete:")
        print(f"   📊 Original: {audio_duration:.1f}s (with pauses)")
        print(f"   🗣️ Speech detected: {total_speech_duration:.1f}s")
        print(f"   🎯 Final output: {final_duration:.1f}s pure speech")
        print(f"   ⚡ Energy level: {rms_energy.item():.6f}")
        print(f"   🔇 Pause removal: {((audio_duration - total_speech_duration) / audio_duration * 100):.1f}% of original was pauses")
        
        return final_speech
        
    except Exception as e:
        print(f"⚠️ Enhanced pause-robust processing failed: {e}")
        print("📋 Fallback: returning original audio")
        return signal

def apply_enrollment_speaker_selection(signal: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Enhanced enrollment processing that selects the best speaker segment for enrollment.
    Uses VAD, diarization, and quality assessment to pick the optimal speaker.
    
    Quality Assessment Metrics:
    - RMS Energy: Overall volume level
    - Energy Dynamics: Speech variability and liveliness
    - SNR Estimation: Signal-to-noise ratio
    - Duration Bonus: Preference for longer speech segments
    
    Args:
        signal: Input audio tensor
        sample_rate: Sample rate of the audio (default: 16000)
    
    Returns:
        Enhanced audio tensor with the best quality speaker segment
    """
    try:
        print("🎯 Applying enrollment-optimized speaker selection...")
        
        original_shape = signal.shape
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)
        
        # Check if signal is too short for meaningful processing
        min_length = sample_rate * 1.0  # At least 1 second for diarization
        if signal.shape[1] < min_length:
            print(f"⚠️ Signal too short ({signal.shape[1]} samples) for speaker selection, using original")
            if len(original_shape) == 1:
                return signal.squeeze(0)
            return signal
        
        enhanced_batch = []
        
        for i in range(signal.shape[0]):
            audio = signal[i]
            print(f"🎙️ Processing enrollment audio sample {i+1}/{signal.shape[0]}")
            
            # Step 1: Voice Activity Detection
            print("🔍 Step 1: Applying VAD for speech detection...")
            speech_segments = []
            try:
                from speechbrain.inference.VAD import VAD
                
                vad_model = VAD.from_hparams(
                    source="speechbrain/vad-crdnn-libriparty",
                    savedir="pretrained_models/vad",
                    run_opts={"device": DEVICE}
                )
                
                # Create temporary file for VAD processing
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        temp_file = f.name
                    
                    # Save audio (ensure it's on CPU)
                    audio_to_save = audio.cpu() if audio.is_cuda else audio
                    torchaudio.save(temp_file, audio_to_save.unsqueeze(0), sample_rate)
                    
                    # Get speech segments
                    vad_segments = vad_model.get_speech_segments(temp_file)
                    
                    for start_time, end_time in vad_segments:
                        start_sample = int(start_time * sample_rate)
                        end_sample = int(end_time * sample_rate)
                        start_sample = max(0, start_sample)
                        end_sample = min(audio.shape[-1], end_sample)
                        if end_sample > start_sample:
                            speech_segments.append((start_sample, end_sample))
                    
                    print(f"✅ VAD: Found {len(speech_segments)} speech segments")
                    
                finally:
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.unlink(temp_file)
                        except Exception:
                            pass
                            
            except Exception as vad_error:
                print(f"⚠️ VAD failed: {vad_error}")
                duration = audio.shape[-1] / sample_rate
                speech_segments = [(0, int(duration * sample_rate))]
            
            # Step 2: Speaker Diarization
            print("👥 Step 2: Applying Speaker Diarization for enrollment...")
            speaker_segments = {}
            try:
                # Try to import and use pyannote diarization
                try:
                    from pyannote.audio import Pipeline
                    
                    # Initialize diarization pipeline
                    local_pyannote_cache = os.path.abspath('./pretrained_models/pyannote')
                    with suppress_stdout_stderr():
                        diarization_pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            cache_dir=local_pyannote_cache,
                            use_auth_token=HF_AUTH_TOKEN
                        )
                    
                    # Create temporary file for diarization
                    temp_file = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                            temp_file = f.name
                        
                        # Save audio (ensure it's on CPU)
                        audio_to_save = audio.cpu() if audio.is_cuda else audio
                        torchaudio.save(temp_file, audio_to_save.unsqueeze(0), sample_rate)
                        
                        # Run diarization
                        diarization_result = diarization_pipeline(temp_file)
                        
                        # Process diarization results
                        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                            if speaker not in speaker_segments:
                                speaker_segments[speaker] = []
                            
                            start_sample = int(turn.start * sample_rate)
                            end_sample = int(turn.end * sample_rate)
                            start_sample = max(0, start_sample)
                            end_sample = min(audio.shape[-1], end_sample)
                            if end_sample > start_sample:
                                speaker_segments[speaker].append((start_sample, end_sample))
                        
                        print(f"✅ Diarization: Found {len(speaker_segments)} speakers")
                        for speaker, segments in speaker_segments.items():
                            total_duration = sum([(end - start) / sample_rate for start, end in segments])
                            print(f"   👤 {speaker}: {len(segments)} segments, {total_duration:.2f}s total")
                            
                    finally:
                        if temp_file and os.path.exists(temp_file):
                            try:
                                os.unlink(temp_file)
                            except Exception:
                                pass
                                
                except ImportError:
                    print("⚠️ pyannote.audio not available for diarization")
                    duration = audio.shape[-1] / sample_rate
                    speaker_segments = {"SPEAKER_00": [(0, int(duration * sample_rate))]}
                    
            except Exception as diar_error:
                print(f"⚠️ Diarization failed: {diar_error}")
                duration = audio.shape[-1] / sample_rate
                speaker_segments = {"SPEAKER_00": [(0, int(duration * sample_rate))]}
            
            # Step 3: Enhanced Quality Assessment for Each Speaker
            print("🏆 Step 3: Evaluating speaker segments for enrollment quality...")
            
            best_speaker = None
            best_quality_score = -1
            best_segment_audio = None
            
            for speaker, segments in speaker_segments.items():
                # Calculate total speaker duration
                total_duration = sum([(end - start) / sample_rate for start, end in segments])
                
                if total_duration < 8.0:  # Enforce minimum 8 seconds for enrollment
                    print(f"   ⏭️ {speaker}: Too short ({total_duration:.2f}s), minimum 8.0s required for enrollment")
                    continue
                
                # Extract audio for this speaker
                speaker_audio_segments = []
                for start_sample, end_sample in segments:
                    if end_sample > start_sample:
                        segment_audio = audio[start_sample:end_sample]
                        speaker_audio_segments.append(segment_audio)
                
                if not speaker_audio_segments:
                    continue
                
                # Concatenate all segments for this speaker
                speaker_full_audio = torch.cat(speaker_audio_segments, dim=-1)
                
                # Quality Metric 1: RMS Energy (Overall Volume)
                rms_energy = torch.sqrt(torch.mean(speaker_full_audio**2)).item()
                
                # Quality Metric 2: Energy Dynamics (Speech Variability)
                frame_size = sample_rate // 10  # 100ms frames
                energy_variance = 0
                if speaker_full_audio.shape[-1] >= frame_size * 2:
                    frames = speaker_full_audio.unfold(-1, frame_size, frame_size // 2)
                    frame_energies = torch.sqrt(torch.mean(frames**2, dim=-1))
                    energy_variance = torch.var(frame_energies).item()
                
                # Quality Metric 3: SNR Estimation (Signal-to-Noise Ratio)
                snr_estimate = 1.0  # Default
                if speaker_full_audio.shape[-1] >= frame_size * 4:
                    frames = speaker_full_audio.unfold(-1, frame_size, frame_size)
                    frame_energies = torch.sqrt(torch.mean(frames**2, dim=-1))
                    sorted_energies = torch.sort(frame_energies)[0]
                    
                    # Bottom 25% as noise estimate, top 25% as signal estimate
                    noise_floor = sorted_energies[:len(sorted_energies)//4].mean().item()
                    signal_peak = sorted_energies[-len(sorted_energies)//4:].mean().item()
                    snr_estimate = signal_peak / (noise_floor + 1e-8)
                
                # Quality Metric 4: Zero-Crossing Rate (Speech characteristics)
                zero_crossings = torch.sum(torch.diff(torch.sign(speaker_full_audio)) != 0).item()
                zcr = zero_crossings / len(speaker_full_audio) if len(speaker_full_audio) > 0 else 0
                
                # Combined Quality Score with weighted components
                quality_score = (
                    rms_energy * 2000 +           # Energy component (scaled)
                    energy_variance * 8000 +      # Dynamics component (scaled)
                    min(snr_estimate, 20) * 0.05 + # SNR component (capped)
                    total_duration * 0.2 +        # Duration bonus
                    min(zcr * 10000, 1.0)        # ZCR component (speech-like)
                )
                
                print(f"   🎯 {speaker}: Quality={quality_score:.3f}")
                print(f"      📊 Metrics: energy={rms_energy:.4f}, dynamics={energy_variance:.4f}")
                print(f"      📊 SNR={snr_estimate:.2f}, duration={total_duration:.1f}s, ZCR={zcr:.4f}")
                
                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                    best_speaker = speaker
                    best_segment_audio = speaker_full_audio
            
            if best_segment_audio is not None:
                print(f"🏆 Selected speaker {best_speaker} for enrollment (quality score: {best_quality_score:.3f})")
                
                # Ensure proper length for further processing
                target_length = audio.shape[0]
                if len(best_segment_audio) > target_length:
                    # Take the middle portion if too long
                    excess = len(best_segment_audio) - target_length
                    start_trim = excess // 2
                    best_segment_audio = best_segment_audio[start_trim:start_trim + target_length]
                elif len(best_segment_audio) < target_length:
                    # Pad with silence if too short
                    padding = target_length - len(best_segment_audio)
                    best_segment_audio = torch.nn.functional.pad(best_segment_audio, (0, padding))
                
                enhanced_batch.append(best_segment_audio.to(signal.device))
            else:
                print("⚠️ No suitable speaker segments found, using original audio")
                enhanced_batch.append(audio)
        
        # Stack results
        result = torch.stack(enhanced_batch)
        
        # Return with original shape
        if len(original_shape) == 1:
            result = result.squeeze(0)
            
        print("🎯 Enrollment speaker selection completed: VAD → Diarization → Quality Assessment → Best Speaker Selection")
        return result
        
    except Exception as e:
        print(f"⚠️ Enrollment speaker selection failed: {e}")
        return signal

def average_embeddings(embeddings: list) -> torch.Tensor:
    """Average multiple embeddings to create a robust voiceprint."""
    if len(embeddings) == 1:
        return embeddings[0]
    
    # Stack all embeddings and compute mean
    stacked = torch.stack([emb.squeeze() for emb in embeddings])
    averaged = torch.mean(stacked, dim=0)
    
    # Normalize the averaged embedding
    averaged = averaged / averaged.norm()
    
    return averaged.unsqueeze(0)

# Global storage for multi-sample enrollment sessions
enrollment_sessions = {}

class EnrollmentSession:
    def __init__(self, user_id: str, target_samples: int = 5):
        self.user_id = user_id
        self.target_samples = target_samples
        self.collected_samples = []
        self.sample_embeddings = []
        self.created_at = os.times().elapsed
    
    def add_sample(self, embedding: torch.Tensor):
        self.sample_embeddings.append(embedding)
        self.collected_samples.append(len(self.sample_embeddings))
    
    def is_complete(self) -> bool:
        return len(self.sample_embeddings) >= self.target_samples
    
    def get_averaged_embedding(self) -> torch.Tensor:
        return average_embeddings(self.sample_embeddings)

@router.post("/enroll-sample", response_model=MultiSampleEnrollmentResponse, tags=["Speaker Management"])
async def enroll_sample(user_id: str = Form(...), file: UploadFile = File(...)):
    """Add a sample to multi-sample enrollment process."""
    from main import speaker_encoder
    
    if not speaker_encoder:
        raise HTTPException(status_code=500, detail="Speaker encoder model not available.")
    
    # Check if user already exists with completed enrollment
    if user_exists(user_id):
        enrollment_info = get_user_enrollment_info(user_id)
        if enrollment_info and enrollment_info.get('enrollment_type') == 'multi_sample':
            raise HTTPException(status_code=409, detail=f"User '{user_id}' already has a complete multi-sample enrollment. Use /enroll-replace to replace it.")
    
    # Get or create enrollment session
    session_key = f"session_{user_id}"
    if session_key not in enrollment_sessions:
        enrollment_sessions[session_key] = EnrollmentSession(user_id)
    
    session = enrollment_sessions[session_key]
    
    temp_file_path = None
    converted_file_path = None
    
    try:
        # Validate file size (2MB limit)
        await validate_file_size(file, max_size_mb=2.0)
        
        # Process audio file
        temp_file_path = await save_upload_to_temp_file(file)
        print(f"📁 Processing sample {len(session.sample_embeddings) + 1} for user '{user_id}'")
        
        # Convert audio to compatible format
        converted_file_path = convert_audio_to_wav(temp_file_path)
        
        # Load and process audio
        signal, fs = torchaudio.load(converted_file_path)
        signal = signal.to(DEVICE)  # Move to GPU
        print(f"🎵 Loaded audio: shape={signal.shape}, sample_rate={fs}")
        
        # 🚦 AUDIO QUALITY GATEKEEPER
        is_valid, error_message = audio_quality_gatekeeper(signal, fs)
        if not is_valid:
            cleanup_temp_file(temp_file_path)
            if converted_file_path != temp_file_path:
                cleanup_temp_file(converted_file_path)
            raise HTTPException(status_code=400, detail=error_message)
        
        # 🎯 OPTIMAL SPEECH SEGMENT SELECTION
        print("🎯 Applying optimal speech segment selection for enrollment...")
        original_signal = signal.clone()
        
        # Apply optimal segment selection for audio longer than 12 seconds
        if signal.shape[-1] / fs > 12.0:
            signal = find_optimal_speech_segment(signal, fs, target_duration=12.0, min_duration=8.0)
            print(f"✅ Selected optimal {signal.shape[-1]/fs:.1f}s segment from original {original_signal.shape[-1]/fs:.1f}s audio")
        else:
            print(f"📋 Audio duration ({signal.shape[-1]/fs:.1f}s) within optimal range, proceeding without segment selection")
        
        # 🔇 ENHANCED PAUSE-ROBUST PROCESSING
        print("🔇 Applying enhanced pause-robust processing for enrollment...")
        signal = enhanced_pause_robust_processing(signal, fs)
        print(f"✅ Pause processing complete: {signal.shape[-1]/fs:.1f}s continuous speech")
        
        # Apply enhanced enrollment speaker selection (optimized for enrollment quality)
        signal = apply_enrollment_speaker_selection(signal, sample_rate=fs)
        signal = signal.to(DEVICE)  # Ensure it's on GPU after processing
        print(f"🎯 Applied enhanced enrollment speaker selection, processed shape={signal.shape}")
        
        # Extract embedding
        embedding = speaker_encoder.encode_batch(signal)
        embedding = embedding.to(DEVICE)  # Ensure embedding is on GPU
        print(f"🔢 Generated embedding shape: {embedding.shape}")
        
        # Add sample to session
        session.add_sample(embedding)
        
        # Check if enrollment is complete
        if session.is_complete():
            # Calculate quality metrics
            quality, avg_similarity = calculate_embedding_quality(session.sample_embeddings)
            
            # Create averaged embedding
            averaged_embedding = session.get_averaged_embedding()
            
            # Save to database
            success = save_user_samples(user_id, session.sample_embeddings, averaged_embedding)
            
            if success:
                # Clean up session
                del enrollment_sessions[session_key]
                
                return MultiSampleEnrollmentResponse(
                    status="complete",
                    user_id=user_id,
                    message=f"Multi-sample enrollment complete! Quality: {quality} (avg similarity: {avg_similarity:.3f})",
                    samples_collected=len(session.sample_embeddings),
                    total_samples_needed=session.target_samples,
                    enrollment_complete=True
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to save enrollment data.")
        else:
            return MultiSampleEnrollmentResponse(
                status="in_progress",
                user_id=user_id,
                message=f"Sample {len(session.sample_embeddings)} collected. Please provide {session.target_samples - len(session.sample_embeddings)} more samples.",
                samples_collected=len(session.sample_embeddings),
                total_samples_needed=session.target_samples,
                enrollment_complete=False
            )
    
    except Exception as e:
        print(f"❌ Sample enrollment error: {e}")
        raise HTTPException(status_code=500, detail=f"Sample enrollment failed: {str(e)}")
    
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
        if converted_file_path and converted_file_path != temp_file_path:
            cleanup_temp_file(converted_file_path)

@router.get("/enrollment-status/{user_id}", response_model=EnrollmentStatusResponse, tags=["Speaker Management"])
async def get_enrollment_status(user_id: str):
    """Get enrollment status and quality information for a user."""
    
    # Check session first
    session_key = f"session_{user_id}"
    if session_key in enrollment_sessions:
        session = enrollment_sessions[session_key]
        if len(session.sample_embeddings) > 1:
            quality, avg_similarity = calculate_embedding_quality(session.sample_embeddings)
        else:
            quality, avg_similarity = "incomplete", 0.0
            
        return EnrollmentStatusResponse(
            user_id=user_id,
            enrollment_type="multi_sample_in_progress",
            num_samples=len(session.sample_embeddings),
            created_at="Session started",
            embedding_quality=f"{quality} (similarity: {avg_similarity:.3f})"
        )
    
    # Check database
    enrollment_info = get_user_enrollment_info(user_id)
    if not enrollment_info:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    
    enrollment_type = enrollment_info.get('enrollment_type', 'single_sample')
    num_samples = enrollment_info.get('num_samples', 1)
    created_at = enrollment_info.get('created_at', 'Unknown')
    
    # Calculate quality if we have sample embeddings
    if 'sample_embeddings' in enrollment_info and len(enrollment_info['sample_embeddings']) > 1:
        # Reconstruct embeddings for quality calculation
        sample_embeddings = [torch.tensor(emb).unsqueeze(0) for emb in enrollment_info['sample_embeddings']]
        quality, avg_similarity = calculate_embedding_quality(sample_embeddings)
        quality_str = f"{quality} (avg similarity: {avg_similarity:.3f})"
    else:
        quality_str = "single_sample"
    
    return EnrollmentStatusResponse(
        user_id=user_id,
        enrollment_type=enrollment_type,
        num_samples=num_samples,
        created_at=created_at,
        embedding_quality=quality_str
    )

@router.delete("/enrollment/{user_id}", tags=["Speaker Management"])
async def clear_enrollment(user_id: str):
    """Clear enrollment session or data for a user."""
    # Clear session if exists
    session_key = f"session_{user_id}"
    if session_key in enrollment_sessions:
        del enrollment_sessions[session_key]
        return {"message": f"Enrollment session cleared for user '{user_id}'"}
    
    # For database clearing, you might want to implement this based on your needs
    return {"message": f"No active enrollment session found for user '{user_id}'"}

@router.post("/enroll", response_model=EnrollmentResponse, tags=["Speaker Management"])
async def enroll_speaker(user_id: str = Form(...), file: UploadFile = File(...)):
    """Enroll a new speaker using SpeechBrain's EncoderClassifier."""
    from main import speaker_encoder
    
    if not speaker_encoder:
        raise HTTPException(status_code=500, detail="Speaker encoder model not available.")
    
    if user_exists(user_id):
        raise HTTPException(status_code=409, detail=f"User '{user_id}' is already enrolled.")
    
    temp_file_path = None
    converted_file_path = None
    try:
        # Validate file size (2MB limit)
        await validate_file_size(file, max_size_mb=2.0)
        
        # Save uploaded file to temporary location
        temp_file_path = await save_upload_to_temp_file(file)
        print(f"📁 Uploaded file type: {file.content_type}, filename: {file.filename}")
        
        # Convert audio to compatible format
        converted_file_path = convert_audio_to_wav(temp_file_path)
        
        # Load audio using torchaudio (as per SpeechBrain documentation)
        signal, fs = torchaudio.load(converted_file_path)
        signal = signal.to(DEVICE)  # Move to GPU
        print(f"🎵 Loaded audio: shape={signal.shape}, sample_rate={fs}")
        
        # 🚦 AUDIO QUALITY GATEKEEPER
        is_valid, error_message = audio_quality_gatekeeper(signal, fs)
        if not is_valid:
            cleanup_temp_file(temp_file_path)
            if converted_file_path != temp_file_path:
                cleanup_temp_file(converted_file_path)
            raise HTTPException(status_code=400, detail=error_message)
        
        # 🎯 OPTIMAL SPEECH SEGMENT SELECTION
        print("🎯 Applying optimal speech segment selection for single enrollment...")
        original_signal = signal.clone()
        
        # Apply optimal segment selection for audio longer than 12 seconds
        if signal.shape[-1] / fs > 12.0:
            signal = find_optimal_speech_segment(signal, fs, target_duration=12.0, min_duration=8.0)
            print(f"✅ Selected optimal {signal.shape[-1]/fs:.1f}s segment from original {original_signal.shape[-1]/fs:.1f}s audio")
        else:
            print(f"📋 Audio duration ({signal.shape[-1]/fs:.1f}s) within optimal range, proceeding without segment selection")
        
        # 🔇 ENHANCED PAUSE-ROBUST PROCESSING
        print("🔇 Applying enhanced pause-robust processing for single enrollment...")
        signal = enhanced_pause_robust_processing(signal, fs)
        print(f"✅ Pause processing complete: {signal.shape[-1]/fs:.1f}s continuous speech")
        
        # Apply enhanced enrollment speaker selection (optimized for enrollment quality)
        signal = apply_enrollment_speaker_selection(signal, sample_rate=fs)
        signal = signal.to(DEVICE)  # Ensure it's on GPU after processing
        print(f"🎯 Applied enhanced enrollment speaker selection, processed shape={signal.shape}")
        
        # Extract embedding using EncoderClassifier
        embedding = speaker_encoder.encode_batch(signal)
        embedding = embedding.to(DEVICE)  # Ensure embedding is on GPU
        print(f"🔢 Generated embedding shape: {embedding.shape}")
        
        # Save to database
        if save_user_embedding(user_id, embedding):
            return EnrollmentResponse(
                status="success", 
                user_id=user_id, 
                message=f"User '{user_id}' enrolled successfully."
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save user enrollment data.")
    
    except Exception as e:
        print(f"❌ Enrollment error: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
        if converted_file_path and converted_file_path != temp_file_path:
            cleanup_temp_file(converted_file_path)

@router.post("/smart-auth", response_model=SmartAuthResponse, tags=["Voice Authentication"])
async def smart_authenticate(background_tasks: BackgroundTasks, user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Smart authentication endpoint that automatically enrolls or verifies based on user existence.
    
    - If user_id exists: Performs verification and returns similarity score
    - If user_id doesn't exist: Enrolls the user and returns enrollment status
    
    This endpoint is perfect for seamless user onboarding and authentication.
    """
    from main import speaker_encoder, speaker_verifier, CONFIG
    
    if not speaker_encoder or not speaker_verifier:
        raise HTTPException(status_code=500, detail="Speaker models not available.")
    
    # Check if user already exists
    user_already_exists = user_exists(user_id)
    
    temp_file_path = None
    converted_file_path = None
    
    try:
        # Validate file size (2MB limit)
        await validate_file_size(file, max_size_mb=2.0)
        
        # Save uploaded file to temporary location
        temp_file_path = await save_upload_to_temp_file(file)
        print(f"📁 Processing smart auth for user '{user_id}' (exists: {user_already_exists})")
        
        # Convert audio to compatible format
        converted_file_path = convert_audio_to_wav(temp_file_path)
        
        # Load and process audio
        signal, fs = torchaudio.load(converted_file_path)
        signal = signal.to(DEVICE)  # Move to GPU
        print(f"🎵 Loaded audio: shape={signal.shape}, sample_rate={fs}")
        
        # 🚦 AUDIO QUALITY GATEKEEPER
        is_valid, error_message = audio_quality_gatekeeper(signal, fs)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Audio quality check failed: {error_message}")
        
        # 🎯 OPTIMAL SPEECH SEGMENT SELECTION
        print("🎯 Applying optimal speech segment selection...")
        
        # Apply optimal segment selection for audio longer than 12 seconds
        if signal.shape[-1] / fs > 12.0:
            signal = find_optimal_speech_segment(signal, fs, target_duration=12.0, min_duration=4.0)
            print(f"✅ Selected optimal segment: {signal.shape[-1]/fs:.1f}s")
        else:
            print(f"✅ Audio duration acceptable: {signal.shape[-1]/fs:.1f}s")
        
    # Continue to verification/enrollment

        if user_already_exists:
            # USER EXISTS: PERFORM VERIFICATION
            print(f"👤 User '{user_id}' exists - performing verification...")
            
            # Apply advanced VAD+Diarization pipeline for multi-speaker environments
            enrolled_embeddings = get_all_user_embeddings()
            enrolled_embeddings_for_pipeline = {user_id: enrolled_embeddings[user_id]}
            signal = apply_advanced_vad_diarization_pipeline(signal, enrolled_embeddings_for_pipeline, sample_rate=fs)
            signal = signal.to(DEVICE)  # Ensure it's on GPU after processing
            print(f"🎯 Applied advanced VAD+Diarization pipeline, processed shape={signal.shape}")
            
            # Extract embedding from verification audio
            verification_embedding = speaker_verifier.encode_batch(signal)
            verification_embedding = verification_embedding.to(DEVICE)  # Ensure embedding is on GPU
            print(f"🔢 Generated verification embedding shape: {verification_embedding.shape}")
            
            # Get enrolled embedding for the specific user
            enrolled_embedding = torch.tensor(enrolled_embeddings[user_id]).to(DEVICE)
            
            # Compute similarity score using cosine similarity
            print(f"🔍 Computing similarity score for user '{user_id}'...")
            
            # Normalize embeddings for cosine similarity
            enrolled_norm = enrolled_embedding / enrolled_embedding.norm(dim=-1, keepdim=True)
            verification_norm = verification_embedding / verification_embedding.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity_score = torch.cosine_similarity(enrolled_norm.squeeze(), verification_norm.squeeze(), dim=0)
            score = similarity_score.item()
            
            print(f"📊 User '{user_id}' similarity score: {score:.4f}")
            
            # Determine if verification passed
            verification_threshold = CONFIG.VERIFICATION_THRESHOLD
            verified = score >= verification_threshold
            
            # Schedule background task for EWMA adaptation and storage
            background_tasks.add_task(
                background_retrain_and_store,
                user_id,
                verification_embedding,
                score,
                signal,
                fs
            )
            
            # Prepare verification response
            if verified:
                status = "success"
                message = f"User '{user_id}' successfully verified with score {score:.4f} (threshold: {verification_threshold:.3f})"
                print(f"✅ {message}")
            else:
                status = "failed"
                message = f"User '{user_id}' verification failed with score {score:.4f} (threshold: {verification_threshold:.3f})"
                print(f"❌ {message}")
            
            return SmartAuthResponse(
                action="verified",
                status=status,
                user_id=user_id,
                message=message,
                similarity_score=score,
                verified=verified
            )
            
        else:
            # USER DOESN'T EXIST: PERFORM ENROLLMENT
            print(f"🆕 User '{user_id}' doesn't exist - performing enrollment...")
            
            # Apply enhanced enrollment speaker selection (optimized for enrollment quality)
            signal = apply_enrollment_speaker_selection(signal, sample_rate=fs)
            signal = signal.to(DEVICE)  # Ensure it's on GPU after processing
            print(f"🎯 Applied enhanced enrollment speaker selection, processed shape={signal.shape}")
            
            # Extract embedding using EncoderClassifier
            embedding = speaker_encoder.encode_batch(signal)
            embedding = embedding.to(DEVICE)  # Ensure embedding is on GPU
            print(f"🔢 Generated enrollment embedding shape: {embedding.shape}")
            
            # Save to database
            if save_user_embedding(user_id, embedding):
                status = "success"
                message = f"User '{user_id}' successfully enrolled with voiceprint from {signal.shape[-1]/fs:.1f}s of audio"
                print(f"✅ {message}")
            else:
                status = "failed"
                message = f"Failed to save enrollment data for user '{user_id}'"
                print(f"❌ {message}")
                raise HTTPException(status_code=500, detail=message)
            
            return SmartAuthResponse(
                action="enrolled",
                status=status,
                user_id=user_id,
                message=message,
                similarity_score=None,
                verified=None
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Smart authentication error: {e}")
        raise HTTPException(status_code=500, detail=f"Smart authentication failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
        if converted_file_path and converted_file_path != temp_file_path:
            cleanup_temp_file(converted_file_path)

@router.post("/smart-auth-labeled", response_model=LabeledSmartAuthResponse, tags=["Voice Authentication"])
async def smart_authenticate_labeled(
    background_tasks: BackgroundTasks, 
    user_id: str = Form(...), 
    is_actual_speaker: bool = Form(..., description="Ground truth label: True if this is the actual speaker, False if imposter"),
    file: UploadFile = File(...)
):
    """
    Labeled smart authentication endpoint for training data collection.
    
    This endpoint works like smart-auth but includes ground truth labels to indicate whether
    the voice sample is from the actual speaker or an imposter. All data is stored in the
    sample_voice_data collection with labels for machine learning model training and evaluation.
    
    - If user_id exists: Performs verification and stores labeled verification data
    - If user_id doesn't exist: Enrolls the user and stores labeled enrollment data
    
    Args:
        user_id: The claimed user identity
        is_actual_speaker: Ground truth label (True = actual speaker, False = imposter)
        file: Audio file for authentication
    
    Returns:
        Authentication result with ground truth labels and storage confirmation
    """
    from main import speaker_encoder, speaker_verifier, CONFIG
    
    if not speaker_encoder or not speaker_verifier:
        raise HTTPException(status_code=500, detail="Speaker models not available.")
    
    # Check if user already exists
    user_already_exists = user_exists(user_id)
    
    temp_file_path = None
    converted_file_path = None
    data_stored = False
    
    try:
        # Validate file size (2MB limit)
        await validate_file_size(file, max_size_mb=2.0)
        
        # Save uploaded file to temporary location
        temp_file_path = await save_upload_to_temp_file(file)
        label_str = "ACTUAL" if is_actual_speaker else "IMPOSTER"
        print(f"🏷️ Processing labeled smart auth for user '{user_id}' [{label_str}] (exists: {user_already_exists})")
        
        # Convert audio to compatible format
        converted_file_path = convert_audio_to_wav(temp_file_path)
        
        # Load and process audio
        signal, fs = torchaudio.load(converted_file_path)
        signal = signal.to(DEVICE)  # Move to GPU
        print(f"🎵 Loaded audio: shape={signal.shape}, sample_rate={fs}")
        
        # 🚦 AUDIO QUALITY GATEKEEPER
        is_valid, error_message = audio_quality_gatekeeper(signal, fs)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Audio quality check failed: {error_message}")
        
        # 🎯 OPTIMAL SPEECH SEGMENT SELECTION
        print("🎯 Applying optimal speech segment selection...")
        
        # Apply optimal segment selection for audio longer than 12 seconds
        if signal.shape[-1] / fs > 12.0:
            signal = find_optimal_speech_segment(signal, fs, target_duration=12.0, min_duration=4.0)
            print(f"✅ Selected optimal segment: {signal.shape[-1]/fs:.1f}s")
        else:
            print(f"✅ Audio duration acceptable: {signal.shape[-1]/fs:.1f}s")
        
        # 🔇 ENHANCED PAUSE-ROBUST PROCESSING
        print("🔇 Applying enhanced pause-robust processing...")
        original_duration_sec = signal.shape[-1] / fs
        signal = enhanced_pause_robust_processing(signal, fs)
        processed_duration_sec = signal.shape[-1] / fs
        print(f"✅ Pause processing complete: {processed_duration_sec:.1f}s continuous speech")

        # 📈 Compute audio metrics
        energy_level = compute_rms_energy(signal)
        snr_db = estimate_snr_db(signal, fs)
        # If you computed diarization earlier, pass diarization_count here to avoid any heavy work.
        # For now, we pass None so the function uses a lightweight fallback (returns 1).
        speakers_detected = detect_number_of_speakers(signal, fs, diarization_count=None)
        
        if user_already_exists:
            # USER EXISTS: PERFORM LABELED VERIFICATION
            print(f"👤 User '{user_id}' exists - performing labeled verification...")
            
            # Apply advanced VAD+Diarization pipeline for multi-speaker environments
            enrolled_embeddings = get_all_user_embeddings()
            enrolled_embeddings_for_pipeline = {user_id: enrolled_embeddings[user_id]}
            signal = apply_advanced_vad_diarization_pipeline(signal, enrolled_embeddings_for_pipeline, sample_rate=fs)
            signal = signal.to(DEVICE)  # Ensure it's on GPU after processing
            print(f"🎯 Applied advanced VAD+Diarization pipeline, processed shape={signal.shape}")
            
            # Extract embedding from verification audio
            verification_embedding = speaker_verifier.encode_batch(signal)
            verification_embedding = verification_embedding.to(DEVICE)  # Ensure embedding is on GPU
            print(f"🔢 Generated verification embedding shape: {verification_embedding.shape}")
            
            # Get enrolled embedding for the specific user
            enrolled_embedding = torch.tensor(enrolled_embeddings[user_id]).to(DEVICE)
            
            # Compute similarity score using cosine similarity
            print(f"🔍 Computing similarity score for user '{user_id}'...")
            
            # Normalize embeddings for cosine similarity
            enrolled_norm = enrolled_embedding / enrolled_embedding.norm(dim=-1, keepdim=True)
            verification_norm = verification_embedding / verification_embedding.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity_score = torch.cosine_similarity(enrolled_norm.squeeze(), verification_norm.squeeze(), dim=0)
            score = similarity_score.item()
            
            print(f"📊 User '{user_id}' similarity score: {score:.4f}")
            
            # Determine if verification passed
            verification_threshold = CONFIG.VERIFICATION_THRESHOLD
            verified = score >= verification_threshold
            
            # Store labeled verification data in database
            audio_info = {
                "sample_rate": fs,
                "action_type": "verification",
                "audio_shape": list(signal.shape),
                "energy_level": energy_level,
                "snr_db": snr_db,
                "original_audio_length_sec": original_duration_sec,
                "speech_detected_audio_length_sec": processed_duration_sec,
                "speakers_detected": speakers_detected
            }
            
            data_stored = store_labeled_voice_sample(
                user_id=user_id,
                embedding=verification_embedding,
                similarity_score=score,
                is_actual_speaker=is_actual_speaker,
                audio_duration=signal.shape[-1] / fs,
                audio_info=audio_info
            )
            
            # Schedule background task for EWMA adaptation (but not regular storage as we already stored labeled data)
            if CONFIG.EWMA_ENABLED and score >= CONFIG.EWMA_ADAPTATION_THRESHOLD:
                background_tasks.add_task(
                    background_retrain_and_store,
                    user_id,
                    verification_embedding,
                    score,
                    signal,
                    fs
                )
            
            # Prepare verification response
            if verified:
                status = "success"
                message = f"User '{user_id}' [{label_str}] successfully verified with score {score:.4f} (threshold: {verification_threshold:.3f})"
                print(f"✅ {message}")
            else:
                status = "failed"
                message = f"User '{user_id}' [{label_str}] verification failed with score {score:.4f} (threshold: {verification_threshold:.3f})"
                print(f"❌ {message}")
            
            return LabeledSmartAuthResponse(
                action="verified",
                status=status,
                user_id=user_id,
                is_actual_speaker=is_actual_speaker,
                message=message,
                similarity_score=score,
                verified=verified,
                data_stored=data_stored
            )
            
        else:
            # USER DOESN'T EXIST: PERFORM LABELED ENROLLMENT
            print(f"🆕 User '{user_id}' doesn't exist - performing labeled enrollment...")
            
            # Apply enhanced enrollment speaker selection (optimized for enrollment quality)
            signal = apply_enrollment_speaker_selection(signal, sample_rate=fs)
            signal = signal.to(DEVICE)  # Ensure it's on GPU after processing
            print(f"🎯 Applied enhanced enrollment speaker selection, processed shape={signal.shape}")
            
            # Extract embedding using EncoderClassifier
            embedding = speaker_encoder.encode_batch(signal)
            embedding = embedding.to(DEVICE)  # Ensure embedding is on GPU
            print(f"🔢 Generated enrollment embedding shape: {embedding.shape}")
            
            # Save enrollment to database
            enrollment_success = save_user_embedding(user_id, embedding)
            
            # Store labeled enrollment data in sample_voice_data collection
            audio_info = {
                "sample_rate": fs,
                "action_type": "enrollment",
                "audio_shape": list(signal.shape),
                "energy_level": energy_level,
                "snr_db": snr_db,
                "original_audio_length_sec": original_duration_sec,
                "speech_detected_audio_length_sec": processed_duration_sec,
                "speakers_detected": speakers_detected
            }
            
            data_stored = store_labeled_voice_sample(
                user_id=user_id,
                embedding=embedding,
                similarity_score=1.0,  # Perfect score for enrollment data
                is_actual_speaker=is_actual_speaker,
                audio_duration=signal.shape[-1] / fs,
                audio_info=audio_info
            )
            
            if enrollment_success:
                status = "success"
                message = f"User '{user_id}' [{label_str}] successfully enrolled with voiceprint from {signal.shape[-1]/fs:.1f}s of audio"
                print(f"✅ {message}")
            else:
                status = "failed"
                message = f"Failed to save enrollment data for user '{user_id}' [{label_str}]"
                print(f"❌ {message}")
                raise HTTPException(status_code=500, detail=message)
            
            return LabeledSmartAuthResponse(
                action="enrolled",
                status=status,
                user_id=user_id,
                is_actual_speaker=is_actual_speaker,
                message=message,
                similarity_score=None,
                verified=None,
                data_stored=data_stored
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Labeled smart authentication error: {e}")
        raise HTTPException(status_code=500, detail=f"Labeled smart authentication failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
        if converted_file_path and converted_file_path != temp_file_path:
            cleanup_temp_file(converted_file_path)

@router.post("/identify", response_model=IdentificationResponse, tags=["Voice Authentication"])
async def identify_speaker(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Identify speaker using SpeechBrain's built-in verification methods."""
    from main import speaker_verifier, CONFIG
    
    if not speaker_verifier:
        raise HTTPException(status_code=500, detail="Speaker verifier model not available.")
    
    enrolled_embeddings = get_all_user_embeddings()
    if not enrolled_embeddings:
        raise HTTPException(status_code=404, detail="No enrolled users found.")
    
    verification_temp_file = None
    converted_file_path = None
    
    try:
        # Validate file size (2MB limit)
        await validate_file_size(file, max_size_mb=2.0)
        
        # Save verification audio to temporary file
        verification_temp_file = await save_upload_to_temp_file(file)
        print(f"📁 Uploaded file type: {file.content_type}, filename: {file.filename}")
        
        # Convert audio to compatible format
        converted_file_path = convert_audio_to_wav(verification_temp_file)
        
        best_score = -1.0
        identified_user = None
        
        print(f"\n🔍 Starting speaker verification for {len(enrolled_embeddings)} enrolled users...")
        
        # Load verification audio
        verification_signal, fs = torchaudio.load(converted_file_path)
        verification_signal = verification_signal.to(DEVICE)  # Move to GPU
        print(f"🎵 Loaded verification audio: shape={verification_signal.shape}, sample_rate={fs}")
        
        # 🚦 AUDIO QUALITY GATEKEEPER
        is_valid, error_message = audio_quality_gatekeeper(verification_signal, fs)
        if not is_valid:
            cleanup_temp_file(verification_temp_file)
            if converted_file_path != verification_temp_file:
                cleanup_temp_file(converted_file_path)
            raise HTTPException(status_code=400, detail=error_message)
        
        # 🎯 OPTIMAL SPEECH SEGMENT SELECTION
        print("🎯 Applying optimal speech segment selection...")
        original_verification_signal = verification_signal.clone()
        
        # Apply optimal segment selection for audio longer than 12 seconds
        if verification_signal.shape[-1] / fs > 12.0:
            verification_signal = find_optimal_speech_segment(verification_signal, fs, target_duration=12.0, min_duration=4.0)
            print(f"✅ Selected optimal {verification_signal.shape[-1]/fs:.1f}s segment from original {original_verification_signal.shape[-1]/fs:.1f}s audio")
        else:
            print(f"📋 Audio duration ({verification_signal.shape[-1]/fs:.1f}s) within optimal range, proceeding without segment selection")
        
        # 🔇 ENHANCED PAUSE-ROBUST PROCESSING
        print("🔇 Applying enhanced pause-robust processing...")
        verification_signal = enhanced_pause_robust_processing(verification_signal, fs)
        print(f"✅ Pause processing complete: {verification_signal.shape[-1]/fs:.1f}s continuous speech")
        
        # Apply advanced VAD+Diarization+Separation pipeline with enrolled embeddings
        verification_signal = apply_advanced_vad_diarization_pipeline(
            verification_signal, 
            enrolled_embeddings=enrolled_embeddings, 
            sample_rate=fs
        )
        verification_signal = verification_signal.to(DEVICE)  # Ensure it's on GPU after processing
        print(f"🎯 Applied advanced VAD+Diarization pipeline to verification audio, processed shape={verification_signal.shape}")
        
        # Import speaker_encoder from main
        from main import speaker_encoder
        verification_embedding = speaker_encoder.encode_batch(verification_signal)
        verification_embedding = verification_embedding.to(DEVICE)  # Ensure GPU
        
        print(f"🎯 Verification embedding shape: {verification_embedding.shape}")
        
        # Compare with all enrolled users using cosine similarity
        for user_id, enrolled_embedding in enrolled_embeddings.items():
            # Ensure enrolled embedding is on the same device
            enrolled_embedding = enrolled_embedding.to(DEVICE)
            
            # Compute cosine similarity between embeddings
            verification_norm = verification_embedding / verification_embedding.norm(dim=-1, keepdim=True)
            enrolled_norm = enrolled_embedding / enrolled_embedding.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = torch.cosine_similarity(verification_norm.squeeze(), enrolled_norm.squeeze(), dim=0)
            current_score = similarity.item()
            
            print(f"👤 User '{user_id}': similarity score = {current_score:.4f}")
            
            if current_score > best_score:
                best_score = current_score
                identified_user = user_id
        
        print(f"\n🎯 Best match: User '{identified_user}' with score {best_score:.4f}")
        print(f"🎚️ Threshold: {CONFIG.VERIFICATION_THRESHOLD}")
        
        if best_score >= CONFIG.VERIFICATION_THRESHOLD:
            # 🚀 OPTIMIZATION: Send response immediately, then do background processing
            print(f"🎯 High-confidence verification (score: {best_score:.4f} ≥ {CONFIG.EWMA_ADAPTATION_THRESHOLD})")
            
            # Schedule background task for EWMA adaptation and voice sample storage
            if identified_user:
                
                background_tasks.add_task(
                    background_retrain_and_store,
                    user_id=identified_user,
                    verification_embedding=verification_embedding,
                    best_score=best_score,
                    verification_signal=verification_signal,
                    fs=fs
                )
                print(f"📋 Background retraining and storage scheduled for {identified_user}")
            
            return IdentificationResponse(
                status="success", 
                identified_user_id=identified_user, 
                message=f"Speaker identified as {identified_user}.", 
                highest_score=best_score
            )
        else:
            return IdentificationResponse(
                status="failure", 
                identified_user_id=None, 
                message="Speaker not recognized or score below threshold.", 
                highest_score=best_score
            )
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions (like quality check failures)
    except Exception as e:
        print(f"❌ Identification error: {e}")
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        if verification_temp_file:
            cleanup_temp_file(verification_temp_file)
        if converted_file_path and converted_file_path != verification_temp_file:
            cleanup_temp_file(converted_file_path)

@router.get("/voice-data-stats")
async def get_voice_data_stats():
    """Get statistics about stored voice samples in the sample_voice_data collection."""
    try:
        from database import get_voice_data_statistics
        stats = get_voice_data_statistics()
        
        return {
            "status": "success",
            "message": "Voice data statistics retrieved successfully",
            "data": stats
        }
    except Exception as e:
        print(f"❌ Error retrieving voice data statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

@router.get("/voice-samples/{user_id}")
async def get_user_voice_samples_endpoint(user_id: str, limit: int = 10, min_score: float = 0.5):
    """Get voice samples for a specific user."""
    try:
        from database import get_user_voice_samples
        samples = get_user_voice_samples(user_id, limit=limit, min_score=min_score)
        
        # Remove the actual embedding data for API response (too large)
        sanitized_samples = []
        for sample in samples:
            sample_copy = sample.copy()
            if 'embedding' in sample_copy:
                sample_copy['embedding_size'] = len(sample_copy['embedding'])
                del sample_copy['embedding']  # Remove large embedding data
            sanitized_samples.append(sample_copy)
        
        return {
            "status": "success",
            "message": f"Retrieved {len(sanitized_samples)} voice samples for {user_id}",
            "user_id": user_id,
            "samples": sanitized_samples,
            "filter": {
                "limit": limit,
                "min_score": min_score
            }
        }
    except Exception as e:
        print(f"❌ Error retrieving voice samples for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve voice samples: {str(e)}")