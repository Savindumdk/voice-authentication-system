# 🎙️ Voice Authentication System

A production-grade, AI-powered voice authentication system built with FastAPI and SpeechBrain. Provides secure speaker enrollment and verification with advanced audio processing pipelines designed for real-world deployment in noisy, multi-speaker environments.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

### Core Capabilities
- ✅ **Multi-Sample Enrollment** - Collects 3-5 voice samples for robust voiceprints
- 🔐 **Smart Authentication** - Auto-enrollment for new users, verification for existing
- 🧠 **Adaptive Learning (EWMA)** - Continuously improves accuracy from high-confidence verifications
- 🚦 **Audio Quality Gatekeeper** - Validates audio before processing (volume, speech presence)
- 👥 **Multi-Speaker Handling** - Detects and separates overlapping speakers
- 🎯 **Optimal Speech Selection** - Extracts best segments from long recordings
- 🔇 **Pause-Robust Processing** - Removes silence for better embeddings
- 📊 **Labeled Data Collection** - Stores ground-truth for continuous improvement

### Advanced Audio Processing Pipeline

```
Raw Audio Input 
    ↓
🚦 Audio Quality Gatekeeper (validation)
    ↓
🎯 Optimal Speech Segment Selection
    ↓
🔇 Pause-Robust Processing (silence removal)
    ↓
📡 Voice Activity Detection (VAD)
    ↓
👥 Speaker Diarization (multi-speaker detection)
    ↓
🔄 Speech Separation (Sepformer)
    ↓
🎯 Source Selection (correct speaker)
    ↓
🔧 Enhancement (noise reduction)
    ↓
🧬 Embedding Extraction (ECAPA-TDNN)
    ↓
✅ Authentication Decision
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- MongoDB Atlas account
- HuggingFace account (for gated models)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Savindumdk/voice-authentication-system.git
cd voice-authentication-system
```

2. **Create virtual environment**
```bash
python -m venv new_venv
# Windows
.\new_venv\Scripts\activate
# Linux/Mac
source new_venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:

```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
DATABASE_NAME=voice_auth
COLLECTION_NAME=user_data

# HuggingFace Token (for pyannote models)
HF_AUTH_TOKEN=your_huggingface_token_here

# Authentication Configuration
VERIFICATION_THRESHOLD=0.50

# EWMA Adaptive Learning
EWMA_ENABLED=true
EWMA_ADAPTATION_THRESHOLD=0.70
EWMA_LEARNING_RATE=0.1
```

5. **Download pretrained models**

Models will be automatically downloaded on first run to the `pretrained_models/` directory.

### Running the Application

```bash
python main.py
```

The server will start at `http://localhost:8000`

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Documentation

### Enrollment Endpoints

#### Single-Sample Enrollment
```http
POST /enroll
Content-Type: multipart/form-data

user_id: string (required)
file: audio file (required, max 2MB)
```

#### Multi-Sample Enrollment
```http
POST /enroll-sample
Content-Type: multipart/form-data

user_id: string (required)
file: audio file (required, max 2MB)
```
Collects 3-5 samples progressively. Returns enrollment status and quality metrics.

#### Check Enrollment Status
```http
GET /enrollment-status/{user_id}
```

#### Clear Enrollment
```http
DELETE /enrollment/{user_id}
```

### Authentication Endpoints

#### Smart Authentication
```http
POST /smart-auth
Content-Type: multipart/form-data

user_id: string (required)
file: audio file (required, max 2MB)
```
Automatically enrolls new users or verifies existing ones.

#### Labeled Smart Authentication
```http
POST /smart-auth-labeled
Content-Type: multipart/form-data

user_id: string (required)
file: audio file (required, max 2MB)
is_actual_speaker: boolean (required)
```
Stores ground-truth labels for model improvement.

#### Verify Against Specific User
```http
POST /verify/{user_id}
Content-Type: multipart/form-data

file: audio file (required, max 2MB)
```

#### Identify Speaker
```http
POST /identify
Content-Type: multipart/form-data

file: audio file (required, max 2MB)
```
Identifies the speaker from all enrolled users.

## 🔬 Technology Stack

### Deep Learning Models

| Model | Purpose | Source |
|-------|---------|--------|
| **ECAPA-TDNN** | Speaker embeddings (192-D) | `speechbrain/spkrec-ecapa-voxceleb` |
| **VAD-CRDNN** | Voice activity detection | `speechbrain/vad-crdnn-libriparty` |
| **PyAnnote VAD** | Advanced voice detection | `pyannote/voice-activity-detection` |
| **Speaker Diarization** | Multi-speaker detection | `pyannote/speaker-diarization-3.1` |
| **Sepformer 2-mix** | 2-speaker separation | `speechbrain/sepformer-wsj02mix` |
| **Sepformer 3-mix** | 3-speaker separation | `speechbrain/sepformer-wsj03mix` |
| **MetricGAN+** | Audio enhancement | `speechbrain/metricgan-plus-voicebank` |

### Core Technologies

- **PyTorch 2.5.1** - Deep learning framework with CUDA support
- **SpeechBrain 1.0.3** - Speech processing toolkit
- **FastAPI 0.116.1** - Modern web framework
- **MongoDB** - NoSQL database for embeddings
- **PyAnnote Audio 3.3.2** - Speaker diarization
- **Torchaudio 2.5.1** - Audio processing

## ⚙️ Configuration

### Verification Threshold
Controls false acceptance rate. Default: `0.50`
- Higher values: More secure, may reject genuine users
- Lower values: More convenient, higher security risk

### EWMA Adaptive Learning
Continuously updates voiceprints from successful authentications.

- **Enabled**: `EWMA_ENABLED=true`
- **Adaptation Threshold**: `0.70` (only adapt from high-confidence verifications)
- **Learning Rate**: `0.1` (10% weight to new samples)

Formula: `E_new = α * E_incoming + (1-α) * E_current`

### Audio Quality Requirements
- **Minimum Speech Duration**: 4 seconds
- **VAD Ratio Threshold**: 15% speech content
- **Energy Threshold**: 0.0005 RMS
- **Maximum File Size**: 2MB

## 🗄️ Database Schema

### User Data Collection

```javascript
{
  "user_id": "string",
  "embedding": [float], // 192-dimensional vector
  "sample_embeddings": [[float]], // For multi-sample enrollment
  "enrollment_type": "single_sample | multi_sample",
  "num_samples": int,
  "created_at": "ISO timestamp",
  "updated_at": "ISO timestamp",
  "last_ewma_update": "ISO timestamp",
  "ewma_alpha": float,
  "adaptation_count": int
}
```

### Voice Samples Collection

Stores authentication attempts with labels for analysis.

## 🎯 Use Cases

- **Banking & Finance** - Secure phone authentication
- **Access Control** - Voice-based building/device access
- **Call Centers** - Customer identity verification
- **Healthcare** - Patient identity confirmation
- **IoT Devices** - Smart home voice authentication
- **Enterprise Security** - Employee verification

## 🔒 Security Features

1. **File Size Validation** - 2MB limit prevents abuse
2. **Audio Quality Gatekeeper** - Rejects invalid inputs
3. **CORS Protection** - Configurable origins
4. **MongoDB Atlas** - Encrypted cloud database
5. **Configurable Thresholds** - Balance security vs convenience
6. **GPU Acceleration** - Fast processing reduces exposure time

## 🚀 Performance Optimizations

- **GPU Acceleration** - All models run on CUDA when available
- **Model Caching** - Pre-loaded at startup for instant access
- **Batch Processing** - Efficient tensor operations
- **Background Tasks** - EWMA adaptation doesn't block responses
- **Smart Model Selection** - Chooses appropriate models based on speaker count
- **Local Model Storage** - Avoids repeated downloads

## 📊 Audio Processing Pipeline Details

### 1. Audio Quality Gatekeeper
Validates input before processing:
- Volume/energy check (prevents silent recordings)
- Speech presence analysis (minimum 4s of speech)
- VAD-based validation

### 2. Optimal Speech Segment Selection
For recordings > 12 seconds:
- Sliding window analysis (0.5s steps)
- Speech density calculation
- Selects best 10-12s segment
- Maximizes speech content

### 3. Multi-Speaker Handling
- Detects number of speakers via diarization
- Separates overlapping speech using Sepformer
- Selects correct speaker using enrolled embeddings
- Energy-based fallback for unknown scenarios

### 4. Enhancement Pipeline
- Noise reduction using MetricGAN+
- Spectral mask-based enhancement
- Preserves speech quality
- GPU-accelerated processing

## 🧪 Testing

Run the included test files:

```bash
# Test complete system
python test_complete_system.py

# Test GPU acceleration
python test_gpu_acceleration.py

# Test advanced pipeline
python test_advanced_pipeline_local.py

# Test multi-speaker handling
python test_enhanced_multispeaker.py

# Test MongoDB connection
python test_mongodb.py
```

## 📁 Project Structure

```
voice-authentication-system/
├── main.py                          # FastAPI application entry point
├── models.py                        # Model management and caching
├── router.py                        # API endpoints and processing logic
├── database.py                      # MongoDB operations
├── enhanced_web.html                # Web interface
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (not in repo)
├── .gitignore                       # Git ignore rules
├── pretrained_models/               # Cached AI models
│   ├── spkrec-ecapa-voxceleb/
│   ├── sepformer-wsj02mix/
│   ├── sepformer-wsj03mix/
│   ├── metricgan-plus-voicebank/
│   ├── vad/
│   └── pyannote/
├── local_models/                    # Custom model implementations
│   └── spoof_detector/
├── test_*.py                        # Test scripts
└── new_venv/                        # Virtual environment
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Savindu MDK** - [GitHub](https://github.com/Savindumdk)

## 🙏 Acknowledgments

- [SpeechBrain](https://speechbrain.github.io/) - Speech processing toolkit
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [HuggingFace](https://huggingface.co/) - Model hosting
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

## 🔄 Version History

- **v2.0.0** (Current)
  - Multi-sample enrollment
  - EWMA adaptive learning
  - Advanced multi-speaker handling
  - Audio quality gatekeeper
  - Optimal speech segment selection
  - GPU acceleration
  - Labeled data collection

## 🗺️ Roadmap

- [ ] Anti-spoofing detection
- [ ] Real-time streaming authentication
- [ ] Mobile SDK
- [ ] Multi-language support
- [ ] Voice cloning detection
- [ ] Federated learning support
- [ ] Docker containerization
- [ ] Kubernetes deployment configs

---

**Built with ❤️ using PyTorch, SpeechBrain, and FastAPI**
