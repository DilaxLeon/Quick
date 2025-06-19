# CaptionsApp

A modern web application for automatic video captioning with a sleek black and white UI.

## Features

### Video Processing
- Supports video uploads up to 1GB
- Accepts multiple video formats (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
- Converts videos to 9:16 aspect ratio using center cropping
- Keeps original aspect ratio if already 9:16
- Transcribes audio using OpenAI Whisper base model
- Adds synchronized captions with timestamps
- Limits captions to 3 words per phrase
- Positions captions at 70% of the vertical screen
- Active speaker tracking (Pro plan only) using YOLOv8 and pyannote.audio
- Automatically centers the active speaker in the frame

### User Interface
- Modern black and white high-tech theme
- Responsive design for all devices
- Drag and drop video uploads
- Real-time upload progress tracking
- Video history management
- Caption customization options
- SRT file downloads

## Tech Stack

### Frontend
- React with Vite
- Styled Components for styling
- Framer Motion for animations
- React Router for navigation
- Axios for API requests

### Backend
- Flask Python backend
- OpenAI Whisper for speech-to-text
- FFmpeg for video processing
- YOLOv8 for person detection and speaker tracking
- pyannote.audio for speaker diarization

## Installation

### Prerequisites
- Node.js (v14+)
- Python 3.7+
- FFmpeg (required for MoviePy)
- Sufficient disk space for video processing

### Setup

1. Clone this repository
2. Install the backend dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install the frontend dependencies:
   ```
   cd frontend
   npm install
   ```
4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Hugging Face token to the `.env` file (required for speaker tracking)
   - You can get a token at https://huggingface.co/settings/tokens

## Usage

### Development Mode

1. Start the backend server:
   ```
   python app.py
   ```
2. In a separate terminal, start the frontend development server:
   ```
   cd frontend
   npm run dev
   ```
3. Open your web browser and navigate to `http://localhost:3000/`

### Production Mode

1. Build the frontend:
   ```
   cd frontend
   npm run build
   ```
2. Start the Flask application:
   ```
   python app.py
   ```
3. Open your web browser and navigate to `http://127.0.0.1:5000/`

## Project Structure

```
captionsapp/
├── app.py                # Flask backend entry point
├── requirements.txt      # Python dependencies
├── package.json          # Node.js dependencies
├── frontend/             # React frontend
│   ├── src/              # Source code
│   │   ├── components/   # Reusable UI components
│   │   ├── pages/        # Page components
│   │   ├── hooks/        # Custom React hooks
│   │   ├── utils/        # Utility functions
│   │   ├── assets/       # Static assets
│   │   ├── App.jsx       # Main App component
│   │   └── main.jsx      # Entry point
│   ├── public/           # Public assets
│   └── index.html        # HTML template
└── processed/            # Processed videos and captions
```

## Note

Processing large videos may take several minutes depending on your hardware. The application uses the Whisper base model for transcription, which offers a good balance between accuracy and speed.

## Speaker Tracking Feature

The Pro plan includes an active speaker tracking feature that:

1. Uses YOLOv8 to detect people in the video
2. Uses pyannote.audio for speaker diarization to identify who is speaking when
3. Matches the audio speaker with the visual person
4. Dynamically crops the video to keep the active speaker centered
5. Smoothly transitions between speakers

This feature requires:
- A Hugging Face token with access to the pyannote.audio models
- Sufficient GPU resources for optimal performance (CPU mode is supported but slower)

To enable speaker tracking:
1. Set your Hugging Face token in the `.env` file
2. Toggle the "Active Speaker Tracking" option in the UI (Pro plan only)
3. Process your video as usual