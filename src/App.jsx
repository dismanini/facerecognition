import { useState, useRef, useEffect } from 'react'
import Webcam from 'react-webcam'
import * as faceapi from 'face-api.js'
import { Camera, CameraOff, RotateCcw, Download, Settings } from 'lucide-react'
import './App.css'

function App() {
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [detections, setDetections] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)
  const [capturedImage, setCapturedImage] = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  const [confidence, setConfidence] = useState(0.5)

  const webcamRef = useRef(null)
  const canvasRef = useRef(null)

  // Load face-api.js models
  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
          faceapi.nets.faceExpressionNet.loadFromUri('/models'),
          faceapi.nets.ageGenderNet.loadFromUri('/models'),
        ])
        setIsModelLoaded(true)
        console.log('Models loaded successfully')
      } catch (error) {
        console.error('Error loading models:', error)
        setError('Failed to load AI models. Please refresh the page.')
      }
    }

    loadModels()
  }, [])

  // Process video stream
  const processVideo = async () => {
    if (!webcamRef.current || !canvasRef.current || !isModelLoaded || !isCameraOn) return

    const video = webcamRef.current.video
    const canvas = canvasRef.current

    if (video.readyState === 4) {
      const displaySize = { width: video.videoWidth, height: video.videoHeight }
      faceapi.matchDimensions(canvas, displaySize)

      try {
        setIsProcessing(true)
        const detections = await faceapi
          .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224 }))
          .withFaceLandmarks()
          .withFaceExpressions()
          .withAgeAndGender()

        setDetections(detections)

        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        
        // Clear canvas
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // Draw detections
        resizedDetections.forEach(detection => {
          const box = detection.detection.box
          const drawBox = new faceapi.draw.DrawBox(box, { 
            label: `Age: ${Math.round(detection.age)} | Gender: ${detection.gender} | Confidence: ${Math.round(detection.genderProbability * 100)}%`
          })
          drawBox.draw(canvas)
        })

      } catch (error) {
        console.error('Error processing video:', error)
      } finally {
        setIsProcessing(false)
      }
    }
  }

  // Start video processing
  useEffect(() => {
    if (isCameraOn && isModelLoaded) {
      const interval = setInterval(processVideo, 100)
      return () => clearInterval(interval)
    }
  }, [isCameraOn, isModelLoaded])

  // Toggle camera
  const toggleCamera = () => {
    setIsCameraOn(!isCameraOn)
    setDetections([])
    setCapturedImage(null)
    setError(null)
  }

  // Capture image
  const captureImage = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot()
      setCapturedImage(imageSrc)
    }
  }

  // Download captured image
  const downloadImage = () => {
    if (capturedImage) {
      const link = document.createElement('a')
      link.download = 'face-detection.jpg'
      link.href = capturedImage
      link.click()
    }
  }

  // Reset everything
  const resetApp = () => {
    setIsCameraOn(false)
    setDetections([])
    setCapturedImage(null)
    setError(null)
  }

  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'user'
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>🤖 AI Face Detector</h1>
          <p>Real-time face detection, age estimation, and gender recognition</p>
        </div>
      </header>

      <main className="main">
        <div className="controls">
          <button 
            onClick={toggleCamera}
            className={`control-btn ${isCameraOn ? 'active' : ''}`}
            disabled={!isModelLoaded}
          >
            {isCameraOn ? <CameraOff size={20} /> : <Camera size={20} />}
            {isCameraOn ? 'Stop Camera' : 'Start Camera'}
          </button>

          {isCameraOn && (
            <>
              <button onClick={captureImage} className="control-btn secondary">
                📸 Capture
              </button>
              <button onClick={resetApp} className="control-btn secondary">
                <RotateCcw size={20} />
                Reset
              </button>
              <button 
                onClick={() => setShowSettings(!showSettings)}
                className="control-btn secondary"
              >
                <Settings size={20} />
                Settings
              </button>
            </>
          )}
        </div>

        {showSettings && (
          <div className="settings-panel">
            <h3>Detection Settings</h3>
            <div className="setting-item">
              <label>Confidence Threshold: {Math.round(confidence * 100)}%</label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
              />
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            <p>⚠️ {error}</p>
          </div>
        )}

        {!isModelLoaded && (
          <div className="loading">
            <div className="loading-spinner"></div>
            <p>Loading AI models...</p>
          </div>
        )}

        <div className="camera-container">
          {isCameraOn ? (
            <div className="video-wrapper">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                className="webcam"
              />
              <canvas
                ref={canvasRef}
                className="detection-canvas"
              />
              {isProcessing && (
                <div className="processing-overlay">
                  <div className="processing-spinner"></div>
                  <span>Processing...</span>
                </div>
              )}
            </div>
          ) : (
            <div className="camera-placeholder">
              <Camera size={64} />
              <p>Click "Start Camera" to begin face detection</p>
            </div>
          )}
        </div>

        {capturedImage && (
          <div className="captured-image">
            <h3>📸 Captured Image</h3>
            <img src={capturedImage} alt="Captured" />
            <button onClick={downloadImage} className="control-btn">
              <Download size={20} />
              Download
            </button>
          </div>
        )}

        {detections.length > 0 && (
          <div className="detections-panel">
            <h3>🔍 Detection Results</h3>
            <div className="detections-grid">
              {detections.map((detection, index) => (
                <div key={index} className="detection-card">
                  <div className="detection-info">
                    <div className="info-item">
                      <span className="label">Age:</span>
                      <span className="value">{Math.round(detection.age)} years</span>
                    </div>
                    <div className="info-item">
                      <span className="label">Gender:</span>
                      <span className="value">{detection.gender}</span>
                    </div>
                    <div className="info-item">
                      <span className="label">Confidence:</span>
                      <span className="value">{Math.round(detection.genderProbability * 100)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by face-api.js & React • Built with ❤️</p>
      </footer>
    </div>
  )
}

export default App
