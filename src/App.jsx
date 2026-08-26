import { useState, useRef, useEffect, useCallback } from 'react'
import Webcam from 'react-webcam'
import * as faceapi from 'face-api.js'

import {
  Camera,
  CameraOff,
  RotateCcw,
  Download,
  Settings,
  Brain,
  UserRound,
  Activity,
  CheckCircle,
  AlertCircle,
  HeartPulse,
  Droplets,
  Stethoscope,
  ShieldCheck,
  Info
} from 'lucide-react'

import logo from './assets/etouchus_face_recognition_logo.png'
import './App.css'

function App() {
  // ============================================================
  // STATE
  // ============================================================

  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [detections, setDetections] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)
  const [capturedImage, setCapturedImage] = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  const [confidence, setConfidence] = useState(0.5)

  // Camera state
  const [cameraReady, setCameraReady] = useState(false)

  // Health measurements
  const [heartRate] = useState(null)
  const [spo2] = useState(null)

  const [bloodPressure] = useState({
    systolic: null,
    diastolic: null
  })

  const [heartRateStatus] = useState('Not measured')
  const [spo2Status] = useState('Waiting for sensor')
  const [bloodPressureStatus] = useState('Waiting for sensor')

  // ============================================================
  // REFS
  // ============================================================

  const webcamRef = useRef(null)
  const canvasRef = useRef(null)

  const processingRef = useRef(false)
  const timerRef = useRef(null)

  // ============================================================
  // LOAD FACE API MODELS
  // ============================================================

  useEffect(() => {
    let mounted = true

    const loadModels = async () => {
      try {
        setError(null)

        console.log('Loading face-api.js models...')

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
          faceapi.nets.faceExpressionNet.loadFromUri('/models'),
          faceapi.nets.ageGenderNet.loadFromUri('/models')
        ])

        if (mounted) {
          setIsModelLoaded(true)
          console.log('Models loaded successfully')
        }
      } catch (err) {
        console.error('Model loading error:', err)

        if (mounted) {
          setError(
            'Failed to load AI models. Make sure the model files are inside public/models.'
          )
        }
      }
    }

    loadModels()

    return () => {
      mounted = false
    }
  }, [])

  // ============================================================
  // CAMERA SUCCESS
  // ============================================================

  const handleCameraReady = () => {
    console.log('Webcam is ready')

    setCameraReady(true)
    setError(null)
  }

  // ============================================================
  // CAMERA ERROR
  // ============================================================

  const handleCameraError = (err) => {
    console.error('Webcam error:', err)

    setCameraReady(false)

    setError(
      'Unable to access webcam. Please allow camera permission in your browser.'
    )

    setIsCameraOn(false)
  }

  // ============================================================
  // PROCESS VIDEO
  // ============================================================

  const processVideo = useCallback(async () => {
    if (
      !webcamRef.current ||
      !canvasRef.current ||
      !isModelLoaded ||
      !isCameraOn ||
      !cameraReady
    ) {
      return
    }

    // Prevent overlapping detection
    if (processingRef.current) {
      return
    }

    const video = webcamRef.current.video
    const canvas = canvasRef.current

    if (!video) {
      return
    }

    if (video.readyState !== 4) {
      return
    }

    const width = video.videoWidth
    const height = video.videoHeight

    if (!width || !height) {
      return
    }

    try {
      processingRef.current = true
      setIsProcessing(true)

      const displaySize = {
        width,
        height
      }

      // Set canvas size
      if (
        canvas.width !== width ||
        canvas.height !== height
      ) {
        faceapi.matchDimensions(
          canvas,
          displaySize
        )
      }

      const results = await faceapi
        .detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 224,
            scoreThreshold: confidence
          })
        )
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender()

      // Save detections
      setDetections(results)

      // Resize detections
      const resizedDetections =
        faceapi.resizeResults(
          results,
          displaySize
        )

      const ctx = canvas.getContext('2d')

      if (!ctx) {
        return
      }

      // Clear canvas
      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      )

      // Draw detections
      resizedDetections.forEach(
        (detection, index) => {
          const box =
            detection.detection.box

          const age =
            Math.round(
              detection.age
            )

          const gender =
            detection.gender

          const confidenceValue =
            Math.round(
              detection.detection.score * 100
            )

          const genderProbability =
            Math.round(
              detection.genderProbability * 100
            )

          // Face box
          const drawBox =
            new faceapi.draw.DrawBox(
              box,
              {
                label:
                  `Face ${index + 1} | ` +
                  `Age: ${age} | ` +
                  `Gender: ${gender} | ` +
                  `Confidence: ${confidenceValue}%`
              }
            )

          drawBox.draw(canvas)

          // Landmarks
          const drawLandmarks =
            new faceapi.draw.DrawFaceLandmarks(
              box,
              detection.landmarks
            )

          drawLandmarks.draw(canvas)

          // Face number
          ctx.fillStyle = '#22c55e'
          ctx.font = 'bold 14px Arial'

          ctx.fillText(
            `Face ${index + 1}`,
            box.x,
            Math.max(
              20,
              box.y - 10
            )
          )

          // Prevent unused variable warning
          void genderProbability
        }
      )
    } catch (err) {
      console.error(
        'Face detection error:',
        err
      )
    } finally {
      processingRef.current = false
      setIsProcessing(false)
    }
  }, [
    isModelLoaded,
    isCameraOn,
    cameraReady,
    confidence
  ])

  // ============================================================
  // DETECTION LOOP
  // ============================================================

  useEffect(() => {
    if (
      !isCameraOn ||
      !isModelLoaded ||
      !cameraReady
    ) {
      return
    }

    let stopped = false

    const runDetection = async () => {
      if (stopped) {
        return
      }

      await processVideo()

      if (!stopped) {
        timerRef.current =
          setTimeout(
            runDetection,
            300
          )
      }
    }

    runDetection()

    return () => {
      stopped = true

      if (timerRef.current) {
        clearTimeout(
          timerRef.current
        )
      }

      processingRef.current = false
    }
  }, [
    isCameraOn,
    isModelLoaded,
    cameraReady,
    processVideo
  ])

  // ============================================================
  // TOGGLE CAMERA
  // ============================================================

  const toggleCamera = () => {
    setError(null)

    if (isCameraOn) {
      // STOP
      if (timerRef.current) {
        clearTimeout(
          timerRef.current
        )
      }

      processingRef.current = false

      setCameraReady(false)
      setIsCameraOn(false)
      setDetections([])
      setCapturedImage(null)
    } else {
      // START
      setCapturedImage(null)
      setDetections([])
      setCameraReady(false)
      setIsCameraOn(true)
    }
  }

  // ============================================================
  // CAPTURE IMAGE
  // ============================================================

  const captureImage = () => {
    if (!webcamRef.current) {
      setError(
        'Camera is not available.'
      )
      return
    }

    if (!cameraReady) {
      setError(
        'Please wait until the camera is ready.'
      )
      return
    }

    const imageSrc =
      webcamRef.current.getScreenshot()

    if (!imageSrc) {
      setError(
        'Unable to capture image.'
      )
      return
    }

    setCapturedImage(imageSrc)
    setError(null)

    console.log(
      'Image captured successfully'
    )
  }

  // ============================================================
  // DOWNLOAD IMAGE
  // ============================================================

  const downloadImage = () => {
    if (!capturedImage) {
      return
    }

    const link =
      document.createElement('a')

    link.download =
      'face-detection.jpg'

    link.href =
      capturedImage

    document.body.appendChild(link)

    link.click()

    document.body.removeChild(link)
  }

  // ============================================================
  // RESET
  // ============================================================

  const resetApp = () => {
    if (timerRef.current) {
      clearTimeout(
        timerRef.current
      )
    }

    processingRef.current = false

    setIsCameraOn(false)
    setCameraReady(false)
    setDetections([])
    setCapturedImage(null)
    setError(null)
    setShowSettings(false)
  }

  // ============================================================
  // VIDEO CONSTRAINTS
  // ============================================================

  const videoConstraints = {
    width: {
      ideal: 1280
    },
    height: {
      ideal: 720
    },
    facingMode: 'user'
  }

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="app">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="header">

        <div className="header-content">

          <div className="brand">

            <div className="logo-container">

              <img
                src={logo}
                alt="eTouchUS Face Recognition"
                className="logo"
              />

            </div>

            <div className="brand-text">

              <h1>
                AI Face & Health Analyzer
              </h1>

              <p>
                Real-time facial analysis and health monitoring
              </p>

            </div>

          </div>

          <div
            className={
              `model-status ${
                isModelLoaded
                  ? 'ready'
                  : 'loading'
              }`
            }
          >

            {isModelLoaded ? (
              <CheckCircle size={17} />
            ) : (
              <Brain size={17} />
            )}

            <span>
              {isModelLoaded
                ? 'AI Model Ready'
                : 'Loading AI Model...'}
            </span>

          </div>

        </div>

      </header>

      {/* ======================================================
          MAIN
      ====================================================== */}

      <main className="main">

        {/* PAGE TITLE */}

        <section className="page-title">

          <div>

            <h2>
              Facial Recognition & Health
            </h2>

            <p>
              Analyze facial attributes and connect validated
              health measurement systems.
            </p>

          </div>

        </section>

        {/* ====================================================
            CONTROLS
        ==================================================== */}

        <section className="control-section">

          <div className="controls">

            {/* START / STOP */}

            <button
              onClick={toggleCamera}
              className={
                `control-btn ${
                  isCameraOn
                    ? 'danger'
                    : 'primary'
                }`
              }
              disabled={!isModelLoaded}
            >

              {isCameraOn ? (
                <CameraOff size={19} />
              ) : (
                <Camera size={19} />
              )}

              {isCameraOn
                ? 'Stop Camera'
                : 'Start Camera'}

            </button>

            {/* CAPTURE */}

            {isCameraOn && (
              <button
                onClick={captureImage}
                className="control-btn secondary"
                disabled={!cameraReady}
              >

                <Camera size={19} />

                Capture

              </button>
            )}

            {/* SETTINGS */}

            {isCameraOn && (
              <button
                onClick={() =>
                  setShowSettings(
                    current => !current
                  )
                }
                className="control-btn secondary"
              >

                <Settings size={19} />

                Settings

              </button>
            )}

            {/* RESET */}

            {isCameraOn && (
              <button
                onClick={resetApp}
                className="control-btn secondary"
              >

                <RotateCcw size={19} />

                Reset

              </button>
            )}

          </div>

          {/* ==================================================
              SETTINGS
          ================================================== */}

          {showSettings && (

            <div className="settings-panel">

              <div className="settings-title">

                <Settings size={19} />

                <h3>
                  Detection Settings
                </h3>

              </div>

              <div className="setting-item">

                <div className="setting-label">

                  <span>
                    Confidence Threshold
                  </span>

                  <strong>
                    {Math.round(
                      confidence * 100
                    )}%
                  </strong>

                </div>

                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.1"
                  value={confidence}
                  onChange={(e) =>
                    setConfidence(
                      parseFloat(
                        e.target.value
                      )
                    )
                  }
                />

                <div className="range-labels">

                  <span>
                    More sensitive
                  </span>

                  <span>
                    More accurate
                  </span>

                </div>

              </div>

            </div>

          )}

        </section>

        {/* ====================================================
            ERROR
        ==================================================== */}

        {error && (

          <div className="error-message">

            <AlertCircle size={20} />

            <div>

              <strong>
                Error
              </strong>

              <p>
                {error}
              </p>

            </div>

          </div>

        )}

        {/* ====================================================
            MODEL LOADING
        ==================================================== */}

        {!isModelLoaded && !error && (

          <div className="loading">

            <div className="loading-spinner"></div>

            <h3>
              Loading AI Models
            </h3>

            <p>
              Please wait while the face recognition
              models are loaded.
            </p>

          </div>

        )}

        {/* ====================================================
            CAMERA
        ==================================================== */}

        <section className="camera-card">

          <div className="camera-header">

            <div>

              <h3>
                Camera
              </h3>

              <p>
                Position your face in front of the camera.
              </p>

            </div>

            <div
              className={
                `camera-indicator ${
                  isCameraOn && cameraReady
                    ? 'active'
                    : ''
                }`
              }
            >

              <span></span>

              {isCameraOn
                ? cameraReady
                  ? 'Live'
                  : 'Starting...'
                : 'Offline'}

            </div>

          </div>

          <div className="camera-container">

            {isCameraOn ? (

              <div className="video-wrapper">

                {/* REAL WORKING WEBCAM */}

                <Webcam
                  ref={webcamRef}
                  audio={false}
                  mirrored={true}
                  screenshotFormat="image/jpeg"
                  videoConstraints={
                    videoConstraints
                  }
                  className="webcam"
                  onUserMedia={
                    handleCameraReady
                  }
                  onUserMediaError={
                    handleCameraError
                  }
                />

                {/* FACE DETECTION CANVAS */}

                <canvas
                  ref={canvasRef}
                  className="detection-canvas"
                />

                {/* SCAN FRAME */}

                <div className="scan-frame">

                  <span className="corner top-left"></span>

                  <span className="corner top-right"></span>

                  <span className="corner bottom-left"></span>

                  <span className="corner bottom-right"></span>

                </div>

                {/* CAMERA STATUS */}

                {!cameraReady && (

                  <div className="camera-starting">

                    <div className="processing-spinner"></div>

                    <span>
                      Starting camera...
                    </span>

                  </div>

                )}

                {/* PROCESSING */}

                {isProcessing && cameraReady && (

                  <div className="processing-overlay">

                    <div className="processing-spinner"></div>

                    <span>
                      AI scanning...
                    </span>

                  </div>

                )}

              </div>

            ) : (

              <div className="camera-placeholder">

                <div className="placeholder-icon">

                  <Camera size={55} />

                </div>

                <h3>
                  Camera is Off
                </h3>

                <p>
                  Click "Start Camera" to begin
                  face detection.
                </p>

              </div>

            )}

          </div>

        </section>

        {/* ====================================================
            HEALTH DASHBOARD
        ==================================================== */}

        <section className="health-section">

          <div className="section-heading">

            <div>

              <h3>
                Health Monitoring
              </h3>

              <p>
                Vital-sign measurements require a validated
                algorithm or physical sensor.
              </p>

            </div>

            <div className="health-disclaimer">

              <ShieldCheck size={16} />

              Measurement status

            </div>

          </div>

          <div className="health-grid">

            {/* HEART RATE */}

            <div className="health-card heart-card">

              <div className="health-card-top">

                <div className="health-icon">

                  <HeartPulse size={25} />

                </div>

                <span className="health-status">

                  {heartRateStatus}

                </span>

              </div>

              <div className="health-card-title">

                Heart Rate

              </div>

              <div className="health-value">

                {heartRate !== null ? (
                  <>
                    {heartRate}
                    <small>BPM</small>
                  </>
                ) : (
                  <>
                    --
                    <small>BPM</small>
                  </>
                )}

              </div>

              <p className="health-description">

                Webcam rPPG measurement can be
                connected here.

              </p>

              {isCameraOn && (

                <div className="measurement-message">

                  <Activity size={15} />

                  Ready for rPPG analysis

                </div>

              )}

            </div>

            {/* SPO2 */}

            <div className="health-card spo2-card">

              <div className="health-card-top">

                <div className="health-icon">

                  <Droplets size={25} />

                </div>

                <span className="health-status">

                  {spo2Status}

                </span>

              </div>

              <div className="health-card-title">

                Blood Oxygen

              </div>

              <div className="health-value">

                {spo2 !== null ? (
                  <>
                    {spo2}
                    <small>%</small>
                  </>
                ) : (
                  <>
                    --
                    <small>%</small>
                  </>
                )}

              </div>

              <p className="health-description">

                Requires a validated SpO₂ sensor
                or supported measurement model.

              </p>

              <div className="measurement-message">

                <Info size={15} />

                Sensor required

              </div>

            </div>

            {/* BLOOD PRESSURE */}

            <div className="health-card bp-card">

              <div className="health-card-top">

                <div className="health-icon">

                  <Stethoscope size={25} />

                </div>

                <span className="health-status">

                  {bloodPressureStatus}

                </span>

              </div>

              <div className="health-card-title">

                Blood Pressure

              </div>

              <div className="bp-values">

                <div>

                  <strong>
                    {bloodPressure.systolic ?? '--'}
                  </strong>

                  <span>
                    SYS
                  </span>

                </div>

                <div className="bp-slash">
                  /
                </div>

                <div>

                  <strong>
                    {bloodPressure.diastolic ?? '--'}
                  </strong>

                  <span>
                    DIA
                  </span>

                </div>

                <div className="bp-unit">
                  mmHg
                </div>

              </div>

              <p className="health-description">

                Requires a validated blood-pressure
                monitor or approved measurement system.

              </p>

              <div className="measurement-message">

                <Info size={15} />

                BP sensor required

              </div>

            </div>

          </div>

        </section>

        {/* ====================================================
            DETECTION RESULTS
        ==================================================== */}

        {detections.length > 0 && (

          <section className="detections-panel">

            <div className="section-heading">

              <div>

                <h3>
                  Detection Results
                </h3>

                <p>

                  {detections.length}

                  {' '}

                  face
                  {detections.length > 1
                    ? 's'
                    : ''}

                  {' '}detected

                </p>

              </div>

              <div className="detection-status">

                <Activity size={18} />

                Live Detection

              </div>

            </div>

            <div className="detections-grid">

              {detections.map(
                (detection, index) => {

                  const age =
                    Math.round(
                      detection.age
                    )

                  const gender =
                    detection.gender

                  const genderProbability =
                    Math.round(
                      detection.genderProbability *
                      100
                    )

                  const faceConfidence =
                    Math.round(
                      detection.detection.score *
                      100
                    )

                  return (

                    <div
                      key={index}
                      className="detection-card"
                    >

                      <div className="face-card-header">

                        <div className="face-icon">

                          <UserRound size={22} />

                        </div>

                        <div>

                          <h4>
                            Face {index + 1}
                          </h4>

                          <span>
                            Detected
                          </span>

                        </div>

                      </div>

                      <div className="detection-info">

                        <div className="info-item">

                          <span className="label">
                            Estimated Age
                          </span>

                          <span className="value">
                            {age} years
                          </span>

                        </div>

                        <div className="info-item">

                          <span className="label">
                            Gender
                          </span>

                          <span className="value">
                            {gender}
                          </span>

                        </div>

                        <div className="info-item">

                          <span className="label">
                            Gender Confidence
                          </span>

                          <span className="value">
                            {genderProbability}%
                          </span>

                        </div>

                        <div className="info-item">

                          <span className="label">
                            Face Confidence
                          </span>

                          <span className="value">
                            {faceConfidence}%
                          </span>

                        </div>

                      </div>

                    </div>

                  )
                }
              )}

            </div>

          </section>

        )}

        {/* ====================================================
            CAPTURED IMAGE
        ==================================================== */}

        {capturedImage && (

          <section className="captured-section">

            <div className="section-heading">

              <div>

                <h3>
                  Captured Image
                </h3>

                <p>
                  Image captured manually from webcam
                </p>

              </div>

            </div>

            <div className="captured-content">

              <img
                src={capturedImage}
                alt="Captured face"
                className="captured-image"
              />

              <button
                onClick={downloadImage}
                className="control-btn primary"
              >

                <Download size={19} />

                Download Image

              </button>

            </div>

          </section>

        )}

        {/* ====================================================
            MEDICAL NOTICE
        ==================================================== */}

        <div className="medical-notice">

          <ShieldCheck size={20} />

          <div>

            <strong>
              Measurement Notice
            </strong>

            <p>
              Heart rate, SpO₂ and blood-pressure
              values are not produced by face-api.js.
              This dashboard currently provides integration
              points for validated algorithms and physical
              medical sensors. Do not use placeholder values
              for diagnosis or medical decisions.
            </p>

          </div>

        </div>

      </main>

      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer className="footer">

        <div className="footer-content">

          <img
            src={logo}
            alt="eTouchUS"
            className="footer-logo"
          />

          <div>

            <strong>
              eTouchUS AI Face & Health Analyzer
            </strong>

            <p>
              Powered by eTouchUS
            </p>

          </div>

        </div>

      </footer>

    </div>
  )
}

export default App