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
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [cameraReady, setCameraReady] = useState(false)

  const [detections, setDetections] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)

  const [error, setError] = useState(null)
  const [capturedImage, setCapturedImage] = useState(null)

  const [showSettings, setShowSettings] = useState(false)
  const [confidence, setConfidence] = useState(0.3)

  const [heartRate] = useState(null)
  const [spo2] = useState(null)

  const [bloodPressure] = useState({
    systolic: null,
    diastolic: null
  })

  const webcamRef = useRef(null)
  const canvasRef = useRef(null)

  const processingRef = useRef(false)
  const timerRef = useRef(null)

  // ============================================================
  // LOAD MODELS
  // ============================================================

  useEffect(() => {
    let cancelled = false

    const loadModels = async () => {
      try {
        setError(null)

        console.log('Loading models...')

        await faceapi.nets.tinyFaceDetector.loadFromUri('/models')
        console.log('Tiny face detector loaded')

        await faceapi.nets.faceLandmark68Net.loadFromUri('/models')
        console.log('Face landmarks loaded')

        await faceapi.nets.faceExpressionNet.loadFromUri('/models')
        console.log('Expressions loaded')

        await faceapi.nets.ageGenderNet.loadFromUri('/models')
        console.log('Age/Gender loaded')

        if (!cancelled) {
          setIsModelLoaded(true)

          // AUTOMATICALLY START CAMERA
          setIsCameraOn(true)

          console.log('ALL MODELS LOADED')
          console.log('Starting camera automatically...')
        }

      } catch (err) {
        console.error('MODEL ERROR:', err)

        if (!cancelled) {
          setError(
            'Failed to load AI models. Check your public/models folder.'
          )
        }
      }
    }

    loadModels()

    return () => {
      cancelled = true
    }
  }, [])

  // ============================================================
  // CAMERA READY
  // ============================================================

  const handleCameraReady = () => {
    console.log('CAMERA READY')

    setCameraReady(true)
    setError(null)
  }

  // ============================================================
  // CAMERA ERROR
  // ============================================================

  const handleCameraError = (err) => {
    console.error('CAMERA ERROR:', err)

    setCameraReady(false)
    setIsCameraOn(false)

    setError(
      'Camera could not be started. Please allow camera permission in your browser.'
    )
  }

  // ============================================================
  // FACE DETECTION
  // ============================================================

  const detectFace = useCallback(async () => {
    if (!isCameraOn) return
    if (!cameraReady) return
    if (!isModelLoaded) return

    if (processingRef.current) return

    if (!webcamRef.current) return

    const video = webcamRef.current.video

    if (!video) return

    if (video.readyState !== 4) return

    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return
    }

    processingRef.current = true
    setIsProcessing(true)

    try {
      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight
      }

      const canvas = canvasRef.current

      if (!canvas) return

      canvas.width = displaySize.width
      canvas.height = displaySize.height

      faceapi.matchDimensions(
        canvas,
        displaySize
      )

      // FACE DETECTION
      const results = await faceapi
        .detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 320,
            scoreThreshold: confidence
          })
        )
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender()

      console.log(
        'Faces detected:',
        results.length
      )

      setDetections(results)

      // RESIZE
      const resizedResults =
        faceapi.resizeResults(
          results,
          displaySize
        )

      // CLEAR CANVAS
      const ctx =
        canvas.getContext('2d')

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      )

      // DRAW DETECTION
      resizedResults.forEach(
        (detection, index) => {

          const box =
            detection.detection.box

          const confidenceValue =
            Math.round(
              detection.detection.score * 100
            )

          const age =
            Math.round(detection.age)

          const gender =
            detection.gender

          const genderConfidence =
            Math.round(
              detection.genderProbability * 100
            )

          const label =
            `Face ${index + 1} | ` +
            `Age: ${age} | ` +
            `Gender: ${gender} | ` +
            `${confidenceValue}%`

          const drawBox =
            new faceapi.draw.DrawBox(
              box,
              {
                label,
                boxColor: '#22c55e',
                lineWidth: 3
              }
            )

          drawBox.draw(canvas)

          // LANDMARKS
          if (detection.landmarks) {

            const drawLandmarks =
              new faceapi.draw.DrawFaceLandmarks(
                detection.landmarks
              )

            drawLandmarks.draw(canvas)
          }

          console.log(
            `Face ${index + 1}:`,
            {
              age,
              gender,
              genderConfidence,
              confidence: confidenceValue
            }
          )
        }
      )

    } catch (err) {

      console.error(
        'FACE DETECTION ERROR:',
        err
      )

    } finally {

      processingRef.current = false
      setIsProcessing(false)

    }

  }, [
    isCameraOn,
    cameraReady,
    isModelLoaded,
    confidence
  ])

  // ============================================================
  // DETECTION LOOP
  // ============================================================

  useEffect(() => {

    if (
      !isCameraOn ||
      !cameraReady ||
      !isModelLoaded
    ) {
      return
    }

    console.log(
      'Starting face detection loop...'
    )

    let stopped = false

    const loop = async () => {

      if (stopped) return

      await detectFace()

      if (!stopped) {

        timerRef.current =
          setTimeout(
            loop,
            500
          )
      }
    }

    loop()

    return () => {

      stopped = true

      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }

      processingRef.current = false

    }

  }, [
    isCameraOn,
    cameraReady,
    isModelLoaded,
    detectFace
  ])

  // ============================================================
  // CAMERA TOGGLE
  // ============================================================

  const toggleCamera = () => {

    setError(null)

    if (isCameraOn) {

      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }

      setIsCameraOn(false)
      setCameraReady(false)
      setDetections([])

      processingRef.current = false

    } else {

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
      setError('Camera is not available.')
      return
    }

    if (!cameraReady) {
      setError('Camera is still starting.')
      return
    }

    const image =
      webcamRef.current.getScreenshot()

    if (!image) {
      setError('Unable to capture image.')
      return
    }

    setCapturedImage(image)
    setError(null)
  }

  // ============================================================
  // DOWNLOAD
  // ============================================================

  const downloadImage = () => {

    if (!capturedImage) return

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
      clearTimeout(timerRef.current)
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
  // CAMERA CONSTRAINTS
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
  // UI
  // ============================================================

  return (
    <div className="app">

      {/* HEADER */}

      <header className="header">

        <div className="header-content">

          <div className="brand">

            <div className="logo-container">

              <img
                src={logo}
                alt="eTouchUS"
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

      {/* MAIN */}

      <main className="main">

        <section className="page-title">

          <div>

            <h2>
              Facial Recognition & Health
            </h2>

            <p>
              Camera starts automatically after AI models load.
            </p>

          </div>

        </section>

        {/* CONTROLS */}

        <section className="control-section">

          <div className="controls">

            <button
              onClick={toggleCamera}
              disabled={!isModelLoaded}
              className={
                `control-btn ${
                  isCameraOn
                    ? 'danger'
                    : 'primary'
                }`
              }
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

            {isCameraOn && (

              <>

                <button
                  onClick={captureImage}
                  className="control-btn secondary"
                >

                  <Camera size={19} />

                  Capture

                </button>

                <button
                  onClick={() =>
                    setShowSettings(
                      !showSettings
                    )
                  }
                  className="control-btn secondary"
                >

                  <Settings size={19} />

                  Settings

                </button>

                <button
                  onClick={resetApp}
                  className="control-btn secondary"
                >

                  <RotateCcw size={19} />

                  Reset

                </button>

              </>

            )}

          </div>

          {/* SETTINGS */}

          {showSettings && (

            <div className="settings-panel">

              <h3>
                Detection Settings
              </h3>

              <label>

                Confidence Threshold:
                {' '}
                {Math.round(
                  confidence * 100
                )}%

              </label>

              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                value={confidence}
                onChange={(e) =>
                  setConfidence(
                    Number(e.target.value)
                  )
                }
              />

            </div>

          )}

        </section>

        {/* ERROR */}

        {error && (

          <div className="error-message">

            <AlertCircle size={20} />

            <span>
              {error}
            </span>

          </div>

        )}

        {/* MODEL LOADING */}

        {!isModelLoaded && !error && (

          <div className="loading">

            <div className="loading-spinner"></div>

            <h3>
              Loading AI Models
            </h3>

            <p>
              Please wait...
            </p>

          </div>

        )}

        {/* CAMERA */}

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
                  cameraReady
                    ? 'active'
                    : ''
                }`
              }
            >

              <span></span>

              {cameraReady
                ? 'Live'
                : 'Starting...'}

            </div>

          </div>

          <div className="camera-container">

            {isCameraOn ? (

              <div className="video-wrapper">

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

                <canvas
                  ref={canvasRef}
                  className="detection-canvas"
                />

                {!cameraReady && (

                  <div className="processing-overlay">

                    <div className="processing-spinner"></div>

                    <span>
                      Starting camera...
                    </span>

                  </div>

                )}

                {isProcessing && cameraReady && (

                  <div className="processing-overlay">

                    <div className="processing-spinner"></div>

                    <span>
                      Detecting face...
                    </span>

                  </div>

                )}

              </div>

            ) : (

              <div className="camera-placeholder">

                <Camera size={60} />

                <h3>
                  Camera is Off
                </h3>

                <p>
                  Click "Start Camera" to begin.
                </p>

              </div>

            )}

          </div>

        </section>

        {/* DETECTION RESULTS */}

        {detections.length > 0 && (

          <section className="detections-panel">

            <div className="section-heading">

              <div>

                <h3>
                  Detection Results
                </h3>

                <p>
                  {detections.length} face
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

                  const faceConfidence =
                    Math.round(
                      detection.detection.score * 100
                    )

                  const genderConfidence =
                    Math.round(
                      detection.genderProbability * 100
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
                            Age
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
                            Face Confidence
                          </span>

                          <span className="value">
                            {faceConfidence}%
                          </span>

                        </div>

                        <div className="info-item">

                          <span className="label">
                            Gender Confidence
                          </span>

                          <span className="value">
                            {genderConfidence}%
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

        {/* CAPTURED IMAGE */}

        {capturedImage && (

          <section className="captured-section">

            <h3>
              Captured Image
            </h3>

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

          </section>

        )}

        {/* HEALTH */}

        <section className="health-section">

          <div className="section-heading">

            <div>

              <h3>
                Health Monitoring
              </h3>

              <p>
                Requires validated algorithms or physical sensors.
              </p>

            </div>

            <ShieldCheck size={20} />

          </div>

          <div className="health-grid">

            <div className="health-card">

              <HeartPulse size={28} />

              <h3>
                Heart Rate
              </h3>

              <div className="health-value">

                {heartRate ?? '--'}

                <small>
                  BPM
                </small>

              </div>

              <p>
                Requires a validated rPPG algorithm or sensor.
              </p>

            </div>

            <div className="health-card">

              <Droplets size={28} />

              <h3>
                Blood Oxygen
              </h3>

              <div className="health-value">

                {spo2 ?? '--'}

                <small>
                  %
                </small>

              </div>

              <p>
                Requires a validated SpO₂ sensor/model.
              </p>

            </div>

            <div className="health-card">

              <Stethoscope size={28} />

              <h3>
                Blood Pressure
              </h3>

              <div className="bp-values">

                <strong>
                  {bloodPressure.systolic ?? '--'}
                </strong>

                <span>/</span>

                <strong>
                  {bloodPressure.diastolic ?? '--'}
                </strong>

                <small>
                  mmHg
                </small>

              </div>

              <p>
                Requires a validated BP monitor/model.
              </p>

            </div>

          </div>

        </section>

        {/* MEDICAL NOTICE */}

        <div className="medical-notice">

          <Info size={20} />

          <div>

            <strong>
              Medical Measurement Notice
            </strong>

            <p>
              Face-api.js provides facial detection,
              landmarks, expressions, age and gender
              estimation. It does not independently measure
              blood pressure, SpO₂ or heart rate.
            </p>

          </div>

        </div>

      </main>

      {/* FOOTER */}

      <footer className="footer">

        <img
          src={logo}
          alt="eTouchUS"
          className="footer-logo"
        />

        <p>
          eTouchUS AI Face & Health Analyzer
        </p>

      </footer>

    </div>
  )
}

export default App