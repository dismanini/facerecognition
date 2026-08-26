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
  const [cameraReady, setCameraReady] = useState(false)

  const [detections, setDetections] = useState([])
  const [capturedDetections, setCapturedDetections] = useState([])

  const [capturedImage, setCapturedImage] = useState(null)

  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)

  const [showSettings, setShowSettings] = useState(false)
  const [confidence, setConfidence] = useState(0.5)

  // ============================================================
  // HEALTH
  // ============================================================

  const [heartRate] = useState(null)
  const [spo2] = useState(null)

  const [bloodPressure] = useState({
    systolic: null,
    diastolic: null
  })

  const [heartRateStatus] =
    useState('Not measured')

  const [spo2Status] =
    useState('Waiting for sensor')

  const [bloodPressureStatus] =
    useState('Waiting for sensor')

  // ============================================================
  // REFS
  // ============================================================

  const webcamRef = useRef(null)
  const canvasRef = useRef(null)

  const processingRef = useRef(false)
  const animationRef = useRef(null)

  // ============================================================
  // LOAD MODELS
  // ============================================================

  useEffect(() => {
    const loadModels = async () => {
      try {
        setError(null)

        console.log('Loading AI models...')

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
          faceapi.nets.faceExpressionNet.loadFromUri('/models'),
          faceapi.nets.ageGenderNet.loadFromUri('/models')
        ])

        setIsModelLoaded(true)

        console.log('AI models loaded successfully')
      } catch (err) {
        console.error('Model loading error:', err)

        setError(
          'Failed to load AI models. Check public/models.'
        )
      }
    }

    loadModels()
  }, [])

  // ============================================================
  // CAMERA STARTED
  // ============================================================

  const handleCameraReady = () => {
    console.log('Camera started successfully')

    setCameraReady(true)
    setError(null)
  }

  // ============================================================
  // CAMERA ERROR
  // ============================================================

  const handleCameraError = (err) => {
    console.error('Webcam error:', err)

    setCameraReady(false)

    setIsCameraOn(false)

    setError(
      'Unable to access webcam. Please allow camera permission in your browser.'
    )
  }

  // ============================================================
  // FACE DETECTION
  // ============================================================

  const processVideo = useCallback(async () => {
    if (!isCameraOn) {
      return
    }

    if (!cameraReady) {
      return
    }

    if (!isModelLoaded) {
      return
    }

    if (!webcamRef.current) {
      return
    }

    if (!canvasRef.current) {
      return
    }

    if (processingRef.current) {
      return
    }

    const video =
      webcamRef.current.video

    const canvas =
      canvasRef.current

    if (!video) {
      return
    }

    if (video.readyState !== 4) {
      return
    }

    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return
    }

    try {
      processingRef.current = true

      setIsProcessing(true)

      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight
      }

      canvas.width =
        displaySize.width

      canvas.height =
        displaySize.height

      faceapi.matchDimensions(
        canvas,
        displaySize
      )

      const results =
        await faceapi
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

      // ========================================================
      // SAVE LIVE DETECTION
      // ========================================================

      setDetections(results)

      // ========================================================
      // DRAW
      // ========================================================

      const resized =
        faceapi.resizeResults(
          results,
          displaySize
        )

      const ctx =
        canvas.getContext('2d')

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      )

      resized.forEach(
        (detection, index) => {
          const box =
            detection.detection.box

          const score =
            detection.detection.score

          const age =
            Math.round(
              detection.age
            )

          const gender =
            detection.gender

          const genderProbability =
            Math.round(
              detection.genderProbability * 100
            )

          const faceConfidence =
            Math.round(score * 100)

          // FACE BOX

          const drawBox =
            new faceapi.draw.DrawBox(
              box,
              {
                label:
                  `Face ${index + 1} | ` +
                  `Age: ${age} | ` +
                  `Gender: ${gender} | ` +
                  `${faceConfidence}%`
              }
            )

          drawBox.draw(canvas)

          // LANDMARKS

          const drawLandmarks =
            new faceapi.draw.DrawFaceLandmarks(
              box,
              detection.landmarks
            )

          drawLandmarks.draw(canvas)

          // LABEL

          ctx.fillStyle =
            '#22c55e'

          ctx.font =
            'bold 14px Arial'

          ctx.fillText(
            `Face ${index + 1}`,
            box.x,
            Math.max(
              20,
              box.y - 10
            )
          )

          void genderProbability
        }
      )
    } catch (err) {
      console.error(
        'Detection error:',
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

    let stopped = false

    const runDetection = async () => {
      if (stopped) {
        return
      }

      await processVideo()

      if (!stopped) {
        animationRef.current =
          setTimeout(
            runDetection,
            300
          )
      }
    }

    runDetection()

    return () => {
      stopped = true

      if (animationRef.current) {
        clearTimeout(
          animationRef.current
        )

        animationRef.current = null
      }

      processingRef.current = false
    }
  }, [
    isCameraOn,
    cameraReady,
    isModelLoaded,
    processVideo
  ])

  // ============================================================
  // START / STOP CAMERA
  // ============================================================

  const toggleCamera = () => {
    setError(null)

    if (!isModelLoaded) {
      setError(
        'Please wait for the AI models to load.'
      )

      return
    }

    if (isCameraOn) {
      // STOP

      if (animationRef.current) {
        clearTimeout(
          animationRef.current
        )

        animationRef.current = null
      }

      processingRef.current = false

      setCameraReady(false)
      setIsCameraOn(false)

      setDetections([])

      return
    }

    // START

    setCapturedImage(null)
    setCapturedDetections([])

    setDetections([])

    setCameraReady(false)

    setIsCameraOn(true)
  }

  // ============================================================
  // CAPTURE
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
        'Camera is still starting. Please wait.'
      )

      return
    }

    const image =
      webcamRef.current.getScreenshot()

    if (!image) {
      setError(
        'Unable to capture image.'
      )

      return
    }

    // SAVE IMAGE

    setCapturedImage(image)

    // ========================================================
    // FREEZE CURRENT DETECTION
    // ========================================================

    const frozen =
      detections.map(
        (detection) => ({
          age:
            detection.age,

          gender:
            detection.gender,

          genderProbability:
            detection.genderProbability,

          detection: {
            score:
              detection.detection.score,

            box: {
              x:
                detection.detection.box.x,

              y:
                detection.detection.box.y,

              width:
                detection.detection.box.width,

              height:
                detection.detection.box.height
            }
          }
        })
      )

    setCapturedDetections(
      frozen
    )

    setError(null)

    console.log(
      'CAPTURED RESULT:',
      frozen
    )
  }

  // ============================================================
  // DOWNLOAD
  // ============================================================

  const downloadImage = () => {
    if (!capturedImage) {
      return
    }

    const link =
      document.createElement('a')

    link.download =
      'face-analysis.jpg'

    link.href =
      capturedImage

    document.body.appendChild(
      link
    )

    link.click()

    document.body.removeChild(
      link
    )
  }

  // ============================================================
  // RESET
  // ============================================================

  const resetApp = () => {
    if (animationRef.current) {
      clearTimeout(
        animationRef.current
      )

      animationRef.current = null
    }

    processingRef.current = false

    setIsCameraOn(false)

    setCameraReady(false)

    setDetections([])

    setCapturedDetections([])

    setCapturedImage(null)

    setError(null)

    setShowSettings(false)
  }

  // ============================================================
  // CAMERA SETTINGS
  // ============================================================

  const videoConstraints = {
    width: {
      ideal: 1280
    },

    height: {
      ideal: 720
    },

    facingMode: {
      ideal: 'user'
    }
  }

  // ============================================================
  // UI
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
                  disabled={!cameraReady}
                >

                  <Camera size={19} />

                  {cameraReady
                    ? 'Capture'
                    : 'Starting Camera...'}

                </button>

                <button
                  onClick={() =>
                    setShowSettings(
                      value => !value
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
            LOADING
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
                  isCameraOn
                    ? 'active'
                    : ''
                }`
              }
            >

              <span></span>

              {cameraReady
                ? 'Live'
                : isCameraOn
                  ? 'Starting...'
                  : 'Offline'}

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

                <div className="scan-frame">

                  <span className="corner top-left"></span>

                  <span className="corner top-right"></span>

                  <span className="corner bottom-left"></span>

                  <span className="corner bottom-right"></span>

                </div>

                {isProcessing && (

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
            LIVE RESULTS
        ==================================================== */}

        {detections.length > 0 && (

          <section className="detections-panel">

            <div className="section-heading">

              <div>

                <h3>
                  Live Detection Results
                </h3>

                <p>
                  {detections.length} face
                  {detections.length > 1
                    ? 's'
                    : ''} detected
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
                            Live
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
            CAPTURED RESULT
        ==================================================== */}

        {capturedImage && (

          <section className="captured-section">

            <div className="section-heading">

              <div>

                <h3>
                  Captured Image
                </h3>

                <p>
                  AI result frozen at capture time
                </p>

              </div>

              <div className="detection-status">

                <CheckCircle size={18} />

                Result Frozen

              </div>

            </div>

            <div className="captured-content">

              <img
                src={capturedImage}
                alt="Captured face"
                className="captured-image"
              />

              <div>

                {capturedDetections.length === 0 ? (

                  <div className="measurement-message">

                    <AlertCircle size={16} />

                    No face detected at capture time.

                  </div>

                ) : (

                  capturedDetections.map(
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
                                Captured / Frozen
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
                  )

                )}

                <button
                  onClick={downloadImage}
                  className="control-btn primary"
                >

                  <Download size={19} />

                  Download Image

                </button>

              </div>

            </div>

          </section>

        )}

        {/* ====================================================
            HEALTH
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

                --

                <small>
                  BPM
                </small>

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

                --

                <small>
                  %
                </small>

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
              Powered by etouchUs
            </p>

          </div>

        </div>

      </footer>

    </div>
  )
}

export default App