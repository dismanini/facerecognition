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

  // Live detection results
  const [detections, setDetections] = useState([])

  // Captured/frozen detection results
  const [capturedDetections, setCapturedDetections] = useState([])

  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)
  const [capturedImage, setCapturedImage] = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  const [confidence, setConfidence] = useState(0.5)

  // ============================================================
  // HEALTH MEASUREMENTS
  // ============================================================

  const [heartRate, setHeartRate] = useState(null)

  const [spo2, setSpo2] = useState(null)

  const [bloodPressure, setBloodPressure] = useState({
    systolic: null,
    diastolic: null
  })

  const [heartRateStatus, setHeartRateStatus] =
    useState('Not measured')

  const [spo2Status, setSpo2Status] =
    useState('Waiting for sensor')

  const [bloodPressureStatus, setBloodPressureStatus] =
    useState('Waiting for sensor')

  // ============================================================
  // REFS
  // ============================================================

  const webcamRef = useRef(null)
  const canvasRef = useRef(null)

  // Prevent overlapping face detection
  const processingRef = useRef(false)

  // Detection loop timer
  const animationRef = useRef(null)

  // ============================================================
  // LOAD FACE API MODELS
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

        console.log('All AI models loaded successfully')
      } catch (err) {
        console.error('Error loading models:', err)

        setError(
          'Failed to load AI models. Make sure the model files are inside public/models.'
        )
      }
    }

    loadModels()
  }, [])

  // ============================================================
  // PROCESS VIDEO
  // ============================================================

  const processVideo = useCallback(async () => {
    if (
      !webcamRef.current ||
      !canvasRef.current ||
      !isModelLoaded ||
      !isCameraOn
    ) {
      return
    }

    // Prevent multiple detections running simultaneously
    if (processingRef.current) {
      return
    }

    const video = webcamRef.current?.video
    const canvas = canvasRef.current

    if (!video || !canvas) {
      return
    }

    if (video.readyState !== 4) {
      return
    }

    const displaySize = {
      width: video.videoWidth,
      height: video.videoHeight
    }

    if (
      displaySize.width === 0 ||
      displaySize.height === 0
    ) {
      return
    }

    try {
      processingRef.current = true

      setIsProcessing(true)

      faceapi.matchDimensions(
        canvas,
        displaySize
      )

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

      // ========================================================
      // LIVE DETECTION
      // ========================================================

      setDetections(results)

      // ========================================================
      // DRAW DETECTIONS
      // ========================================================

      const resizedDetections =
        faceapi.resizeResults(
          results,
          displaySize
        )

      const ctx = canvas.getContext('2d')

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      )

      resizedDetections.forEach(
        (detection, index) => {
          const box =
            detection.detection.box

          const detectionScore =
            detection.detection.score

          const age =
            Math.round(detection.age)

          const gender =
            detection.gender

          const genderProbability =
            Math.round(
              detection.genderProbability * 100
            )

          const faceConfidence =
            Math.round(
              detectionScore * 100
            )

          // ====================================================
          // FACE BOX
          // ====================================================

          const drawBox =
            new faceapi.draw.DrawBox(
              box,
              {
                label:
                  `Face ${index + 1} | ` +
                  `Age: ${age} | ` +
                  `Gender: ${gender} | ` +
                  `Confidence: ${faceConfidence}%`
              }
            )

          drawBox.draw(canvas)

          // ====================================================
          // LANDMARKS
          // ====================================================

          const landmarks =
            detection.landmarks

          const drawLandmarks =
            new faceapi.draw.DrawFaceLandmarks(
              box,
              landmarks
            )

          drawLandmarks.draw(canvas)

          // ====================================================
          // FACE LABEL
          // ====================================================

          ctx.fillStyle = '#22c55e'

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

          // Prevent unused variable warning
          void genderProbability
        }
      )
    } catch (err) {
      console.error(
        'Error processing video:',
        err
      )
    } finally {
      processingRef.current = false

      setIsProcessing(false)
    }
  }, [
    isModelLoaded,
    isCameraOn,
    confidence
  ])

  // ============================================================
  // CONTROLLED DETECTION LOOP
  // ============================================================

  useEffect(() => {
    if (
      !isCameraOn ||
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
    isModelLoaded,
    processVideo
  ])

  // ============================================================
  // TOGGLE CAMERA
  // ============================================================

  const toggleCamera = () => {
    setError(null)

    if (!isModelLoaded) {
      setError(
        'AI models are still loading. Please wait.'
      )

      return
    }

    // ========================================================
    // STOP CAMERA
    // ========================================================

    if (isCameraOn) {
      if (animationRef.current) {
        clearTimeout(
          animationRef.current
        )

        animationRef.current = null
      }

      processingRef.current = false

      setIsCameraOn(false)

      setDetections([])

      return
    }

    // ========================================================
    // START CAMERA
    // ========================================================

    setCapturedImage(null)

    setCapturedDetections([])

    setDetections([])

    setIsCameraOn(true)
  }

  // ============================================================
  // CAPTURE IMAGE + FREEZE CURRENT RESULT
  // ============================================================

  const captureImage = () => {
    if (!webcamRef.current) {
      setError(
        'Camera is not available.'
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

    // ========================================================
    // SAVE IMAGE
    // ========================================================

    setCapturedImage(imageSrc)

    // ========================================================
    // FREEZE CURRENT DETECTION
    //
    // Important:
    // We create a copy of the current detection array.
    //
    // Future live detection changes will NOT change this
    // captured result.
    // ========================================================

    const frozenDetections =
      detections.map(
        (detection) => ({
          age: detection.age,

          gender: detection.gender,

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
      frozenDetections
    )

    setError(null)

    console.log(
      'Captured image and froze detection results:',
      frozenDetections
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
    if (animationRef.current) {
      clearTimeout(
        animationRef.current
      )

      animationRef.current = null
    }

    processingRef.current = false

    setIsCameraOn(false)

    setDetections([])

    setCapturedDetections([])

    setCapturedImage(null)

    setError(null)

    setShowSettings(false)

    // Reset health measurements

    setHeartRate(null)

    setSpo2(null)

    setBloodPressure({
      systolic: null,
      diastolic: null
    })

    setHeartRateStatus(
      'Not measured'
    )

    setSpo2Status(
      'Waiting for sensor'
    )

    setBloodPressureStatus(
      'Waiting for sensor'
    )
  }

  // ============================================================
  // CAMERA ERROR
  // ============================================================

  const handleCameraError = (cameraError) => {
    console.error(
      'Camera error:',
      cameraError
    )

    setError(
      'Camera could not be accessed. Please allow camera permission in your browser.'
    )

    setIsCameraOn(false)
  }

  // ============================================================
  // CAMERA SUCCESS
  // ============================================================

  const handleCameraSuccess = () => {
    console.log(
      'Camera started successfully'
    )

    setError(null)
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
                      current => !current
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

        {/* ERROR */}

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

        {/* LOADING */}

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

              {isCameraOn
                ? 'Live'
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
                    handleCameraSuccess
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
            LIVE DETECTION RESULTS
        ==================================================== */}

        {detections.length > 0 && (

          <section className="detections-panel">

            <div className="section-heading">

              <div>

                <h3>
                  Live Detection Results
                </h3>

                <p>
                  {detections.length}
                  {' '}
                  face
                  {detections.length > 1
                    ? 's'
                    : ''}
                  {' '}
                  detected
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

                          <UserRound
                            size={22}
                          />

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
                  Face analysis frozen at capture time
                </p>

              </div>

              <div className="detection-status">

                <CheckCircle size={18} />

                Result Frozen

              </div>

            </div>

            <div className="captured-content">

              <div className="captured-image-wrapper">

                <img
                  src={capturedImage}
                  alt="Captured face"
                  className="captured-image"
                />

              </div>

              <div className="captured-results">

                <h4>
                  Captured Analysis
                </h4>

                {capturedDetections.length === 0 ? (

                  <div className="no-captured-face">

                    <AlertCircle size={18} />

                    <span>
                      No face was detected at the moment
                      of capture.
                    </span>

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
                          className="captured-result-card"
                        >

                          <div className="captured-face-title">

                            <div className="face-icon">

                              <UserRound
                                size={20}
                              />

                            </div>

                            <strong>
                              Face {index + 1}
                            </strong>

                          </div>

                          <div className="captured-result-grid">

                            <div>

                              <span>
                                Estimated Age
                              </span>

                              <strong>
                                {age} years
                              </strong>

                            </div>

                            <div>

                              <span>
                                Gender
                              </span>

                              <strong>
                                {gender}
                              </strong>

                            </div>

                            <div>

                              <span>
                                Gender Confidence
                              </span>

                              <strong>
                                {genderProbability}%
                              </strong>

                            </div>

                            <div>

                              <span>
                                Face Confidence
                              </span>

                              <strong>
                                {faceConfidence}%
                              </strong>

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

            {/* =================================================
                HEART RATE
            ================================================= */}

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

                    <small>
                      BPM
                    </small>

                  </>

                ) : (

                  <>

                    --

                    <small>
                      BPM
                    </small>

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

            {/* =================================================
                SPO2
            ================================================= */}

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

                    <small>
                      %
                    </small>

                  </>

                ) : (

                  <>

                    --

                    <small>
                      %
                    </small>

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

            {/* =================================================
                BLOOD PRESSURE
            ================================================= */}

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