import { useState, useRef, useEffect } from 'react'
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
  const webcamRef = useRef(null)
  const canvasRef = useRef(null)
  const detectionTimer = useRef(null)
  const processing = useRef(false)

  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [cameraOn, setCameraOn] = useState(false)
  const [cameraReady, setCameraReady] = useState(false)

  const [detections, setDetections] = useState([])
  const [capturedImage, setCapturedImage] = useState(null)
  const [capturedDetections, setCapturedDetections] = useState([])

  const [error, setError] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [confidence, setConfidence] = useState(0.5)

  // ------------------------------------------------------------
  // LOAD MODELS
  // ------------------------------------------------------------

  useEffect(() => {
    async function loadModels() {
      try {
        console.log('Loading models...')

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
          faceapi.nets.faceExpressionNet.loadFromUri('/models'),
          faceapi.nets.ageGenderNet.loadFromUri('/models')
        ])

        console.log('Models loaded')

        setModelsLoaded(true)
      } catch (err) {
        console.error(err)

        setError(
          'AI models could not be loaded. Check public/models.'
        )
      }
    }

    loadModels()
  }, [])

  // ------------------------------------------------------------
  // CAMERA SUCCESS
  // ------------------------------------------------------------

  const cameraStarted = () => {
    console.log('WEBCAM STARTED')

    setCameraReady(true)
    setError('')
  }

  // ------------------------------------------------------------
  // CAMERA ERROR
  // ------------------------------------------------------------

  const cameraError = (err) => {
    console.error('WEBCAM ERROR:', err)

    setCameraReady(false)

    setCameraOn(false)

    setError(
      'Camera could not start. Please allow camera permission in Chrome.'
    )
  }

  // ------------------------------------------------------------
  // FACE DETECTION
  // ------------------------------------------------------------

  const detectFace = async () => {
    if (!cameraOn) return
    if (!cameraReady) return
    if (!modelsLoaded) return
    if (processing.current) return

    if (!webcamRef.current) return
    if (!canvasRef.current) return

    const video = webcamRef.current.video

    if (!video) return

    if (video.readyState !== 4) return

    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return
    }

    processing.current = true

    try {
      const canvas = canvasRef.current

      const size = {
        width: video.videoWidth,
        height: video.videoHeight
      }

      canvas.width = size.width
      canvas.height = size.height

      faceapi.matchDimensions(
        canvas,
        size
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

      setDetections(results)

      const resized =
        faceapi.resizeResults(
          results,
          size
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
        (result, index) => {
          const box =
            result.detection.box

          const age =
            Math.round(result.age)

          const gender =
            result.gender

          const score =
            Math.round(
              result.detection.score * 100
            )

          const drawBox =
            new faceapi.draw.DrawBox(
              box,
              {
                label:
                  `Face ${index + 1} | ` +
                  `Age ${age} | ` +
                  `${gender} | ` +
                  `${score}%`
              }
            )

          drawBox.draw(canvas)

          const landmarks =
            new faceapi.draw.DrawFaceLandmarks(
              box,
              result.landmarks
            )

          landmarks.draw(canvas)
        }
      )
    } catch (err) {
      console.error(
        'Face detection error:',
        err
      )
    }

    processing.current = false
  }

  // ------------------------------------------------------------
  // DETECTION LOOP
  // ------------------------------------------------------------

  useEffect(() => {
    if (
      !cameraOn ||
      !cameraReady ||
      !modelsLoaded
    ) {
      return
    }

    const loop = async () => {
      await detectFace()

      detectionTimer.current =
        setTimeout(
          loop,
          400
        )
    }

    loop()

    return () => {
      if (detectionTimer.current) {
        clearTimeout(
          detectionTimer.current
        )

        detectionTimer.current = null
      }

      processing.current = false
    }
  }, [
    cameraOn,
    cameraReady,
    modelsLoaded,
    confidence
  ])

  // ------------------------------------------------------------
  // START CAMERA
  // ------------------------------------------------------------

  const startCamera = () => {
    setError('')

    if (!modelsLoaded) {
      setError(
        'Please wait for the AI models to finish loading.'
      )

      return
    }

    setCameraOn(true)
  }

  // ------------------------------------------------------------
  // STOP CAMERA
  // ------------------------------------------------------------

  const stopCamera = () => {
    if (detectionTimer.current) {
      clearTimeout(
        detectionTimer.current
      )
    }

    processing.current = false

    setCameraReady(false)
    setCameraOn(false)
    setDetections([])
  }

  // ------------------------------------------------------------
  // CAPTURE
  // ------------------------------------------------------------

  const captureImage = () => {
    if (!cameraReady) {
      setError(
        'Camera is not ready yet.'
      )

      return
    }

    const image =
      webcamRef.current?.getScreenshot()

    if (!image) {
      setError(
        'Could not capture image.'
      )

      return
    }

    setCapturedImage(image)

    // Freeze current AI results
    setCapturedDetections(
      detections.map(
        detection => ({
          age: detection.age,
          gender: detection.gender,
          genderProbability:
            detection.genderProbability,
          score:
            detection.detection.score
        })
      )
    )

    setError('')

    console.log(
      'IMAGE CAPTURED'
    )
  }

  // ------------------------------------------------------------
  // RESET
  // ------------------------------------------------------------

  const reset = () => {
    stopCamera()

    setCapturedImage(null)

    setCapturedDetections([])

    setDetections([])

    setShowSettings(false)

    setError('')
  }

  // ------------------------------------------------------------
  // DOWNLOAD
  // ------------------------------------------------------------

  const downloadImage = () => {
    if (!capturedImage) return

    const link =
      document.createElement('a')

    link.href = capturedImage

    link.download =
      'face-analysis.jpg'

    document.body.appendChild(link)

    link.click()

    document.body.removeChild(link)
  }

  // ------------------------------------------------------------
  // CAMERA SETTINGS
  // ------------------------------------------------------------

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
                modelsLoaded
                  ? 'ready'
                  : 'loading'
              }`
            }
          >

            {modelsLoaded ? (
              <CheckCircle size={17} />
            ) : (
              <Brain size={17} />
            )}

            {modelsLoaded
              ? 'AI Model Ready'
              : 'Loading AI Model...'}

          </div>

        </div>

      </header>


      {/* MAIN */}

      <main className="main">

        <section className="page-title">

          <h2>
            Facial Recognition & Health
          </h2>

          <p>
            Analyze facial attributes and connect
            validated health measurement systems.
          </p>

        </section>


        {/* ====================================================
            CONTROLS
        ==================================================== */}

        <section className="control-section">

          <div className="controls">

            {!cameraOn ? (

              <button
                className="control-btn primary"
                onClick={startCamera}
                disabled={!modelsLoaded}
              >

                <Camera size={19} />

                Start Camera

              </button>

            ) : (

              <button
                className="control-btn danger"
                onClick={stopCamera}
              >

                <CameraOff size={19} />

                Stop Camera

              </button>

            )}


            <button
              className="control-btn secondary"
              onClick={captureImage}
              disabled={!cameraReady}
            >

              <Camera size={19} />

              Capture

            </button>


            <button
              className="control-btn secondary"
              onClick={() =>
                setShowSettings(
                  value => !value
                )
              }
            >

              <Settings size={19} />

              Settings

            </button>


            <button
              className="control-btn secondary"
              onClick={reset}
            >

              <RotateCcw size={19} />

              Reset

            </button>

          </div>


          {/* SETTINGS */}

          {showSettings && (

            <div className="settings-panel">

              <div className="settings-title">

                <Settings size={18} />

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
                      Number(e.target.value)
                    )
                  }
                />

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
                Camera / AI Error
              </strong>

              <p>
                {error}
              </p>

            </div>

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
                  cameraReady
                    ? 'active'
                    : ''
                }`
              }
            >

              <span></span>

              {cameraReady
                ? 'Live'
                : cameraOn
                  ? 'Starting...'
                  : 'Offline'}

            </div>

          </div>


          <div className="camera-container">

            {cameraOn ? (

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
                    cameraStarted
                  }

                  onUserMediaError={
                    cameraError
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
                  Click Start Camera to begin.
                </p>

              </div>

            )}

          </div>

        </section>


        {/* ====================================================
            RESULTS
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

                <Activity size={17} />

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

                  const score =
                    Math.round(
                      detection.detection.score *
                      100
                    )

                  return (

                    <div
                      className="detection-card"
                      key={index}
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
                            {score}%
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
                  Image captured manually
                </p>

              </div>

              <div className="detection-status">

                <CheckCircle size={17} />

                Captured

              </div>

            </div>


            <div className="captured-content">

              <img
                src={capturedImage}
                alt="Captured face"
                className="captured-image"
              />

              <div>

                {capturedDetections.map(
                  (detection, index) => (

                    <div
                      className="detection-card"
                      key={index}
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
                            Captured
                          </span>

                        </div>

                      </div>

                      <div className="detection-info">

                        <div className="info-item">

                          <span className="label">
                            Estimated Age
                          </span>

                          <span className="value">
                            {Math.round(
                              detection.age
                            )} years
                          </span>

                        </div>

                        <div className="info-item">

                          <span className="label">
                            Gender
                          </span>

                          <span className="value">
                            {detection.gender}
                          </span>

                        </div>

                      </div>

                    </div>

                  )
                )}

                <button
                  className="control-btn primary"
                  onClick={
                    downloadImage
                  }
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
                Validated algorithms or physical sensors
                are required for vital-sign measurements.
              </p>

            </div>

            <div className="health-disclaimer">

              <ShieldCheck size={16} />

              Measurement status

            </div>

          </div>


          <div className="health-grid">

            <div className="health-card">

              <div className="health-card-top">

                <div className="health-icon">

                  <HeartPulse size={25} />

                </div>

                <span className="health-status">
                  Not measured
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
                Requires validated rPPG
                or a heart-rate sensor.
              </p>

              {cameraOn && (

                <div className="measurement-message">

                  <Activity size={15} />

                  Camera available for rPPG integration

                </div>

              )}

            </div>


            <div className="health-card">

              <div className="health-card-top">

                <div className="health-icon">

                  <Droplets size={25} />

                </div>

                <span className="health-status">
                  Waiting for sensor
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
                Requires a validated SpO₂
                sensor or approved model.
              </p>

              <div className="measurement-message">

                <Info size={15} />

                Sensor required

              </div>

            </div>


            <div className="health-card">

              <div className="health-card-top">

                <div className="health-icon">

                  <Stethoscope size={25} />

                </div>

                <span className="health-status">
                  Waiting for sensor
                </span>

              </div>

              <div className="health-card-title">
                Blood Pressure
              </div>

              <div className="bp-values">

                <div>

                  <strong>
                    --
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
                    --
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


        {/* NOTICE */}

        <div className="medical-notice">

          <ShieldCheck size={20} />

          <div>

            <strong>
              Measurement Notice
            </strong>

            <p>
              Heart rate, SpO₂ and blood-pressure
              values are not generated by face-api.js.
              Do not use placeholder values for diagnosis
              or medical decisions.
            </p>

          </div>

        </div>

      </main>


      {/* FOOTER */}

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
              Powered by etouchUS
            </p>

          </div>

        </div>

      </footer>

    </div>
  )
}

export default App