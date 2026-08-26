import { useState, useRef, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";

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
  Info,
} from "lucide-react";

import logo from "./assets/etouchus_face_recognition_logo.png";
import "./App.css";

function App() {
  // ============================================================
  // REFS
  // ============================================================

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const detectionRunning = useRef(false);
  const timerRef = useRef(null);

  // ============================================================
  // AI STATE
  // ============================================================

  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  // ============================================================
  // CAMERA STATE
  // ============================================================

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  // ============================================================
  // FACE DETECTION
  // ============================================================

  const [detections, setDetections] = useState([]);
  const [capturedDetections, setCapturedDetections] = useState([]);

  // ============================================================
  // CAPTURE
  // ============================================================

  const [capturedImage, setCapturedImage] = useState(null);

  // ============================================================
  // UI
  // ============================================================

  const [error, setError] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [confidence, setConfidence] = useState(0.5);

  // ============================================================
  // HEALTH UI
  // ============================================================

  const [heartRate] = useState(null);
  const [spo2] = useState(null);

  const [bloodPressure] = useState({
    systolic: null,
    diastolic: null,
  });

  // ============================================================
  // LOAD FACE API MODELS
  // ============================================================

  useEffect(() => {
    let mounted = true;

    async function loadModels() {
      try {
        setError("");

        console.log("Loading face-api.js models...");

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
          faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
          faceapi.nets.faceExpressionNet.loadFromUri("/models"),
          faceapi.nets.ageGenderNet.loadFromUri("/models"),
        ]);

        if (mounted) {
          setModelsLoaded(true);
        }

        console.log("All models loaded successfully.");
      } catch (err) {
        console.error("MODEL ERROR:", err);

        if (mounted) {
          setError(
            "AI models could not be loaded. Check that the model files are inside public/models."
          );
        }
      }
    }

    loadModels();

    return () => {
      mounted = false;
    };
  }, []);

  // ============================================================
  // CAMERA SUCCESS
  // ============================================================

  const handleCameraSuccess = useCallback(() => {
    console.log("CAMERA SUCCESS");

    setCameraReady(true);
    setError("");
  }, []);

  // ============================================================
  // CAMERA ERROR
  // ============================================================

  const handleCameraError = useCallback((err) => {
    console.error("CAMERA ERROR:", err);

    setCameraReady(false);
    setCameraOn(false);

    setError(
      "Camera could not start. Please allow camera permission in your browser and make sure another application is not using the webcam."
    );
  }, []);

  // ============================================================
  // FACE DETECTION
  // ============================================================

  const detectFaces = useCallback(async () => {
    if (!cameraOn) return;
    if (!cameraReady) return;
    if (!modelsLoaded) return;
    if (detectionRunning.current) return;

    const webcam = webcamRef.current;
    const canvas = canvasRef.current;

    if (!webcam || !canvas) return;

    const video = webcam.video;

    if (!video) return;

    if (video.readyState !== 4) return;

    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return;
    }

    detectionRunning.current = true;
    setIsProcessing(true);

    try {
      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight,
      };

      canvas.width = displaySize.width;
      canvas.height = displaySize.height;

      faceapi.matchDimensions(
        canvas,
        displaySize
      );

      const results = await faceapi
        .detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 224,
            scoreThreshold: confidence,
          })
        )
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender();

      setDetections(results);

      const resizedResults =
        faceapi.resizeResults(
          results,
          displaySize
        );

      const ctx = canvas.getContext("2d");

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      resizedResults.forEach(
        (result, index) => {
          const box = result.detection.box;

          const age = Math.round(result.age);

          const gender = result.gender;

          const faceScore = Math.round(
            result.detection.score * 100
          );

          const genderScore = Math.round(
            result.genderProbability * 100
          );

          const drawBox =
            new faceapi.draw.DrawBox(
              box,
              {
                label:
                  `Face ${index + 1} | ` +
                  `Age: ${age} | ` +
                  `Gender: ${gender} | ` +
                  `Confidence: ${faceScore}%`,
              }
            );

          drawBox.draw(canvas);

          const drawLandmarks =
            new faceapi.draw.DrawFaceLandmarks(
              box,
              result.landmarks
            );

          drawLandmarks.draw(canvas);

          ctx.fillStyle = "#22c55e";
          ctx.font = "bold 14px Arial";

          ctx.fillText(
            `Gender ${genderScore}%`,
            box.x,
            Math.max(20, box.y - 10)
          );
        }
      );
    } catch (err) {
      console.error(
        "FACE DETECTION ERROR:",
        err
      );
    } finally {
      detectionRunning.current = false;
      setIsProcessing(false);
    }
  }, [
    cameraOn,
    cameraReady,
    modelsLoaded,
    confidence,
  ]);

  // ============================================================
  // DETECTION LOOP
  // ============================================================

  useEffect(() => {
    if (
      !cameraOn ||
      !cameraReady ||
      !modelsLoaded
    ) {
      return;
    }

    let stopped = false;

    const runLoop = async () => {
      if (stopped) return;

      await detectFaces();

      if (!stopped) {
        timerRef.current = setTimeout(
          runLoop,
          400
        );
      }
    };

    runLoop();

    return () => {
      stopped = true;

      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }

      detectionRunning.current = false;
    };
  }, [
    cameraOn,
    cameraReady,
    modelsLoaded,
    detectFaces,
  ]);

  // ============================================================
  // START CAMERA
  // ============================================================

  const startCamera = () => {
    setError("");

    if (!modelsLoaded) {
      setError(
        "Please wait until the AI models finish loading."
      );

      return;
    }

    setCameraReady(false);
    setCameraOn(true);
  };

  // ============================================================
  // STOP CAMERA
  // ============================================================

  const stopCamera = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    detectionRunning.current = false;

    setIsProcessing(false);
    setCameraReady(false);
    setCameraOn(false);
    setDetections([]);
  };

  // ============================================================
  // CAPTURE IMAGE
  // ============================================================

  const captureImage = () => {
    if (!cameraOn) {
      setError(
        "Please start the camera first."
      );

      return;
    }

    if (!cameraReady) {
      setError(
        "Camera is still starting. Please wait a moment."
      );

      return;
    }

    if (!webcamRef.current) {
      setError(
        "Webcam is not available."
      );

      return;
    }

    const image =
      webcamRef.current.getScreenshot();

    if (!image) {
      setError(
        "Could not capture the camera image."
      );

      return;
    }

    setCapturedImage(image);

    // Save current AI results.
    setCapturedDetections(
      detections.map((detection) => ({
        age: detection.age,
        gender: detection.gender,
        genderProbability:
          detection.genderProbability,
        confidence:
          detection.detection.score,
      }))
    );

    setError("");

    console.log("IMAGE CAPTURED");
  };

  // ============================================================
  // DOWNLOAD IMAGE
  // ============================================================

  const downloadImage = () => {
    if (!capturedImage) return;

    const link =
      document.createElement("a");

    link.href = capturedImage;
    link.download = "face-analysis.jpg";

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);
  };

  // ============================================================
  // RESET
  // ============================================================

  const resetApp = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    detectionRunning.current = false;

    setCameraReady(false);
    setCameraOn(false);

    setDetections([]);
    setCapturedDetections([]);

    setCapturedImage(null);

    setError("");

    setShowSettings(false);
  };

  // ============================================================
  // VIDEO SETTINGS
  // ============================================================

  const videoConstraints = {
    width: {
      ideal: 1280,
    },

    height: {
      ideal: 720,
    },

    facingMode: "user",
  };

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
                  ? "ready"
                  : "loading"
              }`
            }
          >

            {modelsLoaded ? (
              <CheckCircle size={17} />
            ) : (
              <Brain size={17} />
            )}

            <span>
              {modelsLoaded
                ? "AI Model Ready"
                : "Loading AI Model..."}
            </span>

          </div>

        </div>

      </header>


      {/* ======================================================
          MAIN
      ====================================================== */}

      <main className="main">

        {/* TITLE */}

        <section className="page-title">

          <div>

            <h2>
              Facial Recognition & Health
            </h2>

            <p>
              Analyze facial attributes and connect
              validated health measurement systems.
            </p>

          </div>

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
                  (value) => !value
                )
              }
            >

              <Settings size={19} />

              Settings

            </button>


            <button
              className="control-btn secondary"
              onClick={resetApp}
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
                    ? "active"
                    : ""
                }`
              }
            >

              <span></span>

              {cameraReady
                ? "Live"
                : cameraOn
                ? "Starting..."
                : "Offline"}

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
            LIVE DETECTION
        ==================================================== */}

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
                    ? "s"
                    : ""} detected
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
                    );

                  const gender =
                    detection.gender;

                  const genderProbability =
                    Math.round(
                      detection.genderProbability *
                      100
                    );

                  const score =
                    Math.round(
                      detection.detection.score *
                      100
                    );

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

                  );
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
                  This image was captured manually.
                </p>

              </div>

              <div className="detection-status">

                <CheckCircle size={17} />

                Captured

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

                  <p>
                    No face was detected at the
                    moment of capture.
                  </p>

                ) : (

                  capturedDetections.map(
                    (detection, index) => (

                      <div
                        className="captured-result-card"
                        key={index}
                      >

                        <strong>
                          Face {index + 1}
                        </strong>

                        <span>
                          Age:{" "}
                          {Math.round(
                            detection.age
                          )} years
                        </span>

                        <span>
                          Gender:{" "}
                          {detection.gender}
                        </span>

                        <span>
                          Face confidence:{" "}
                          {Math.round(
                            detection.confidence *
                            100
                          )}%
                        </span>

                      </div>

                    )
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
            HEALTH DASHBOARD
        ==================================================== */}

        <section className="health-section">

          <div className="section-heading">

            <div>

              <h3>
                Health Monitoring
              </h3>

              <p>
                Vital-sign measurements require
                validated algorithms or physical sensors.
              </p>

            </div>

            <div className="health-disclaimer">

              <ShieldCheck size={16} />

              Measurement status

            </div>

          </div>


          <div className="health-grid">

            {/* HEART RATE */}

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

                {heartRate ?? "--"}

                <small>
                  BPM
                </small>

              </div>

              <p className="health-description">
                Requires validated rPPG or a
                heart-rate sensor.
              </p>

              <div className="measurement-message">

                <Activity size={15} />

                Ready for rPPG integration

              </div>

            </div>


            {/* SPO2 */}

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

                {spo2 ?? "--"}

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


            {/* BLOOD PRESSURE */}

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
                    {bloodPressure.systolic ?? "--"}
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
                    {bloodPressure.diastolic ?? "--"}
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
              values are not generated by face-api.js.
              These features require validated algorithms
              or physical sensors. Do not use placeholder
              values for diagnosis or medical decisions.
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
  );
}

export default App;