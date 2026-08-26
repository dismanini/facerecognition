import { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";
import {
  Camera,
  CameraOff,
  RotateCcw,
  Download,
  Settings,
  ScanFace,
  User,
  Activity,
} from "lucide-react";

import logo from "./assets/etouchus_face_recognition_logo.png";
import "./App.css";

function App() {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [detections, setDetections] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [confidence, setConfidence] = useState(0.5);

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // ============================================================
  // LOAD FACE-API MODELS
  // ============================================================

  useEffect(() => {
    const loadModels = async () => {
      try {
        setError(null);

        const MODEL_URL = "/models";

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
          faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
        ]);

        setIsModelLoaded(true);

        console.log("Face API models loaded successfully");
      } catch (err) {
        console.error("Model loading error:", err);

        setError(
          "Failed to load AI models. Make sure your model files are inside public/models."
        );
      }
    };

    loadModels();
  }, []);

  // ============================================================
  // PROCESS VIDEO
  // ============================================================

  const processVideo = async () => {
    if (
      !webcamRef.current ||
      !canvasRef.current ||
      !isModelLoaded ||
      !isCameraOn
    ) {
      return;
    }

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;

    if (!video || video.readyState !== 4) {
      return;
    }

    const displaySize = {
      width: video.videoWidth,
      height: video.videoHeight,
    };

    if (!displaySize.width || !displaySize.height) {
      return;
    }

    try {
      setIsProcessing(true);

      faceapi.matchDimensions(canvas, displaySize);

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

      const resizedDetections = faceapi.resizeResults(
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

      // Draw detection boxes
      resizedDetections.forEach((detection, index) => {
        const box = detection.detection.box;

        // Bounding box
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: `Face ${index + 1} | Age: ${Math.round(
            detection.age
          )} | ${detection.gender}`,
        });

        drawBox.draw(canvas);

        // Draw landmarks
        const landmarks =
          detection.landmarks;

        const drawLandmarks =
          new faceapi.draw.DrawFaceLandmarks(
            box,
            {
              lineWidth: 1,
              drawLines: true,
            }
          );

        if (landmarks) {
          faceapi.draw.drawFaceLandmarks(
            canvas,
            [detection]
          );
        }

        // Draw expression
        if (detection.expressions) {
          const expressions =
            detection.expressions;

          const sortedExpressions =
            Object.entries(expressions).sort(
              (a, b) => b[1] - a[1]
            );

          const topExpression =
            sortedExpressions[0];

          if (topExpression) {
            const expressionName =
              topExpression[0];

            const expressionConfidence =
              Math.round(
                topExpression[1] * 100
              );

            ctx.font =
              "bold 14px Arial";

            ctx.fillStyle =
              "#00ff88";

            ctx.fillText(
              `${expressionName} ${expressionConfidence}%`,
              box.x,
              box.y + box.height + 20
            );
          }
        }
      });
    } catch (err) {
      console.error(
        "Video processing error:",
        err
      );
    } finally {
      setIsProcessing(false);
    }
  };

  // ============================================================
  // START VIDEO PROCESSING
  // ============================================================

  useEffect(() => {
    if (
      !isCameraOn ||
      !isModelLoaded
    ) {
      return;
    }

    const interval = setInterval(
      processVideo,
      150
    );

    return () => {
      clearInterval(interval);
    };
  }, [
    isCameraOn,
    isModelLoaded,
    confidence,
  ]);

  // ============================================================
  // TOGGLE CAMERA
  // ============================================================

  const toggleCamera = () => {
    setError(null);
    setDetections([]);
    setCapturedImage(null);

    setIsCameraOn(
      (current) => !current
    );
  };

  // ============================================================
  // CAPTURE IMAGE
  // ============================================================

  const captureImage = () => {
    if (!webcamRef.current) {
      setError(
        "Camera is not available."
      );
      return;
    }

    if (detections.length === 0) {
      setError(
        "No face detected. Please position your face in front of the camera."
      );
      return;
    }

    const imageSrc =
      webcamRef.current.getScreenshot();

    if (!imageSrc) {
      setError(
        "Unable to capture image."
      );
      return;
    }

    setCapturedImage(imageSrc);
    setError(null);
  };

  // ============================================================
  // DOWNLOAD IMAGE
  // ============================================================

  const downloadImage = () => {
    if (!capturedImage) {
      return;
    }

    const link =
      document.createElement("a");

    link.download =
      "face-detection.jpg";

    link.href = capturedImage;

    link.click();
  };

  // ============================================================
  // RESET
  // ============================================================

  const resetApp = () => {
    setIsCameraOn(false);
    setDetections([]);
    setCapturedImage(null);
    setError(null);
    setShowSettings(false);

    if (canvasRef.current) {
      const ctx =
        canvasRef.current.getContext(
          "2d"
        );

      ctx.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );
    }
  };

  // ============================================================
  // CAMERA CONSTRAINTS
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

            <img
              src={logo}
              alt="eTouchUS Face Recognition"
              className="logo"
            />

            <div className="brand-text">

              <h1>
                AI Face Detector
              </h1>

              <p>
                Real-time face detection,
                age estimation and gender
                recognition
              </p>

            </div>

          </div>

          <div className="system-status">

            <span
              className={
                isModelLoaded
                  ? "status-dot online"
                  : "status-dot"
              }
            />

            {isModelLoaded
              ? "AI Model Ready"
              : "Loading AI Model..."}

          </div>

        </div>

      </header>

      {/* ======================================================
          MAIN
      ====================================================== */}

      <main className="main">

        {/* ====================================================
            TOP CONTROL BAR
        ==================================================== */}

        <div className="top-bar">

          <div className="page-title">

            <div className="title-icon">
              <ScanFace
                size={25}
              />
            </div>

            <div>
              <h2>
                Face Recognition
              </h2>

              <p>
                AI-powered facial analysis
              </p>
            </div>

          </div>

          <div className="controls">

            <button
              onClick={toggleCamera}
              className={`control-btn ${
                isCameraOn
                  ? "danger"
                  : "primary"
              }`}
              disabled={!isModelLoaded}
            >

              {isCameraOn ? (
                <CameraOff size={19} />
              ) : (
                <Camera size={19} />
              )}

              {isCameraOn
                ? "Stop Camera"
                : "Start Camera"}

            </button>

            {isCameraOn && (
              <>

                <button
                  onClick={captureImage}
                  className="control-btn secondary"
                  disabled={
                    detections.length === 0
                  }
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
                  <RotateCcw
                    size={19}
                  />
                  Reset
                </button>

              </>
            )}

          </div>

        </div>

        {/* ====================================================
            SETTINGS
        ==================================================== */}

        {showSettings && (
          <div className="settings-panel">

            <div>
              <h3>
                Detection Settings
              </h3>

              <p>
                Adjust face detection
                confidence threshold.
              </p>
            </div>

            <div className="setting-item">

              <label>
                Confidence Threshold:
                <strong>
                  {Math.round(
                    confidence * 100
                  )}
                  %
                </strong>
              </label>

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

            </div>

          </div>
        )}

        {/* ====================================================
            ERROR
        ==================================================== */}

        {error && (
          <div className="error-message">

            <strong>
              ⚠️ Error
            </strong>

            <p>
              {error}
            </p>

          </div>
        )}

        {/* ====================================================
            LOADING
        ==================================================== */}

        {!isModelLoaded && !error && (
          <div className="loading">

            <div className="loading-spinner" />

            <p>
              Loading AI models...
            </p>

            <small>
              Please wait
            </small>

          </div>
        )}

        {/* ====================================================
            CONTENT GRID
        ==================================================== */}

        <div className="content-grid">

          {/* ==================================================
              CAMERA CARD
          ================================================== */}

          <section className="camera-card">

            <div className="card-header">

              <div>

                <h2>
                  Live Camera
                </h2>

                <p>
                  Position your face
                  inside the camera
                </p>

              </div>

              <div
                className={`face-status ${
                  detections.length > 0
                    ? "detected"
                    : ""
                }`}
              >

                <User size={17} />

                {detections.length > 0
                  ? `${detections.length} Face${
                      detections.length > 1
                        ? "s"
                        : ""
                    }`
                  : "No Face"}

              </div>

            </div>

            {/* CAMERA */}

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
                  />

                  <canvas
                    ref={canvasRef}
                    className="detection-canvas"
                  />

                  {/* Scanner corners */}

                  <div className="scan-frame">

                    <span className="corner top-left" />
                    <span className="corner top-right" />
                    <span className="corner bottom-left" />
                    <span className="corner bottom-right" />

                  </div>

                  {/* Camera status */}

                  <div className="camera-status">

                    {detections.length > 0 ? (
                      <>
                        <span className="live-dot" />
                        Face detected
                      </>
                    ) : (
                      <>
                        <span className="search-dot" />
                        Searching for face...
                      </>
                    )}

                  </div>

                  {isProcessing && (
                    <div className="processing-overlay">

                      <div className="processing-spinner" />

                      <span>
                        Processing...
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
                    Camera is off
                  </h3>

                  <p>
                    Click "Start Camera"
                    to begin face detection.
                  </p>

                  <button
                    onClick={toggleCamera}
                    disabled={!isModelLoaded}
                    className="placeholder-button"
                  >
                    <Camera size={18} />
                    Start Camera
                  </button>

                </div>

              )}

            </div>

          </section>

          {/* ==================================================
              RESULTS CARD
          ================================================== */}

          <section className="results-card">

            <div className="card-header">

              <div>

                <h2>
                  Detection Results
                </h2>

                <p>
                  AI facial analysis
                </p>

              </div>

              <Activity
                size={24}
              />

            </div>

            {detections.length === 0 ? (

              <div className="empty-results">

                <div className="empty-icon">
                  <ScanFace
                    size={35}
                  />
                </div>

                <h3>
                  No face detected
                </h3>

                <p>
                  Start the camera and
                  position a face in view
                  to see AI analysis.
                </p>

              </div>

            ) : (

              <div className="results-list">

                {detections.map(
                  (detection, index) => {

                    const expressions =
                      detection.expressions;

                    const sortedExpressions =
                      expressions
                        ? Object.entries(
                            expressions
                          ).sort(
                            (a, b) =>
                              b[1] - a[1]
                          )
                        : [];

                    const expression =
                      sortedExpressions[0];

                    return (
                      <div
                        className="detection-card"
                        key={index}
                      >

                        <div className="detection-card-header">

                          <div className="face-number">

                            <User
                              size={19}
                            />

                            Face {index + 1}

                          </div>

                          <span className="detected-badge">
                            Detected
                          </span>

                        </div>

                        <div className="result-grid">

                          <div className="result-item">

                            <span>
                              Age
                            </span>

                            <strong>
                              {Math.round(
                                detection.age
                              )}{" "}
                              years
                            </strong>

                          </div>

                          <div className="result-item">

                            <span>
                              Gender
                            </span>

                            <strong>
                              {detection.gender}
                            </strong>

                          </div>

                          <div className="result-item">

                            <span>
                              Confidence
                            </span>

                            <strong>
                              {Math.round(
                                detection.genderProbability *
                                  100
                              )}
                              %
                            </strong>

                          </div>

                          <div className="result-item">

                            <span>
                              Expression
                            </span>

                            <strong>
                              {expression
                                ? expression[0]
                                : "Unknown"}
                            </strong>

                          </div>

                        </div>

                        {/* Expression probabilities */}

                        {expressions && (
                          <div className="expressions">

                            <h4>
                              Expressions
                            </h4>

                            {Object.entries(
                              expressions
                            ).map(
                              (
                                [
                                  name,
                                  value,
                                ]
                              ) => (
                                <div
                                  className="expression-row"
                                  key={name}
                                >

                                  <span>
                                    {name}
                                  </span>

                                  <div className="expression-bar">

                                    <div
                                      className="expression-fill"
                                      style={{
                                        width: `${
                                          value *
                                          100
                                        }%`,
                                      }}
                                    />

                                  </div>

                                  <strong>
                                    {Math.round(
                                      value *
                                        100
                                    )}
                                    %
                                  </strong>

                                </div>
                              )
                            )}

                          </div>
                        )}

                      </div>
                    );
                  }
                )}

              </div>

            )}

          </section>

        </div>

        {/* ====================================================
            CAPTURED IMAGE
        ==================================================== */}

        {capturedImage && (
          <section className="captured-section">

            <div className="section-heading">

              <div>
                <h2>
                  Captured Image
                </h2>

                <p>
                  Face capture from camera
                </p>
              </div>

              <button
                onClick={
                  downloadImage
                }
                className="control-btn primary"
              >
                <Download size={18} />
                Download
              </button>

            </div>

            <div className="captured-content">

              <img
                src={capturedImage}
                alt="Captured face"
                className="captured-image"
              />

            </div>

          </section>
        )}

        {/* ====================================================
            FEATURES
        ==================================================== */}

        <section className="features">

          <div className="feature-card">

            <div className="feature-icon">
              <ScanFace size={25} />
            </div>

            <h3>
              Face Detection
            </h3>

            <p>
              Real-time face detection
              using AI.
            </p>

          </div>

          <div className="feature-card">

            <div className="feature-icon">
              <User size={25} />
            </div>

            <h3>
              Age Estimation
            </h3>

            <p>
              AI-based estimated age
              analysis.
            </p>

          </div>

          <div className="feature-card">

            <div className="feature-icon">
              <Activity size={25} />
            </div>

            <h3>
              Expression Analysis
            </h3>

            <p>
              Detect facial expressions
              in real time.
            </p>

          </div>

          <div className="feature-card">

            <div className="feature-icon">
              <Settings size={25} />
            </div>

            <h3>
              Adjustable Detection
            </h3>

            <p>
              Configure AI confidence
              threshold.
            </p>

          </div>

        </section>

      </main>

      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer className="footer">

        <div className="footer-brand">

          <img
            src={logo}
            alt="eTouchUS"
          />

          <span>
            eTouchUS AI Face Recognition
          </span>

        </div>

        <p>
          Powered by React & face-api.js
        </p>

      </footer>

    </div>
  );
}

export default App;