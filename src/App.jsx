import { useState, useRef, useEffect } from "react";
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
} from "lucide-react";

import logo from "./assets/etouchus_face_recognition_logo.png";

import "./App.css";

function App() {
  // ============================================================
  // STATE
  // ============================================================

  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);

  // Live detections
  const [detections, setDetections] = useState([]);

  // Captured detections - these stay fixed
  const [capturedDetections, setCapturedDetections] = useState([]);

  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const [capturedImage, setCapturedImage] = useState(null);

  const [showSettings, setShowSettings] = useState(false);
  const [confidence, setConfidence] = useState(0.5);

  // IMPORTANT:
  // When true, live detection stops
  const [isCaptured, setIsCaptured] = useState(false);

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // ============================================================
  // LOAD MODELS
  // ============================================================

  useEffect(() => {
    const loadModels = async () => {
      try {
        setError(null);

        console.log("Loading AI models...");

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
          faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
          faceapi.nets.faceExpressionNet.loadFromUri("/models"),
          faceapi.nets.ageGenderNet.loadFromUri("/models"),
        ]);

        setIsModelLoaded(true);

        console.log("All AI models loaded successfully");
      } catch (err) {
        console.error("Error loading models:", err);

        setError(
          "Failed to load AI models. Make sure the model files are inside the public/models folder."
        );
      }
    };

    loadModels();
  }, []);

  // ============================================================
  // LIVE VIDEO PROCESSING
  // ============================================================

  const processVideo = async () => {
    // VERY IMPORTANT:
    // Don't process if an image has already been captured
    if (isCaptured) {
      return;
    }

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

    if (displaySize.width === 0 || displaySize.height === 0) {
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

      // Don't update live results if capture happened
      if (isCaptured) {
        return;
      }

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

      resizedDetections.forEach((detection, index) => {
        const box = detection.detection.box;

        const detectionScore =
          detection.detection.score;

        const age = Math.round(detection.age);

        const gender = detection.gender;

        const genderProbability = Math.round(
          detection.genderProbability * 100
        );

        // --------------------------------------------------------
        // FACE BOX
        // --------------------------------------------------------

        const drawBox = new faceapi.draw.DrawBox(box, {
          label:
            `Face ${index + 1} | ` +
            `Age: ${age} | ` +
            `Gender: ${gender} | ` +
            `Confidence: ${Math.round(
              detectionScore * 100
            )}%`,
        });

        drawBox.draw(canvas);

        // --------------------------------------------------------
        // LANDMARKS
        // --------------------------------------------------------

        const landmarks = detection.landmarks;

        const drawLandmarks =
          new faceapi.draw.DrawFaceLandmarks(
            box,
            landmarks
          );

        drawLandmarks.draw(canvas);

        // --------------------------------------------------------
        // FACE NUMBER
        // --------------------------------------------------------

        ctx.fillStyle = "#22c55e";

        ctx.font = "bold 14px Arial";

        ctx.fillText(
          `Face ${index + 1}`,
          box.x,
          Math.max(20, box.y - 10)
        );
      });
    } catch (err) {
      console.error(
        "Error processing video:",
        err
      );
    } finally {
      setIsProcessing(false);
    }
  };

  // ============================================================
  // START LIVE DETECTION
  // ============================================================

  useEffect(() => {
    // IMPORTANT:
    // If image has been captured, DON'T start interval
    if (
      !isCameraOn ||
      !isModelLoaded ||
      isCaptured
    ) {
      return;
    }

    const interval = setInterval(() => {
      processVideo();
    }, 150);

    return () => {
      clearInterval(interval);
    };
  }, [
    isCameraOn,
    isModelLoaded,
    confidence,
    isCaptured,
  ]);

  // ============================================================
  // CAPTURE IMAGE + DETECT ONCE
  // ============================================================

  const captureImage = async () => {
    if (!webcamRef.current) {
      setError("Camera is not available.");
      return;
    }

    try {
      setIsProcessing(true);
      setError(null);

      // --------------------------------------------------------
      // 1. TAKE A PHOTO FROM WEBCAM
      // --------------------------------------------------------

      const imageSrc =
        webcamRef.current.getScreenshot();

      if (!imageSrc) {
        setError("Unable to capture image.");
        setIsProcessing(false);
        return;
      }

      // --------------------------------------------------------
      // 2. FREEZE LIVE DETECTION
      // --------------------------------------------------------

      setIsCaptured(true);

      // Clear live detections
      setDetections([]);

      // --------------------------------------------------------
      // 3. SAVE CAPTURED IMAGE
      // --------------------------------------------------------

      setCapturedImage(imageSrc);

      // --------------------------------------------------------
      // 4. LOAD CAPTURED IMAGE
      // --------------------------------------------------------

      const image = await faceapi.fetchImage(
        imageSrc
      );

      // --------------------------------------------------------
      // 5. DETECT FACE ON CAPTURED IMAGE ONLY
      // --------------------------------------------------------

      const results = await faceapi
        .detectAllFaces(
          image,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 224,
            scoreThreshold: confidence,
          })
        )
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender();

      // --------------------------------------------------------
      // 6. SAVE FIXED CAPTURE RESULTS
      // --------------------------------------------------------

      setCapturedDetections(results);

      console.log(
        "Captured image analysis:",
        results
      );

      // --------------------------------------------------------
      // 7. DRAW RESULTS ON CAPTURE CANVAS
      // --------------------------------------------------------

      if (canvasRef.current) {
        const canvas = canvasRef.current;

        const displaySize = {
          width: image.width,
          height: image.height,
        };

        faceapi.matchDimensions(
          canvas,
          displaySize
        );

        const resizedDetections =
          faceapi.resizeResults(
            results,
            displaySize
          );

        const ctx =
          canvas.getContext("2d");

        ctx.clearRect(
          0,
          0,
          canvas.width,
          canvas.height
        );

        resizedDetections.forEach(
          (detection, index) => {
            const box =
              detection.detection.box;

            const detectionScore =
              detection.detection.score;

            const age =
              Math.round(
                detection.age
              );

            const gender =
              detection.gender;

            const drawBox =
              new faceapi.draw.DrawBox(
                box,
                {
                  label:
                    `Face ${
                      index + 1
                    } | ` +
                    `Age: ${age} | ` +
                    `Gender: ${gender} | ` +
                    `Confidence: ${Math.round(
                      detectionScore * 100
                    )}%`,
                }
              );

            drawBox.draw(canvas);

            const drawLandmarks =
              new faceapi.draw.DrawFaceLandmarks(
                box,
                detection.landmarks
              );

            drawLandmarks.draw(canvas);
          }
        );
      }
    } catch (err) {
      console.error(
        "Capture error:",
        err
      );

      setError(
        "Unable to analyze the captured image."
      );
    } finally {
      setIsProcessing(false);
    }
  };

  // ============================================================
  // RETAKE
  // ============================================================

  const retakeImage = () => {
    // Remove captured image
    setCapturedImage(null);

    // Remove captured results
    setCapturedDetections([]);

    // Remove error
    setError(null);

    // Clear canvas
    if (canvasRef.current) {
      const canvas =
        canvasRef.current;

      const ctx =
        canvas.getContext("2d");

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );
    }

    // Enable live detection again
    setIsCaptured(false);

    // Camera stays ON
    setIsCameraOn(true);
  };

  // ============================================================
  // TOGGLE CAMERA
  // ============================================================

  const toggleCamera = () => {
    setError(null);

    setDetections([]);

    setCapturedDetections([]);

    setCapturedImage(null);

    setIsCaptured(false);

    setIsCameraOn((current) => !current);
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

    link.href =
      capturedImage;

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);
  };

  // ============================================================
  // RESET
  // ============================================================

  const resetApp = () => {
    setIsCameraOn(false);

    setIsCaptured(false);

    setDetections([]);

    setCapturedDetections([]);

    setCapturedImage(null);

    setError(null);

    setShowSettings(false);

    if (canvasRef.current) {
      const canvas =
        canvasRef.current;

      const ctx =
        canvas.getContext("2d");

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );
    }
  };

  // ============================================================
  // VIDEO CONSTRAINTS
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
                alt="eTouchUS Face Recognition"
                className="logo"
              />

            </div>

            <div className="brand-text">

              <h1>
                AI Face Detector
              </h1>

              <p>
                Real-time face detection,
                age estimation and analysis
              </p>

            </div>

          </div>

          <div
            className={
              `model-status ${
                isModelLoaded
                  ? "ready"
                  : "loading"
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

        <section className="page-title">

          <div>

            <h2>
              Facial Recognition
            </h2>

            <p>
              Use your camera to detect faces
              and analyze facial attributes.
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
                    ? "danger"
                    : "primary"
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
                ? "Stop Camera"
                : "Start Camera"}

            </button>

            {isCameraOn && !isCaptured && (
              <>
                <button
                  onClick={captureImage}
                  className="control-btn secondary"
                  disabled={isProcessing}
                >

                  <Camera size={19} />

                  {isProcessing
                    ? "Capturing..."
                    : "Capture"}

                </button>

                <button
                  onClick={() =>
                    setShowSettings(
                      (current) =>
                        !current
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

            {/* RETAKE BUTTON */}

            {isCaptured && (
              <>

                <button
                  onClick={retakeImage}
                  className="control-btn primary"
                >

                  <RotateCcw size={19} />

                  Retake

                </button>

                <button
                  onClick={downloadImage}
                  className="control-btn secondary"
                >

                  <Download size={19} />

                  Download Image

                </button>

              </>
            )}

          </div>

          {/* ==================================================
              SETTINGS
          ================================================== */}

          {showSettings && !isCaptured && (

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
              Please wait while the face
              recognition models are loaded.
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
                {isCaptured
                  ? "Captured image - analysis is frozen."
                  : "Position your face in front of the camera."}
              </p>

            </div>

            <div
              className={
                `camera-indicator ${
                  isCameraOn
                    ? "active"
                    : ""
                }`
              }
            >

              <span></span>

              {isCameraOn
                ? isCaptured
                  ? "Captured"
                  : "Live"
                : "Offline"}

            </div>

          </div>

          <div className="camera-container">

            {isCameraOn ? (

              <div className="video-wrapper">

                {/* ==================================================
                    LIVE WEBCAM
                ================================================== */}

                {!isCaptured && (

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

                )}

                {/* ==================================================
                    CAPTURED IMAGE
                ================================================== */}

                {isCaptured &&
                  capturedImage && (

                    <img
                      src={capturedImage}
                      alt="Captured face"
                      className="webcam captured-camera-image"
                    />

                  )}

                <canvas
                  ref={canvasRef}
                  className="detection-canvas"
                />

                {/* ==================================================
                    SCAN FRAME
                ================================================== */}

                {!isCaptured && (

                  <div className="scan-frame">

                    <span className="corner top-left"></span>

                    <span className="corner top-right"></span>

                    <span className="corner bottom-left"></span>

                    <span className="corner bottom-right"></span>

                  </div>

                )}

                {/* ==================================================
                    PROCESSING
                ================================================== */}

                {isProcessing && (

                  <div className="processing-overlay">

                    <div className="processing-spinner"></div>

                    <span>
                      {isCaptured
                        ? "Analyzing captured image..."
                        : "Processing..."}
                    </span>

                  </div>

                )}

                {/* ==================================================
                    CAPTURED LABEL
                ================================================== */}

                {isCaptured && (

                  <div className="captured-label">

                    <CheckCircle size={16} />

                    Image Captured

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
                  Click "Start Camera" to
                  begin face detection.
                </p>

              </div>

            )}

          </div>

        </section>

        {/* ====================================================
            LIVE DETECTION RESULTS
        ==================================================== */}

        {!isCaptured &&
          detections.length > 0 && (

            <section className="detections-panel">

              <div className="section-heading">

                <div>

                  <h3>
                    Live Detection Results
                  </h3>

                  <p>
                    {detections.length} face
                    {detections.length > 1
                      ? "s"
                      : ""}{" "}
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
                      );

                    const gender =
                      detection.gender;

                    const genderProbability =
                      Math.round(
                        detection.genderProbability *
                          100
                      );

                    const faceConfidence =
                      Math.round(
                        detection.detection.score *
                          100
                      );

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

                    );
                  }
                )}

              </div>

            </section>

          )}


   {isCaptured &&
          capturedDetections.length > 0 && (

            <section className="detections-panel">

              <div className="section-heading">

                <div>

                  <h3>
                    Captured Face Results
                  </h3>

                  <p>
                    These results are from
                    the captured image.
                  </p>

                </div>

                <div className="detection-status">

                  <CheckCircle size={18} />

                  Fixed Result

                </div>

              </div>

              <div className="detections-grid">

                {capturedDetections.map(
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

                    const faceConfidence =
                      Math.round(
                        detection.detection.score *
                          100
                      );

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

                    );
                  }
                )}

              </div>

            </section>

          )}





        {/* ====================================================
            CAPTURED RESULTS
        ==================================================== */}

     

        {/* ====================================================
            NO FACE FOUND IN CAPTURE
        ==================================================== */}

        {isCaptured &&
          !isProcessing &&
          capturedDetections.length === 0 && (

            <section className="detections-panel">

              <div className="error-message">

                <AlertCircle size={20} />

                <div>

                  <strong>
                    No face detected
                  </strong>

                  <p>
                    No face was found in the
                    captured image. Click
                    Retake and position your
                    face inside the frame.
                  </p>

                </div>

              </div>

            </section>

          )}

        {/* ====================================================
            CAPTURED IMAGE DOWNLOAD SECTION
        ==================================================== */}

        {capturedImage && (

          <section className="captured-section">

            <div className="section-heading">

              <div>

                <h3>
                  Captured Image
                </h3>

                <p>
                  This image will remain
                  unchanged until you retake.
                </p>

              </div>

            </div>

            <div className="captured-content">

              <img
                src={capturedImage}
                alt="Captured face"
                className="captured-image"
              />

              <div
                style={{
                  display: "flex",
                  gap: "10px",
                  flexWrap: "wrap",
                }}
              >

                <button
                  onClick={downloadImage}
                  className="control-btn primary"
                >

                  <Download size={19} />

                  Download Image

                </button>

                <button
                  onClick={retakeImage}
                  className="control-btn secondary"
                >

                  <RotateCcw size={19} />

                  Retake

                </button>

              </div>

            </div>

          </section>

        )}

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
              eTouchUS AI Face Recognition
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