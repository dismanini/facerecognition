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

  const detectionRunningRef = useRef(false);
  const detectionTimerRef = useRef(null);

  // rPPG
  const rppgCanvasRef = useRef(null);
  const rppgSamplesRef = useRef([]);
  const rppgTimerRef = useRef(null);
  const heartRateTimerRef = useRef(null);

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
  // HEALTH
  // ============================================================

  const [heartRate, setHeartRate] = useState(null);
  const [heartRateStatus, setHeartRateStatus] =
    useState("Waiting");

  const [spo2, setSpo2] = useState(null);
  const [spo2Status, setSpo2Status] =
    useState("Sensor required");

  const [bloodPressure, setBloodPressure] = useState({
    systolic: null,
    diastolic: null,
  });

  const [bloodPressureStatus, setBloodPressureStatus] =
    useState("Sensor required");

  // ============================================================
  // LOAD FACE API MODELS
  // ============================================================

  useEffect(() => {
    let mounted = true;

    const loadModels = async () => {
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

        console.log("All face-api.js models loaded.");
      } catch (err) {
        console.error("MODEL ERROR:", err);

        if (mounted) {
          setError(
            "AI models could not be loaded. Make sure all face-api.js model files are inside public/models."
          );
        }
      }
    };

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
      "Camera could not start. Please allow camera permission and make sure another application is not using the webcam."
    );
  }, []);

  // ============================================================
  // CREATE rPPG CANVAS
  // ============================================================

  const createRppgCanvas = useCallback(() => {
    if (!rppgCanvasRef.current) {
      const canvas = document.createElement("canvas");

      canvas.width = 320;
      canvas.height = 240;

      rppgCanvasRef.current = canvas;
    }

    return rppgCanvasRef.current;
  }, []);

  // ============================================================
  // COLLECT rPPG SAMPLE
  // ============================================================

  const collectRppgSample = useCallback(() => {
    if (!cameraOn || !cameraReady) {
      return;
    }

    const webcam = webcamRef.current;

    if (!webcam || !webcam.video) {
      return;
    }

    const video = webcam.video;

    if (
      video.readyState !== 4 ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return;
    }

    const canvas = createRppgCanvas();

    const ctx = canvas.getContext("2d", {
      willReadFrequently: true,
    });

    if (!ctx) {
      return;
    }

    ctx.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    // Approximate central face ROI.
    // This is experimental and not medically validated.
    const roiX = 90;
    const roiY = 45;
    const roiWidth = 140;
    const roiHeight = 70;

    const imageData = ctx.getImageData(
      roiX,
      roiY,
      roiWidth,
      roiHeight
    );

    let red = 0;
    let green = 0;
    let blue = 0;
    let validPixels = 0;

    for (
      let i = 0;
      i < imageData.data.length;
      i += 4
    ) {
      const r = imageData.data[i];
      const g = imageData.data[i + 1];
      const b = imageData.data[i + 2];

      const brightness = (r + g + b) / 3;

      if (
        brightness > 40 &&
        brightness < 240
      ) {
        red += r;
        green += g;
        blue += b;
        validPixels++;
      }
    }

    if (validPixels === 0) {
      setHeartRateStatus("No signal");
      return;
    }

    const sample = {
      time: performance.now(),
      red: red / validPixels,
      green: green / validPixels,
      blue: blue / validPixels,
    };

    rppgSamplesRef.current.push(sample);

    if (rppgSamplesRef.current.length > 300) {
      rppgSamplesRef.current =
        rppgSamplesRef.current.slice(-300);
    }
  }, [
    cameraOn,
    cameraReady,
    createRppgCanvas,
  ]);

  // ============================================================
  // CALCULATE HEART RATE
  // ============================================================

  const calculateHeartRate = useCallback(() => {
    const samples = rppgSamplesRef.current;

    if (samples.length < 150) {
      setHeartRateStatus(
        "Collecting signal " +
          samples.length +
          "/150"
      );

      return;
    }

    const recentSamples = samples.slice(-300);

    const greenValues = recentSamples.map(
      (sample) => sample.green
    );

    if (greenValues.length < 2) {
      return;
    }

    // Remove DC component
    const mean =
      greenValues.reduce(
        (sum, value) => sum + value,
        0
      ) / greenValues.length;

    const signal = greenValues.map(
      (value) => value - mean
    );

    // Find amplitude
    let min = Infinity;
    let max = -Infinity;

    signal.forEach((value) => {
      if (value < min) {
        min = value;
      }

      if (value > max) {
        max = value;
      }
    });

    const amplitude = max - min;

    if (amplitude < 0.5) {
      setHeartRateStatus("Signal too weak");
      return;
    }

    // Zero-crossing estimation
    let crossings = 0;

    for (let i = 1; i < signal.length; i++) {
      if (
        signal[i - 1] <= 0 &&
        signal[i] > 0
      ) {
        crossings++;
      }
    }

    const firstTime = recentSamples[0].time;

    const lastTime =
      recentSamples[
        recentSamples.length - 1
      ].time;

    const durationSeconds =
      (lastTime - firstTime) / 1000;

    if (durationSeconds <= 0) {
      return;
    }

    const frequency =
      crossings / durationSeconds;

    const bpm = Math.round(
      frequency * 60
    );

    if (bpm >= 45 && bpm <= 180) {
      setHeartRate(bpm);
      setHeartRateStatus("Estimated");
    } else {
      setHeartRateStatus("Signal unclear");
    }
  }, []);

  // ============================================================
  // START rPPG
  // ============================================================

  useEffect(() => {
    if (!cameraOn || !cameraReady) {
      return;
    }

    console.log(
      "Starting experimental rPPG collection..."
    );

    rppgSamplesRef.current = [];

    setHeartRate(null);
    setHeartRateStatus("Collecting signal");

    rppgTimerRef.current = setInterval(() => {
      collectRppgSample();
    }, 33);

    return () => {
      if (rppgTimerRef.current) {
        clearInterval(
          rppgTimerRef.current
        );

        rppgTimerRef.current = null;
      }
    };
  }, [
    cameraOn,
    cameraReady,
    collectRppgSample,
  ]);

  // ============================================================
  // CALCULATE HEART RATE EVERY 3 SECONDS
  // ============================================================

  useEffect(() => {
    if (!cameraOn || !cameraReady) {
      return;
    }

    heartRateTimerRef.current =
      setInterval(() => {
        calculateHeartRate();
      }, 3000);

    return () => {
      if (heartRateTimerRef.current) {
        clearInterval(
          heartRateTimerRef.current
        );

        heartRateTimerRef.current = null;
      }
    };
  }, [
    cameraOn,
    cameraReady,
    calculateHeartRate,
  ]);

  // ============================================================
  // FACE DETECTION
  // ============================================================

  const detectFaces = useCallback(async () => {
    if (!cameraOn) {
      return;
    }

    if (!cameraReady) {
      return;
    }

    if (!modelsLoaded) {
      return;
    }

    if (detectionRunningRef.current) {
      return;
    }

    const webcam = webcamRef.current;
    const canvas = canvasRef.current;

    if (!webcam || !canvas) {
      return;
    }

    const video = webcam.video;

    if (!video) {
      return;
    }

    if (video.readyState !== 4) {
      return;
    }

    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return;
    }

    detectionRunningRef.current = true;
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

      const results =
        await faceapi
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

      const ctx =
        canvas.getContext("2d");

      if (!ctx) {
        return;
      }

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      resizedResults.forEach(
        (result, index) => {
          const box =
            result.detection.box;

          const age = Math.round(
            result.age
          );

          const gender =
            result.gender;

          const faceScore =
            Math.round(
              result.detection.score * 100
            );

          const genderScore =
            Math.round(
              result.genderProbability * 100
            );

          const label =
            "Face " +
            (index + 1) +
            " | Age: " +
            age +
            " | Gender: " +
            gender +
            " | Confidence: " +
            faceScore +
            "%";

          const drawBox =
            new faceapi.draw.DrawBox(
              box,
              {
                label: label,
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
          ctx.font =
            "bold 14px Arial";

          ctx.fillText(
            "Gender " +
              genderScore +
              "%",
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
      detectionRunningRef.current =
        false;

      setIsProcessing(false);
    }
  }, [
    cameraOn,
    cameraReady,
    modelsLoaded,
    confidence,
  ]);

  // ============================================================
  // FACE DETECTION LOOP
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
      if (stopped) {
        return;
      }

      await detectFaces();

      if (!stopped) {
        detectionTimerRef.current =
          setTimeout(
            runLoop,
            400
          );
      }
    };

    runLoop();

    return () => {
      stopped = true;

      if (detectionTimerRef.current) {
        clearTimeout(
          detectionTimerRef.current
        );

        detectionTimerRef.current =
          null;
      }

      detectionRunningRef.current =
        false;
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

    setCapturedImage(null);
    setCapturedDetections([]);

    setCameraReady(false);
    setCameraOn(true);
  };

  // ============================================================
  // STOP CAMERA
  // ============================================================

  const stopCamera = () => {
    if (detectionTimerRef.current) {
      clearTimeout(
        detectionTimerRef.current
      );

      detectionTimerRef.current = null;
    }

    if (rppgTimerRef.current) {
      clearInterval(
        rppgTimerRef.current
      );

      rppgTimerRef.current = null;
    }

    if (heartRateTimerRef.current) {
      clearInterval(
        heartRateTimerRef.current
      );

      heartRateTimerRef.current = null;
    }

    detectionRunningRef.current =
      false;

    rppgSamplesRef.current = [];

    setHeartRate(null);
    setHeartRateStatus("Waiting");

    setCameraReady(false);
    setCameraOn(false);

    setDetections([]);
    setIsProcessing(false);
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
        "Camera is still starting. Please wait."
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
        "Could not capture camera image."
      );

      return;
    }

    setCapturedImage(image);

    const captured =
      detections.map((detection) => ({
        age: detection.age,
        gender: detection.gender,
        genderProbability:
          detection.genderProbability,
        confidence:
          detection.detection.score,
      }));

    setCapturedDetections(captured);

    setError("");

    console.log("IMAGE CAPTURED");
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

    link.href = capturedImage;
    link.download =
      "face-analysis.jpg";

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);
  };

  // ============================================================
  // RESET
  // ============================================================

  const resetApp = () => {
    if (detectionTimerRef.current) {
      clearTimeout(
        detectionTimerRef.current
      );

      detectionTimerRef.current = null;
    }

    if (rppgTimerRef.current) {
      clearInterval(
        rppgTimerRef.current
      );

      rppgTimerRef.current = null;
    }

    if (heartRateTimerRef.current) {
      clearInterval(
        heartRateTimerRef.current
      );

      heartRateTimerRef.current = null;
    }

    detectionRunningRef.current =
      false;

    rppgSamplesRef.current = [];

    setCameraReady(false);
    setCameraOn(false);

    setDetections([]);
    setCapturedDetections([]);
    setCapturedImage(null);

    setHeartRate(null);
    setHeartRateStatus("Waiting");

    setSpo2(null);
    setSpo2Status("Sensor required");

    setBloodPressure({
      systolic: null,
      diastolic: null,
    });

    setBloodPressureStatus(
      "Sensor required"
    );

    setError("");
    setShowSettings(false);
  };

  // ============================================================
  // OPTIONAL SENSOR FUNCTIONS
  // ============================================================
  //
  // These functions are placeholders for REAL hardware.
  //
  // Do not generate fake SpO2/BP values.
  //
  // A real implementation should receive data from:
  //
  // SpO2:
  //   Bluetooth pulse oximeter / approved sensor
  //
  // Blood pressure:
  //   Bluetooth/USB BP monitor
  //
  // ============================================================

  const updateSpO2FromSensor = (
    oxygenValue
  ) => {
    const numericValue =
      Number(oxygenValue);

    if (
      Number.isFinite(numericValue) &&
      numericValue >= 50 &&
      numericValue <= 100
    ) {
      setSpo2(numericValue);
      setSpo2Status("Sensor connected");
    }
  };

  const updateBloodPressureFromSensor = (
    systolic,
    diastolic
  ) => {
    const sys = Number(systolic);
    const dia = Number(diastolic);

    if (
      Number.isFinite(sys) &&
      Number.isFinite(dia) &&
      sys > 0 &&
      dia > 0
    ) {
      setBloodPressure({
        systolic: sys,
        diastolic: dia,
      });

      setBloodPressureStatus(
        "Sensor connected"
      );
    }
  };

  // Prevent unused-function warnings in some configurations.
  // These functions will be called when the hardware integration
  // is added.
  void updateSpO2FromSensor;
  void updateBloodPressureFromSensor;

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
                Real-time facial analysis
                and health monitoring
              </p>
            </div>

          </div>

          <div
            className={
              "model-status " +
              (modelsLoaded
                ? "ready"
                : "loading")
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
              Analyze facial attributes and
              connect validated health
              measurement systems.
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
                    )}
                    %
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
                      Number(
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
            CAMERA
        ==================================================== */}

        <section className="camera-card">

          <div className="camera-header">

            <div>
              <h3>
                Camera
              </h3>

              <p>
                Position your face in front
                of the camera.
              </p>
            </div>

            <div
              className={
                "camera-indicator " +
                (cameraReady
                  ? "active"
                  : "")
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
                  Click "Start Camera"
                  to begin face detection.
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
                    : ""}{" "}
                  detected
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
                          <UserRound
                            size={22}
                          />
                        </div>

                        <div>
                          <h4>
                            Face{" "}
                            {index + 1}
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
                  This image was captured
                  manually.
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

                {capturedDetections.length ===
                0 ? (
                  <p>
                    No face was detected
                    at the moment of capture.
                  </p>
                ) : (
                  capturedDetections.map(
                    (detection, index) => (
                      <div
                        className="captured-result-card"
                        key={index}
                      >

                        <strong>
                          Face{" "}
                          {index + 1}
                        </strong>

                        <span>
                          Age:{" "}
                          {Math.round(
                            detection.age
                          )}{" "}
                          years
                        </span>

                        <span>
                          Gender:{" "}
                          {detection.gender}
                        </span>

                        <span>
                          Gender confidence:{" "}
                          {Math.round(
                            detection.genderProbability *
                              100
                          )}
                          %
                        </span>

                        <span>
                          Face confidence:{" "}
                          {Math.round(
                            detection.confidence *
                              100
                          )}
                          %
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
                Heart-rate camera estimation
                is experimental. SpO₂ and
                blood-pressure measurements
                require appropriate validated
                sensors.
              </p>

            </div>

            <div className="health-disclaimer">
              <ShieldCheck size={16} />
              Measurement status
            </div>

          </div>

          <div className="health-grid">

            {/* ==================================================
                HEART RATE
            ================================================== */}

            <div className="health-card">

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

                {heartRate ?? "--"}

                <small>
                  BPM
                </small>

              </div>

              <p className="health-description">
                Experimental camera-based
                rPPG estimate. Keep your face
                still and use stable lighting.
              </p>

              <div className="measurement-message">
                <Activity size={15} />

                {heartRate
                  ? "Pulse signal detected"
                  : "Collecting facial pulse signal"}
              </div>

            </div>

            {/* ==================================================
                SPO2
            ================================================== */}

            <div className="health-card">

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

                {spo2 ?? "--"}

                <small>
                  %
                </small>

              </div>

              <p className="health-description">
                Real SpO₂ requires a pulse
                oximeter or other validated
                oxygen sensor.
              </p>

              <div className="measurement-message">
                <Info size={15} />
                Sensor required
              </div>

            </div>

            {/* ==================================================
                BLOOD PRESSURE
            ================================================== */}

            <div className="health-card">

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
                    {bloodPressure.systolic ??
                      "--"}
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
                    {bloodPressure.diastolic ??
                      "--"}
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
                Real blood pressure requires
                a validated BP monitor or
                approved measurement system.
              </p>

              <div className="measurement-message">
                <Info size={15} />
                BP sensor required
              </div>

            </div>

          </div>
        </section>

        {/* ====================================================
            REAL SENSOR INTEGRATION NOTICE
        ==================================================== */}

        <div className="medical-notice">

          <ShieldCheck size={20} />

          <div>

            <strong>
              Real Measurement Notice
            </strong>

            <p>
              The heart-rate value from this
              prototype is an experimental
              camera/rPPG estimate. Blood oxygen
              and blood pressure are intentionally
              not generated from the webcam.
              To display real SpO₂ and BP values,
              connect a compatible validated
              sensor/device and send its readings
              to this React application.
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