// src/components/ModelTester.jsx
import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import MediaHandUtils from '../utils/mediaHandUtils';
import './ModelTester.css';

function ModelTester({ model, metadata }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [mediaHandUtils, setMediaHandUtils] = useState(null);
  const [isTracking, setIsTracking] = useState(false);
  const [currentLandmarks, setCurrentLandmarks] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);

  // For visual feedback
  const [lastGesture, setLastGesture] = useState(null);
  const [confidenceHistory, setConfidenceHistory] = useState([]);
  const maxHistoryLength = 30; // Store 30 frames of confidence history

  // Initialize MediaPipe Hands
  useEffect(() => {
    let isMounted = true;
    setIsInitializing(true);

    const initializeHandTracking = async () => {
      try {
        const handUtils = new MediaHandUtils();
        await handUtils.initialize();

        if (isMounted) {
          setMediaHandUtils(handUtils);
          setIsInitializing(false);
          setError(null);
        }
      } catch (err) {
        if (isMounted) {
          setError(`Initialization error: ${err.message}`);
          setIsInitializing(false);
        }
      }
    };

    initializeHandTracking();

    return () => {
      isMounted = false;
      if (mediaHandUtils && isTracking) {
        try {
          mediaHandUtils.stopTracking();
        } catch (err) {
          console.error('Error stopping tracking on cleanup:', err);
        }
      }
    };
  }, []);

  // Start/stop tracking
  const toggleTracking = async () => {
    try {
      if (!mediaHandUtils) {
        setError("Hand tracking not initialized yet. Please wait...");
        return;
      }

      if (isTracking) {
        await mediaHandUtils.stopTracking();
        setIsTracking(false);
        setCurrentLandmarks(null);
        setPrediction(null);
      } else {
        if (videoRef.current) {
          mediaHandUtils.setVideoElement(videoRef.current);

          // Register hand update callback
          mediaHandUtils.onHandUpdate(results => {
            drawResults(results);

            // Store the current landmarks for testing
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
              setCurrentLandmarks(results.multiHandLandmarks[0]);

              // Predict gesture with the model
              predictGesture(results.multiHandLandmarks[0]);
            } else {
              setCurrentLandmarks(null);
              setPrediction(null);
            }
          });

          await mediaHandUtils.startTracking();
          setIsTracking(true);
          setError(null);
        }
      }
    } catch (err) {
      console.error('Error toggling tracking:', err);
      setError(`Error: ${err.message}`);
      setIsTracking(false);
    }
  };

  // Draw hand tracking results
  const drawResults = (results) => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw hand landmarks if available
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        // Draw connections
        drawHandConnections(ctx, landmarks, width, height);

        // Draw landmarks
        for (const landmark of landmarks) {
          ctx.fillStyle = '#00FF00';
          ctx.beginPath();
          ctx.arc(
            landmark.x * width,
            landmark.y * height,
            5,
            0,
            2 * Math.PI
          );
          ctx.fill();
        }
      }
    }

    // Draw the prediction text
    if (prediction) {
      drawPredictionOverlay(ctx, width, height);
    }
  };

  // Draw connections between landmarks
  const drawHandConnections = (ctx, landmarks, width, height) => {
    if (!landmarks || landmarks.length < 21) return;

    // Define connections between landmarks (finger joints)
    const connections = [
      // Thumb
      [0, 1], [1, 2], [2, 3], [3, 4],
      // Index finger
      [0, 5], [5, 6], [6, 7], [7, 8],
      // Middle finger
      [0, 9], [9, 10], [10, 11], [11, 12],
      // Ring finger
      [0, 13], [13, 14], [14, 15], [15, 16],
      // Pinky
      [0, 17], [17, 18], [18, 19], [19, 20],
      // Palm
      [0, 5], [5, 9], [9, 13], [13, 17]
    ];

    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;

    for (const [i, j] of connections) {
      ctx.beginPath();
      ctx.moveTo(
        landmarks[i].x * width,
        landmarks[i].y * height
      );
      ctx.lineTo(
        landmarks[j].x * width,
        landmarks[j].y * height
      );
      ctx.stroke();
    }
  };

  // Draw prediction overlay
  const drawPredictionOverlay = (ctx, width, height) => {
    if (!prediction) return;

    ctx.save();

    // Draw a semi-transparent panel at the bottom
    const panelHeight = 100;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(0, height - panelHeight, width, panelHeight);

    // Draw the detected gesture
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(
      `Detected: ${formatGestureName(prediction.topGesture.gesture)}`,
      width / 2,
      height - panelHeight / 2 - 15
    );

    // Draw confidence
    const confidence = prediction.topGesture.score;
    ctx.fillStyle = getConfidenceColor(confidence);
    ctx.fillText(
      `Confidence: ${(confidence * 100).toFixed(1)}%`,
      width / 2,
      height - panelHeight / 2 + 15
    );

    // Draw confidence bar
    const barWidth = width * 0.7;
    const barHeight = 10;
    const barX = (width - barWidth) / 2;
    const barY = height - 20;

    // Bar background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.fillRect(barX, barY, barWidth, barHeight);

    // Bar fill
    ctx.fillStyle = getConfidenceColor(confidence);
    ctx.fillRect(barX, barY, barWidth * confidence, barHeight);

    ctx.restore();
  };

  // Get color based on confidence
  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return '#4cd137'; // High confidence: green
    if (confidence > 0.5) return '#fbc531'; // Medium confidence: yellow
    return '#e84118'; // Low confidence: red
  };

  // Setup canvas for visualization
  useEffect(() => {
    if (!canvasRef.current || !videoRef.current) return;

    const updateCanvasSize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const { videoWidth, videoHeight } = videoRef.current;
      if (videoWidth && videoHeight) {
        canvas.width = videoWidth;
        canvas.height = videoHeight;
      }
    };

    // Initial size update
    updateCanvasSize();

    // Update size when video dimensions change
    const handleResize = () => updateCanvasSize();
    videoRef.current.addEventListener('loadedmetadata', handleResize);
    window.addEventListener('resize', handleResize);

    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener('loadedmetadata', handleResize);
      }
      window.addEventListener('resize', handleResize);
    };
  }, [videoRef, canvasRef]);

  // Normalize landmarks for model input
  const normalizeLandmarks = (landmarks) => {
    if (!landmarks || landmarks.length !== 21) {
      return null;
    }

    // Flatten landmarks to [x1,y1,z1,x2,y2,z2,...]
    const flatLandmarks = landmarks.flatMap(landmark => [
      landmark.x, landmark.y, landmark.z
    ]);

    // Get wrist coordinates (first landmark)
    const wristX = flatLandmarks[0];
    const wristY = flatLandmarks[1];
    const wristZ = flatLandmarks[2];

    // Subtract wrist position to center at origin
    const centeredLandmarks = new Array(flatLandmarks.length);
    for (let i = 0; i < flatLandmarks.length; i += 3) {
      centeredLandmarks[i] = flatLandmarks[i] - wristX;
      centeredLandmarks[i + 1] = flatLandmarks[i + 1] - wristY;
      centeredLandmarks[i + 2] = flatLandmarks[i + 2] - wristZ;
    }

    // Calculate scale factor based on middle finger tip distance
    const middleTipIndex = 12 * 3; // Index 12 is middle finger tip
    const dx = centeredLandmarks[middleTipIndex];
    const dy = centeredLandmarks[middleTipIndex + 1];
    const dz = centeredLandmarks[middleTipIndex + 2];
    const distance = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
    const scale = 1 / distance;

    // Apply scaling to all points
    const normalizedLandmarks = new Array(centeredLandmarks.length);
    for (let i = 0; i < centeredLandmarks.length; i++) {
      normalizedLandmarks[i] = centeredLandmarks[i] * scale;
    }

    return normalizedLandmarks;
  };

  // Predict gesture using the model
  const predictGesture = async (landmarks) => {
    if (!model || !landmarks) return;

    try {
      const normalizedLandmarks = normalizeLandmarks(landmarks);
      if (!normalizedLandmarks) return;

      // Prepare input tensor
      const inputTensor = tf.tensor2d([normalizedLandmarks]);

      // Get prediction
      const predictionTensor = model.predict(inputTensor);
      const scores = predictionTensor.dataSync();

      // Get gesture mapping
      const gestures = metadata?.gestures ||
        Array.from({ length: scores.length }, (_, i) => `gesture_${i}`);

      // Create scores object
      const scoreObj = {};
      gestures.forEach((gesture, i) => {
        scoreObj[gesture] = scores[i];
      });

      // Get top 3 predictions
      const topPredictions = Object.entries(scoreObj)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([gesture, score]) => ({ gesture, score }));

      // Use smoothing for more stable predictions
      updatePrediction(topPredictions[0]);

      // Clean up tensors
      inputTensor.dispose();
      predictionTensor.dispose();
    } catch (error) {
      console.error('Error predicting gesture:', error);
    }
  };

  // Update prediction with temporal smoothing
  const updatePrediction = (newPrediction) => {
    // Update confidence history for visualization
    setConfidenceHistory(prev => {
      const updated = [...prev, newPrediction.score];
      if (updated.length > maxHistoryLength) {
        return updated.slice(-maxHistoryLength);
      }
      return updated;
    });

    // Apply temporal stability - only update if the gesture changes or confidence is higher
    setLastGesture(prev => {
      if (!prev ||
          prev.gesture !== newPrediction.gesture ||
          newPrediction.score > prev.score + 0.1) {
        return newPrediction;
      }
      return prev;
    });

    // Update the current prediction
    setPrediction({
      topGesture: newPrediction,
      time: Date.now()
    });
  };

  // Format gesture name for display
  const formatGestureName = (name) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // Generate confidence history chart
  const renderConfidenceChart = () => {
    if (!confidenceHistory.length) return null;

    const chartHeight = 100;
    const chartWidth = 300;

    return (
      <div className="confidence-chart">
        <h4>Confidence History</h4>
        <svg width={chartWidth} height={chartHeight} className="chart-svg">
          {/* Draw the background grid */}
          <line x1="0" y1={chartHeight} x2={chartWidth} y2={chartHeight}
                stroke="#666" strokeWidth="1" />

          <line x1="0" y1={chartHeight * 0.2} x2={chartWidth} y2={chartHeight * 0.2}
                stroke="#666" strokeDasharray="2,2" strokeWidth="1" />

          <line x1="0" y1={chartHeight * 0.5} x2={chartWidth} y2={chartHeight * 0.5}
                stroke="#666" strokeDasharray="2,2" strokeWidth="1" />

          <line x1="0" y1={chartHeight * 0.8} x2={chartWidth} y2={chartHeight * 0.8}
                stroke="#666" strokeDasharray="2,2" strokeWidth="1" />

          {/* Draw the y-axis labels */}
          <text x="5" y={chartHeight * 0.2 - 5} fontSize="10" fill="#999">0.8</text>
          <text x="5" y={chartHeight * 0.5 - 5} fontSize="10" fill="#999">0.5</text>
          <text x="5" y={chartHeight * 0.8 - 5} fontSize="10" fill="#999">0.2</text>

          {/* Draw the confidence line */}
          <polyline
            points={confidenceHistory.map((confidence, index) => {
              const x = (index / (maxHistoryLength - 1)) * chartWidth;
              const y = chartHeight - (confidence * chartHeight);
              return `${x},${y}`;
            }).join(' ')}
            fill="none"
            stroke="#4a8fe7"
            strokeWidth="2"
          />
        </svg>
      </div>
    );
  };

  return (
    <div className="model-tester">
      <h3>Test Your Trained Model</h3>
      <p className="tester-description">
        Check how well your model recognizes gestures in real-time
      </p>

      <div className="tester-controls">
        <button
          onClick={toggleTracking}
          disabled={isInitializing}
          className={isTracking ? "stop-button" : "start-button"}
        >
          {isTracking ? 'Stop Testing' : 'Start Testing'}
        </button>
      </div>

      {isInitializing && <div className="status">Initializing hand tracking...</div>}
      {error && <div className="error">{error}</div>}

      <div className="test-view-container">
        <div className="video-container">
          <video
            ref={videoRef}
            className="input-video"
            playsInline
            muted
          ></video>
          <canvas
            ref={canvasRef}
            className="output-canvas"
          ></canvas>
        </div>

        <div className="prediction-panel">
          <h4>Real-time Prediction</h4>

          <div className="current-gesture">
            <span className="label">Detected Gesture:</span>
            <span className="value">
              {prediction ? formatGestureName(prediction.topGesture.gesture) : 'None'}
            </span>
          </div>

          <div className="confidence-meter">
            <span className="label">Confidence:</span>
            <div className="meter-container">
              <div
                className="meter-fill"
                style={{
                  width: `${prediction ? prediction.topGesture.score * 100 : 0}%`,
                  backgroundColor: prediction ? getConfidenceColor(prediction.topGesture.score) : '#ccc'
                }}
              ></div>
            </div>
            <span className="meter-value">
              {prediction ? `${(prediction.topGesture.score * 100).toFixed(1)}%` : '0%'}
            </span>
          </div>

          {renderConfidenceChart()}

          <div className="test-instructions">
            <h4>Testing Instructions:</h4>
            <ul>
              <li>Try each gesture you trained at different positions and angles</li>
              <li>Check if the model can recognize gestures consistently</li>
              <li>If accuracy is low, consider collecting more training data</li>
              <li>Ideal confidence should be above 70% for reliable detection</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelTester;
