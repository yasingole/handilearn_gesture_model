import React, { useEffect, useRef, useState } from 'react';
import MediaHandUtils from '../utils/mediaHandUtils';
import GesturePreview from './GesturePreview';
import './GestureDataCollector.css';

function GestureDataCollector({ gestureLabels = [] }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [mediaHandUtils, setMediaHandUtils] = useState(null);
  const [isTracking, setIsTracking] = useState(false);
  const [currentLandmarks, setCurrentLandmarks] = useState(null);
  const [error, setError] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);

  // For data collection
  const [currentLabel, setCurrentLabel] = useState(gestureLabels[0] || 'open_hand');
  const [collectedSamples, setCollectedSamples] = useState({});
  const [recordingInterval, setRecordingInterval] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [samplesCount, setSamplesCount] = useState(0);
  const [recordingRate, setRecordingRate] = useState(500); // ms between samples

  // Default gesture labels if none provided
  const defaultGestureLabels = [
    'open_hand',
    'fist',
    'point',
    'thumbs_up',
    'pinch',
    'wave',
    'ok_sign',
    'victory'
  ];

  const effectiveGestureLabels = gestureLabels.length > 0 ? gestureLabels : defaultGestureLabels;

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
          console.log('Hand tracking initialized successfully');
        }
      } catch (err) {
        if (isMounted) {
          console.error('Error initializing hand tracking:', err);
          setError(`Initialization error: ${err.message}`);
          setIsInitializing(false);
        }
      }
    };

    initializeHandTracking();

    // Initialize collectedSamples
    const initialSamples = {};
    effectiveGestureLabels.forEach(label => {
      initialSamples[label] = [];
    });
    setCollectedSamples(initialSamples);

    // Clean up on unmount
    return () => {
      isMounted = false;
      if (recordingInterval) {
        clearInterval(recordingInterval);
      }
      if (mediaHandUtils && isTracking) {
        try {
          mediaHandUtils.stopTracking();
        } catch (err) {
          console.error('Error stopping tracking on cleanup:', err);
        }
      }
    };
  }, []);

  // Handle starting/stopping tracking
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

        // Also stop recording if active
        if (recordingInterval) {
          clearInterval(recordingInterval);
          setRecordingInterval(null);
          setIsRecording(false);
        }
      } else {
        if (videoRef.current) {
          mediaHandUtils.setVideoElement(videoRef.current);

          // Register hand update callback
          mediaHandUtils.onHandUpdate(results => {
            drawResults(results);

            // Store the current landmarks for capture
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
              setCurrentLandmarks(results.multiHandLandmarks[0]);
            } else {
              setCurrentLandmarks(null);
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

  // Draw hand tracking results on canvas
  const drawResults = (results) => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw hand landmarks if available
    if (results.multiHandLandmarks) {
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
  };

  // Draw connections between landmarks to show hand structure
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
      window.removeEventListener('resize', handleResize);
    };
  }, [videoRef, canvasRef]);

  // Capture a single hand landmark sample
  const captureSample = () => {
    if (!currentLandmarks || currentLandmarks.length !== 21) {
      return;
    }

    // Extract just the x, y, z coordinates into a flat array
    const flattenedLandmarks = currentLandmarks.flatMap(landmark => [
      landmark.x, landmark.y, landmark.z
    ]);

    // Add to the appropriate label's collection
    setCollectedSamples(prevSamples => {
      const updatedSamples = { ...prevSamples };
      updatedSamples[currentLabel] = [
        ...updatedSamples[currentLabel],
        {
          landmarks: flattenedLandmarks,
          label: currentLabel,
          timestamp: Date.now()
        }
      ];
      return updatedSamples;
    });

    // Update the total samples count
    setSamplesCount(prevCount => prevCount + 1);
  };

  // Start/stop recording samples at intervals
  const toggleRecording = () => {
    if (isRecording) {
      // Stop recording
      if (recordingInterval) {
        clearInterval(recordingInterval);
        setRecordingInterval(null);
      }
      setIsRecording(false);
    } else {
      // Start recording
      if (!isTracking) {
        setError("Please start tracking first!");
        return;
      }

      // Set up an interval to capture samples
      const interval = setInterval(() => {
        if (currentLandmarks) {
          captureSample();
        }
      }, recordingRate);

      setRecordingInterval(interval);
      setIsRecording(true);
    }
  };

  // Export collected data as a JSON file
  const exportData = () => {
    // Prepare the data in the desired format
    const dataToExport = {
      gestures: collectedSamples,
      metadata: {
        totalSamples: samplesCount,
        dateCollected: new Date().toISOString(),
        gestureCounts: Object.fromEntries(
          Object.entries(collectedSamples).map(([label, samples]) => [
            label, samples.length
          ])
        )
      }
    };

    // Convert to JSON
    const jsonData = JSON.stringify(dataToExport, null, 2);

    // Create a blob and download link
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    // Create temporary download link
    const downloadLink = document.createElement('a');
    downloadLink.href = url;
    downloadLink.download = 'handilearn_gesture_data.json';

    // Trigger download
    document.body.appendChild(downloadLink);
    downloadLink.click();

    // Clean up
    document.body.removeChild(downloadLink);
    URL.revokeObjectURL(url);
  };

  // Get counts for each gesture
  const getGestureCounts = () => {
    return Object.fromEntries(
      Object.entries(collectedSamples).map(([label, samples]) => [
        label, samples.length
      ])
    );
  };

  // Format gesture name for display
  const formatGestureName = (name) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <div className="gesture-data-collector">
      <h2>Gesture Data Collector</h2>

      <div className="controls">
        <button
          onClick={toggleTracking}
          disabled={isInitializing}
          className={isTracking ? "stop-button" : "start-button"}
        >
          {isTracking ? 'Stop Tracking' : 'Start Tracking'}
        </button>
      </div>

      {isInitializing && <div className="status">Initializing hand tracking...</div>}
      {error && <div className="error">{error}</div>}

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

      <div className="gesture-preview">
        {currentLandmarks && (
          <GesturePreview
            landmarks={currentLandmarks}
            label={currentLabel}
          />
        )}
      </div>

      <div className="data-collection-panel">
        <div className="gesture-selector">
          <label htmlFor="gesture-select">Current Gesture:</label>
          <select
            id="gesture-select"
            value={currentLabel}
            onChange={(e) => setCurrentLabel(e.target.value)}
            disabled={isRecording}
          >
            {effectiveGestureLabels.map(label => (
              <option key={label} value={label}>
                {formatGestureName(label)}
              </option>
            ))}
          </select>
        </div>

        <div className="recording-controls">
          <div className="recording-rate">
            <label htmlFor="rate-select">Sample Rate:</label>
            <select
              id="rate-select"
              value={recordingRate}
              onChange={(e) => setRecordingRate(Number(e.target.value))}
              disabled={isRecording}
            >
              <option value="200">Very Fast (200ms)</option>
              <option value="500">Normal (500ms)</option>
              <option value="1000">Slow (1s)</option>
            </select>
          </div>

          <div className="recording-buttons">
            <button
              className={isRecording ? "stop-recording-button" : "start-recording-button"}
              onClick={toggleRecording}
              disabled={!isTracking}
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>

            <button
              className="capture-button"
              onClick={captureSample}
              disabled={!isTracking || !currentLandmarks}
            >
              Capture Single Frame
            </button>

            <button
              className="export-button"
              onClick={exportData}
              disabled={samplesCount === 0}
            >
              Export Data ({samplesCount} samples)
            </button>
          </div>
        </div>

        <div className="samples-counter">
          <h3>Samples Collected:</h3>
          <div className="samples-grid">
            {Object.entries(getGestureCounts()).map(([label, count]) => (
              <div
                key={label}
                className={`sample-item ${currentLabel === label ? 'active' : ''}`}
              >
                <span className="label">{formatGestureName(label)}</span>
                <span className="count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="instructions">
        <h3>Instructions:</h3>
        <ol>
          <li>Start tracking using the button above</li>
          <li>Select a gesture from the dropdown</li>
          <li>Position your hand in the camera view</li>
          <li>Press "Start Recording" to collect samples automatically</li>
          <li>Make the selected gesture while recording, moving your hand slightly to capture varied positions</li>
          <li>Stop recording and switch to a different gesture</li>
          <li>Repeat for all gestures you want to train</li>
          <li>Click "Export Data" to download the collected samples</li>
        </ol>
        <p><strong>Note:</strong> Try to collect at least 20-30 samples per gesture in different positions and orientations for better model training.</p>

        <div className="gesture-tips">
          <h4>Tips for specific gestures:</h4>
          <ul>
            <li><strong>Dynamic gestures</strong> (wave, swipe): Make the complete motion during recording</li>
            <li><strong>Static gestures</strong> (open_hand, point): Move hand position but maintain the gesture</li>
            <li><strong>OK sign</strong>: Form a circle with thumb and index finger</li>
            <li><strong>Pinch</strong>: Bring thumb and index finger together</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default GestureDataCollector;
