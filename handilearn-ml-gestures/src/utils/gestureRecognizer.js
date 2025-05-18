/**
 * GestureRecognizer - ML-based hand gesture recognition using TensorFlow.js
 * Optimized for TV-based detection at 4-5 feet distance
 */
import * as tf from '@tensorflow/tfjs';

class GestureRecognizer {
  constructor() {
    this.model = null;
    this.metadata = null;
    this.isLoaded = false;

    // Distance-optimized recognition parameters
    this.confidenceThreshold = 0.5;  // Lower threshold for distance detection
    this.smoothingWindow = [];
    this.windowSize = 7;  // Larger window for more temporal smoothing
    this.lastGesture = null;
    this.gestureHoldTime = 0;
    this.minHoldFrames = 5;  // Require more consistent frames at distance

    // Advanced filtering options
    this.useDoubleSmoothing = true;  // Apply secondary smoothing for distances
    this.gestureHistory = {};  // Track multiple gestures over time
    this.stabilityBonus = 0.1;  // Bonus for consistent gestures
  }

  /**
   * Normalize hand landmarks for model input with distance optimization
   * @param {Array} landmarks - Array of hand landmarks (21 points with x,y,z coordinates)
   * @returns {Array} - Normalized flattened landmarks
   */
  normalizeLandmarks(landmarks) {
    if (!landmarks || landmarks.length !== 21) {
      throw new Error('Invalid landmarks format: expected 21 landmarks');
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

    // Additional normalization for distance-specific noise
    // Apply subtle smoothing to reduce jitter in normalized space
    if (this.useDoubleSmoothing && this.lastNormalizedLandmarks) {
      const smoothingFactor = 0.15; // 15% from previous frame
      for (let i = 0; i < normalizedLandmarks.length; i++) {
        normalizedLandmarks[i] = (normalizedLandmarks[i] * (1 - smoothingFactor)) +
          (this.lastNormalizedLandmarks[i] * smoothingFactor);
      }
    }

    // Save for next frame smoothing
    this.lastNormalizedLandmarks = [...normalizedLandmarks];

    return normalizedLandmarks;
  }

  /**
   * Load a trained gesture model
   * @param {string} modelUrl - URL to the model file
   * @param {string} metadataUrl - URL to the metadata file
   * @returns {Promise<boolean>} - Resolves to true when model is loaded
   */
  async loadModel(modelUrl, metadataUrl) {
    try {
      // Load model
      console.log(`Loading gesture model from ${modelUrl}`);
      this.model = await tf.loadLayersModel(modelUrl);

      // Load metadata
      if (metadataUrl) {
        console.log(`Loading model metadata from ${metadataUrl}`);
        const response = await fetch(metadataUrl);
        this.metadata = await response.json();
      }

      this.isLoaded = true;
      console.log('Gesture model loaded successfully');

      if (this.metadata && this.metadata.gestures) {
        console.log(`Available gestures: ${this.metadata.gestures.join(', ')}`);
      }

      return true;
    } catch (error) {
      console.error('Error loading gesture model:', error);
      throw error;
    }
  }

  /**
   * Set model parameters - optimized for TV distance
   * @param {Object} params - Parameters object
   * @param {number} params.confidenceThreshold - Minimum confidence for detection (0-1)
   * @param {number} params.windowSize - Number of frames to smooth over
   * @param {number} params.minHoldFrames - Minimum frames to hold a gesture
   * @param {boolean} params.useDoubleSmoothing - Apply secondary smoothing
   * @param {number} params.stabilityBonus - Bonus for consistent gestures
   */
  configure(params = {}) {
    if (params.confidenceThreshold !== undefined) {
      this.confidenceThreshold = Math.max(0, Math.min(1, params.confidenceThreshold));
    }

    if (params.windowSize !== undefined) {
      this.windowSize = Math.max(1, params.windowSize);
      this.smoothingWindow = []; // Reset window
    }

    if (params.minHoldFrames !== undefined) {
      this.minHoldFrames = Math.max(1, params.minHoldFrames);
    }

    if (params.useDoubleSmoothing !== undefined) {
      this.useDoubleSmoothing = params.useDoubleSmoothing;
    }

    if (params.stabilityBonus !== undefined) {
      this.stabilityBonus = Math.max(0, Math.min(0.5, params.stabilityBonus));
    }
  }

  /**
   * Recognize gesture from hand landmarks with distance optimization
   * @param {Array} landmarks - MediaPipe hand landmarks
   * @returns {Object|null} - Recognized gesture or null
   */
  recognizeGesture(landmarks) {
    if (!this.isLoaded || !landmarks || landmarks.length !== 21) {
      return null;
    }

    try {
      // Normalize landmarks
      const normalizedLandmarks = this.normalizeLandmarks(landmarks);

      // Prepare input tensor
      const inputTensor = tf.tensor2d([normalizedLandmarks]);

      // Get prediction
      const prediction = this.model.predict(inputTensor);
      const scores = prediction.dataSync();

      // Get gesture mapping
      const gestures = this.metadata?.gestures ||
        Array.from({ length: scores.length }, (_, i) => `gesture_${i}`);

      // Find the best score and gesture
      let bestScore = 0;
      let bestGestureIndex = -1;

      for (let i = 0; i < scores.length; i++) {
        if (scores[i] > bestScore) {
          bestScore = scores[i];
          bestGestureIndex = i;
        }
      }

      // Create prediction object
      const rawPrediction = {
        gesture: gestures[bestGestureIndex],
        confidence: bestScore,
        allScores: Object.fromEntries(gestures.map((g, i) => [g, scores[i]]))
      };

      // Apply enhanced temporal smoothing for distance
      const smoothedPrediction = this.enhancedSmoothPrediction(rawPrediction);

      // Clean up tensors
      inputTensor.dispose();
      prediction.dispose();

      return smoothedPrediction;

    } catch (error) {
      console.error('Error recognizing gesture:', error);
      return null;
    }
  }

  /**
   * Apply enhanced temporal smoothing to predictions for distance-based detection
   * @param {Object} rawPrediction - Raw prediction from model
   * @returns {Object|null} - Smoothed prediction
   */
  enhancedSmoothPrediction(rawPrediction) {
    // Add current prediction to window
    this.smoothingWindow.push(rawPrediction);

    // Keep only the last N predictions
    if (this.smoothingWindow.length > this.windowSize) {
      this.smoothingWindow.shift();
    }

    // Count gesture occurrences and accumulate confidence
    const gestureCounts = {};
    const confidenceSum = {};

    this.smoothingWindow.forEach(pred => {
      gestureCounts[pred.gesture] = (gestureCounts[pred.gesture] || 0) + 1;
      confidenceSum[pred.gesture] = (confidenceSum[pred.gesture] || 0) + pred.confidence;
    });

    // Update gesture history
    Object.keys(gestureCounts).forEach(gesture => {
      if (!this.gestureHistory[gesture]) {
        this.gestureHistory[gesture] = {
          consecutiveFrames: 0,
          lastSeen: 0
        };
      }
    });

    // Find dominant gesture
    let maxCount = 0;
    let dominantGesture = null;
    let avgConfidence = 0;

    Object.entries(gestureCounts).forEach(([gesture, count]) => {
      // Calculate base confidence
      const baseConfidence = confidenceSum[gesture] / count;

      // Add stability bonus for consistent gestures
      let stabilityScore = 0;
      if (this.lastGesture?.gesture === gesture) {
        stabilityScore = this.stabilityBonus;
      }

      // Compute adjusted confidence
      const adjustedConfidence = baseConfidence + stabilityScore;

      // Update if this is the new best
      if (count > maxCount || (count === maxCount && adjustedConfidence > avgConfidence)) {
        maxCount = count;
        dominantGesture = gesture;
        avgConfidence = adjustedConfidence;
      }
    });

    // Update gesture history for all tracked gestures
    Object.keys(this.gestureHistory).forEach(gesture => {
      const history = this.gestureHistory[gesture];

      if (gesture === dominantGesture) {
        history.consecutiveFrames++;
        history.lastSeen = 0;
      } else {
        history.consecutiveFrames = 0;
        history.lastSeen++;
      }

      // Clean up history (remove gestures not seen in 60 frames)
      if (history.lastSeen > 60) {
        delete this.gestureHistory[gesture];
      }
    });

    // Check if it's a continuing gesture
    const isNewGesture = this.lastGesture?.gesture !== dominantGesture;

    if (isNewGesture) {
      this.gestureHoldTime = 1;
    } else {
      this.gestureHoldTime++;
    }

    // Only return if confidence is above threshold and held for minimum frames
    if (avgConfidence >= this.confidenceThreshold && this.gestureHoldTime >= this.minHoldFrames) {
      const result = {
        gesture: dominantGesture,
        confidence: avgConfidence,
        holdTime: this.gestureHoldTime,
        allScores: rawPrediction.allScores
      };

      this.lastGesture = result;
      return result;
    }

    return null;
  }

  /**
   * Test mode - get raw prediction without smoothing
   * @param {Array} landmarks - MediaPipe hand landmarks
   * @returns {Object|null} - Raw prediction or null
   */
  getRawPrediction(landmarks) {
    if (!this.isLoaded || !landmarks || landmarks.length !== 21) {
      return null;
    }

    try {
      // Normalize landmarks
      const normalizedLandmarks = this.normalizeLandmarks(landmarks);

      // Prepare input tensor
      const inputTensor = tf.tensor2d([normalizedLandmarks]);

      // Get prediction
      const prediction = this.model.predict(inputTensor);
      const scores = prediction.dataSync();

      // Get gesture mapping
      const gestures = this.metadata?.gestures ||
        Array.from({ length: scores.length }, (_, i) => `gesture_${i}`);

      // Create scores object
      const scoreObj = {};
      gestures.forEach((gesture, i) => {
        scoreObj[gesture] = scores[i];
      });

      // Clean up tensors
      inputTensor.dispose();
      prediction.dispose();

      return {
        scores: scoreObj,
        topScores: Object.entries(scoreObj)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([gesture, score]) => ({
            gesture,
            score
          }))
      };

    } catch (error) {
      console.error('Error getting raw prediction:', error);
      return null;
    }
  }

  /**
   * Reset the recognizer state
   * Useful when changing scenes or users
   */
  reset() {
    this.smoothingWindow = [];
    this.lastGesture = null;
    this.gestureHoldTime = 0;
    this.gestureHistory = {};
    this.lastNormalizedLandmarks = null;
  }
}

export default GestureRecognizer;
