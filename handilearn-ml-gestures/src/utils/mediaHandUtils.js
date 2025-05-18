/**
 * MediaHandUtils - A class to handle MediaPipe Hands initialization and tracking
 * Optimized for TV-based gesture recognition at 4-5 feet distance
 */
import { Hands } from '@mediapipe/hands';

class MediaHandUtils {
  constructor() {
    this.hands = null;
    this.videoElement = null;
    this.isInitialized = false;
    this.isTracking = false;
    this.handUpdateCallbacks = [];
    this.lastResults = null;
    this.animationFrameId = null;
    this.preprocessCanvas = document.createElement('canvas');
    this.preprocessCtx = this.preprocessCanvas.getContext('2d');
    this.enhanceVideo = true; // Enable video enhancement by default
    this.enhancementLevel = 1.3; // Contrast enhancement factor
  }

  /**
   * Initialize MediaPipe Hands with distance-optimized settings
   * @returns {Promise<boolean>} - Promise that resolves when initialized
   */
  async initialize() {
    try {
      this.hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
      });

      // Configure MediaPipe with optimized settings for distance detection
      await this.hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,          // Maximum complexity for better distant detection
        minDetectionConfidence: 0.3, // Lower threshold for distant hands
        minTrackingConfidence: 0.3   // Lower for better tracking continuity
      });

      // Set up result handler
      this.hands.onResults(results => this.handleResults(results));

      this.isInitialized = true;
      console.log('MediaPipe Hands initialized with distance-optimized settings');
      return true;
    } catch (error) {
      console.error('Failed to initialize MediaPipe Hands:', error);
      throw error;
    }
  }

  /**
   * Set video element for tracking
   * @param {HTMLVideoElement} videoElement - The video element to use
   */
  setVideoElement(videoElement) {
    this.videoElement = videoElement;

    // Initialize preprocessing canvas with video dimensions
    if (videoElement.videoWidth && videoElement.videoHeight) {
      this.updatePreprocessingCanvas();
    } else {
      // Set dimensions once video metadata is loaded
      videoElement.addEventListener('loadedmetadata', () => {
        this.updatePreprocessingCanvas();
      });
    }
  }

  /**
   * Update preprocessing canvas dimensions
   */
  updatePreprocessingCanvas() {
    this.preprocessCanvas.width = this.videoElement.videoWidth;
    this.preprocessCanvas.height = this.videoElement.videoHeight;
  }

  /**
   * Configure video enhancement
   * @param {boolean} enable - Whether to enable video enhancement
   * @param {number} level - Enhancement level (1.0-2.0)
   */
  configureVideoEnhancement(enable = true, level = 1.3) {
    this.enhanceVideo = enable;
    this.enhancementLevel = Math.max(1, Math.min(2, level));
  }

  /**
   * Start hand tracking with distance optimization
   * @returns {Promise<boolean>} - Promise that resolves when tracking started
   */
  async startTracking() {
    if (!this.isInitialized) {
      throw new Error('MediaHandUtils not initialized');
    }

    if (this.isTracking) {
      return true; // Already tracking
    }

    try {
      if (!this.videoElement) {
        throw new Error('Video element not set. Call setVideoElement() first.');
      }

      // Start the camera feed
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Request high resolution for better distance detection
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          }
        });

        this.videoElement.srcObject = stream;
        this.videoElement.play();

        // Wait for video to be ready
        await new Promise(resolve => {
          this.videoElement.onloadedmetadata = () => {
            this.updatePreprocessingCanvas();
            resolve();
          };
          if (this.videoElement.readyState >= 2) {
            this.updatePreprocessingCanvas();
            resolve();
          }
        });

        // Create a function to process each frame
        const processFrame = async () => {
          if (!this.isTracking) return;

          try {
            // Process with enhancement if enabled
            if (this.enhanceVideo) {
              const enhancedFrame = this.preprocessVideoFrame();
              await this.hands.send({ image: enhancedFrame });
            } else {
              await this.hands.send({ image: this.videoElement });
            }
          } catch (e) {
            console.warn('Frame processing error:', e);
          }

          this.animationFrameId = requestAnimationFrame(processFrame);
        };

        this.isTracking = true;
        processFrame();

        console.log('Hand tracking started with distance optimization');
        return true;
      } else {
        throw new Error('getUserMedia not supported in this browser');
      }
    } catch (error) {
      console.error('Failed to start hand tracking:', error);
      this.isTracking = false;
      throw error;
    }
  }

  /**
   * Preprocess video frame for better hand detection at distance
   * @returns {HTMLCanvasElement} - Canvas with enhanced video frame
   */
  preprocessVideoFrame() {
    if (!this.videoElement || !this.preprocessCanvas) return this.videoElement;

    const ctx = this.preprocessCtx;
    const canvas = this.preprocessCanvas;

    // Draw current video frame to canvas
    ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);

    // Apply image enhancements for better distance detection
    try {
      // Get image data for pixel manipulation
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Apply contrast enhancement
      const factor = this.enhancementLevel;
      const midpoint = 128;

      for (let i = 0; i < data.length; i += 4) {
        // Enhance RGB channels (skip alpha)
        for (let j = 0; j < 3; j++) {
          const channel = data[i + j];
          // Apply contrast formula: new = (old - midpoint) * factor + midpoint
          data[i + j] = Math.max(0, Math.min(255,
            (channel - midpoint) * factor + midpoint));
        }
      }

      // Put enhanced image back
      ctx.putImageData(imageData, 0, 0);
    } catch (e) {
      console.warn('Video enhancement error:', e);
      // Fallback to unenhanced frame on error
      ctx.drawImage(this.videoElement, 0, 0);
    }

    return canvas;
  }

  /**
   * Stop hand tracking
   * @returns {Promise<boolean>} - Promise that resolves when tracking stopped
   */
  async stopTracking() {
    if (!this.isTracking) return true;

    try {
      this.isTracking = false;

      // Cancel animation frame
      if (this.animationFrameId) {
        cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
      }

      // Stop all video tracks
      if (this.videoElement && this.videoElement.srcObject) {
        const tracks = this.videoElement.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        this.videoElement.srcObject = null;
      }

      return true;
    } catch (error) {
      console.error('Error stopping hand tracking:', error);
      throw error;
    }
  }

  /**
   * Register callback for hand updates
   * @param {Function} callback - Function to call with hand data
   * @returns {Function} - Function to unregister the callback
   */
  onHandUpdate(callback) {
    this.handUpdateCallbacks.push(callback);
    return () => {
      this.handUpdateCallbacks = this.handUpdateCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Handle results from MediaPipe Hands
   * @param {Object} results - Results from MediaPipe Hands
   */
  handleResults(results) {
    // Store latest results
    this.lastResults = results;

    // Notify callbacks
    if (this.handUpdateCallbacks.length > 0) {
      this.handUpdateCallbacks.forEach(callback => callback(results));
    }
  }

  /**
   * Get the latest hand tracking results
   * @returns {Object|null} - The latest results or null if none available
   */
  getLatestResults() {
    return this.lastResults;
  }
}

export default MediaHandUtils;
