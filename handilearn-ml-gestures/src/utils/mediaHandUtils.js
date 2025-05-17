/**
 * MediaHandUtils - A class to handle MediaPipe Hands initialization and tracking
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
  }

  /**
   * Initialize MediaPipe Hands
   * @returns {Promise<boolean>} - Promise that resolves when initialized
   */
  async initialize() {
    try {
      this.hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
      });

      // Configure MediaPipe
      await this.hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      // Set up result handler
      this.hands.onResults(results => this.handleResults(results));

      this.isInitialized = true;
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
  }

  /**
   * Start hand tracking
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
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' }
        });

        this.videoElement.srcObject = stream;
        this.videoElement.play();

        // Wait for video to be ready
        await new Promise(resolve => {
          this.videoElement.onloadedmetadata = () => resolve();
          if (this.videoElement.readyState >= 2) resolve();
        });

        // Create a function to process each frame
        const processFrame = async () => {
          if (!this.isTracking) return;

          await this.hands.send({ image: this.videoElement });
          this.animationFrameId = requestAnimationFrame(processFrame);
        };

        this.isTracking = true;
        processFrame();

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
