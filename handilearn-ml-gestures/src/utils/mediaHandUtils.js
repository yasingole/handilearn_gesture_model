/**
 * Utility functions for handling camera and media streams
 */

/**
 * Sets up the camera and connects it to the provided video element
 * @param {HTMLVideoElement} videoElement - The video element to display the stream
 * @returns {Promise<MediaStream>} - The media stream if successful
 */
export const setupCamera = async (videoElement) => {
  if (!videoElement) {
    throw new Error('Video element is required')
  }

  try {
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      }
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints)
    videoElement.srcObject = stream

    return new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        videoElement.play()
        resolve(stream)
      }
    })
  } catch (error) {
    console.error('Error accessing camera:', error)
    throw error
  }
}

/**
 * Extract hand landmarks from a video frame
 * In a real implementation, this would use a library like MediaPipe Hands
 * @param {HTMLVideoElement} videoElement - The video element with camera stream
 * @returns {Object|null} - Hand landmarks data, or null if not detected
 */
export const extractHandLandmarks = (videoElement) => {
  if (!videoElement || !videoElement.srcObject) {
    console.error('No video stream available')
    return null
  }

  // This is a placeholder function
  // In a real implementation, you would:
  // 1. Capture the current frame from the video
  // 2. Process it with MediaPipe Hands or similar library
  // 3. Return landmarks data

  // For now, return mock data
  return {
    landmarks: [
      // These would be 3D coordinates of hand landmarks
      { x: 0.5, y: 0.5, z: 0 },
      // ... more landmarks
    ]
  }
}

/**
 * Stops all tracks in the provided media stream
 * @param {MediaStream} stream - The media stream to stop
 */
export const stopMediaStream = (stream) => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop())
  }
}
