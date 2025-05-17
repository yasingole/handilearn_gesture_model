import { useState, useRef, useEffect } from 'react'
import { setupCamera } from '../utils/mediaHandUtils'
import { recognizeGesture } from '../utils/gestureRecognizer'

function GesturePreview({ model }) {
  const videoRef = useRef(null)
  const [isActive, setIsActive] = useState(false)
  const [detectedGesture, setDetectedGesture] = useState(null)
  const [confidence, setConfidence] = useState(0)
  const recognitionInterval = useRef(null)

  useEffect(() => {
    // Clean up any existing recognition process when component unmounts
    // or when model changes
    return () => {
      if (recognitionInterval.current) {
        clearInterval(recognitionInterval.current)
      }

      const stream = videoRef.current?.srcObject
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [model])

  const startPreview = async () => {
    if (!model) {
      alert('No trained model available. Please train a model first.')
      return
    }

    try {
      const stream = await setupCamera(videoRef.current)
      if (stream) {
        setIsActive(true)

        // Start gesture recognition at regular intervals
        recognitionInterval.current = setInterval(() => {
          // In a real app, we would capture hand landmarks from the video
          // and pass them to the recognition function
          const mockHandData = { /* This would be real hand landmark data */ }

          // Simulate gesture recognition
          const result = recognizeGesture(mockHandData, model)
          setDetectedGesture(result.gesture)
          setConfidence(result.confidence)
        }, 500)
      }
    } catch (error) {
      console.error('Error setting up camera for preview:', error)
    }
  }

  const stopPreview = () => {
    setIsActive(false)

    if (recognitionInterval.current) {
      clearInterval(recognitionInterval.current)
      recognitionInterval.current = null
    }

    const stream = videoRef.current?.srcObject
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
    }

    setDetectedGesture(null)
    setConfidence(0)
  }

  return (
    <div className="gesture-preview">
      <h2>Gesture Recognition Preview</h2>

      {!model ? (
        <div className="no-model-warning">
          <p>No trained model available. Train a model to see the preview.</p>
        </div>
      ) : (
        <>
          <div className="video-container">
            <video ref={videoRef} autoPlay playsInline className="video-preview" />

            {isActive && detectedGesture && (
              <div className="gesture-overlay">
                <div className="gesture-name">{detectedGesture}</div>
                <div className="confidence-meter">
                  <div className="confidence-bar" style={{ width: `${confidence * 100}%` }}></div>
                  <span>{Math.round(confidence * 100)}%</span>
                </div>
              </div>
            )}
          </div>

          <div className="controls">
            {!isActive ? (
              <button onClick={startPreview} className="start-btn">Start Preview</button>
            ) : (
              <button onClick={stopPreview} className="stop-btn">Stop Preview</button>
            )}
          </div>

          {isActive && (
            <div className="instructions">
              <p>Make gestures in front of the camera to test the model.</p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default GesturePreview
