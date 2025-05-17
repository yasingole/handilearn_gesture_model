import { useState, useRef, useEffect } from 'react'
import { setupCamera } from '../utils/mediaHandUtils'
import './GestureDataCollector.css'

function GestureDataCollector({ onDataCollected }) {
  const videoRef = useRef(null)
  const [isRecording, setIsRecording] = useState(false)
  const [gestureName, setGestureName] = useState('')
  const [recordings, setRecordings] = useState([])

  useEffect(() => {
    const setupVideoStream = async () => {
      try {
        const stream = await setupCamera(videoRef.current)
        if (stream) {
          console.log('Camera setup successful')
        }
      } catch (error) {
        console.error('Error setting up camera:', error)
      }
    }

    setupVideoStream()

    return () => {
      // Cleanup video stream when component unmounts
      const stream = videoRef.current?.srcObject
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  const startRecording = () => {
    if (!gestureName.trim()) {
      alert('Please enter a gesture name')
      return
    }

    setIsRecording(true)
    // Recording logic would go here
    console.log(`Started recording gesture: ${gestureName}`)
  }

  const stopRecording = () => {
    setIsRecording(false)

    // This would be where we capture the actual gesture data
    // For this template, we're just simulating captured data
    const newRecording = {
      id: Date.now(),
      name: gestureName,
      frames: [], // This would contain actual hand position data
      timestamp: new Date().toISOString()
    }

    const updatedRecordings = [...recordings, newRecording]
    setRecordings(updatedRecordings)
    onDataCollected(updatedRecordings)

    console.log(`Stopped recording gesture: ${gestureName}`)
    setGestureName('')
  }

  return (
    <div className="gesture-collector">
      <h2>Collect Gesture Data</h2>

      <div className="video-container">
        <video ref={videoRef} autoPlay playsInline className="video-preview" />
      </div>

      <div className="controls">
        <input
          type="text"
          value={gestureName}
          onChange={(e) => setGestureName(e.target.value)}
          placeholder="Enter gesture name"
          disabled={isRecording}
        />

        {!isRecording ? (
          <button className="record-btn" onClick={startRecording}>
            Start Recording
          </button>
        ) : (
          <button className="stop-btn" onClick={stopRecording}>
            Stop Recording
          </button>
        )}
      </div>

      <div className="recordings-list">
        <h3>Recorded Gestures ({recordings.length})</h3>
        {recordings.length > 0 ? (
          <ul>
            {recordings.map(rec => (
              <li key={rec.id}>
                {rec.name} - {new Date(rec.timestamp).toLocaleTimeString()}
              </li>
            ))}
          </ul>
        ) : (
          <p>No gestures recorded yet</p>
        )}
      </div>
    </div>
  )
}

export default GestureDataCollector
