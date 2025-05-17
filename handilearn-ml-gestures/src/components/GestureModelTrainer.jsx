import { useState } from 'react'
import { processGestureData } from '../utils/gestureDataProcessor'
import './GestureModelTrainer.css'

function GestureModelTrainer({ data, onModelTrained }) {
  const [isTraining, setIsTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [modelInfo, setModelInfo] = useState(null)

  const startTraining = async () => {
    if (data.length === 0) {
      alert('No gesture data available for training')
      return
    }

    setIsTraining(true)
    setProgress(0)

    try {
      // In a real implementation, we'd process data and train a model
      // Here we're simulating this process with timeouts

      // Process the data
      const processedData = processGestureData(data)

      // Simulate training progress
      const intervalId = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + 10
          if (newProgress >= 100) {
            clearInterval(intervalId)
            return 100
          }
          return newProgress
        })
      }, 500)

      // Simulate model training completion
      setTimeout(() => {
        const mockModel = {
          id: 'model-' + Date.now(),
          created: new Date().toISOString(),
          gestures: data.map(item => item.name).filter((name, i, arr) => arr.indexOf(name) === i),
          accuracy: 0.92,
          parameters: {
            epochs: 100,
            batchSize: 32,
            optimizer: 'adam'
          }
        }

        setModelInfo(mockModel)
        setIsTraining(false)
        onModelTrained(mockModel)
      }, 5500)

    } catch (error) {
      console.error('Error training model:', error)
      setIsTraining(false)
    }
  }

  return (
    <div className="gesture-trainer">
      <h2>Train Gesture Model</h2>

      <div className="data-summary">
        <h3>Available Training Data</h3>
        <p>Total recordings: <strong>{data.length}</strong></p>

        {data.length > 0 && (
          <div className="gesture-categories">
            <p>Gesture categories:</p>
            <ul>
              {[...new Set(data.map(item => item.name))].map(name => (
                <li key={name}>
                  {name}: {data.filter(d => d.name === name).length} samples
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="training-controls">
        <button
          className="train-btn"
          onClick={startTraining}
          disabled={isTraining || data.length === 0}
        >
          {isTraining ? 'Training...' : 'Train Model'}
        </button>

        {isTraining && (
          <div className="progress-container">
            <div className="progress-bar" style={{ width: `${progress}%` }} />
            <span>{progress}%</span>
          </div>
        )}
      </div>

      {modelInfo && (
        <div className="model-info">
          <h3>Trained Model</h3>
          <p>Model ID: <code>{modelInfo.id}</code></p>
          <p>Created: {new Date(modelInfo.created).toLocaleString()}</p>
          <p>Accuracy: {(modelInfo.accuracy * 100).toFixed(1)}%</p>
          <p>Recognized gestures: {modelInfo.gestures.join(', ')}</p>
        </div>
      )}
    </div>
  )
}

export default GestureModelTrainer
