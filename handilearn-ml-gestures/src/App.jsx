import { useState } from 'react'
import GestureDataCollector from './components/GestureDataCollector'
import GestureModelTrainer from './components/GestureModelTrainer'
import GesturePreview from './components/GesturePreview'
import './App.css'

function App() {
  const [gestureData, setGestureData] = useState([])
  const [trainedModel, setTrainedModel] = useState(null)

  return (
    <div className="app-container">
      <h1>HandiLearn Gesture Model</h1>
      <div className="content">
        <GestureDataCollector onDataCollected={setGestureData} />
        <GestureModelTrainer data={gestureData} onModelTrained={setTrainedModel} />
        <GesturePreview model={trainedModel} />
      </div>
    </div>
  )
}

export default App
