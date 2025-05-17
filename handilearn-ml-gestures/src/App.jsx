import React, { useState } from 'react';
import GestureDataCollector from './components/GestureDataCollector';
import GestureModelTrainer from './components/GestureModelTrainer';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('collect');

  // Expanded gesture set
  const gestureLabels = [
    // Core Gestures
    'open_hand',
    'fist',
    'point',
    'wave',
    'pinch',

    // Additional Static Gestures
    'thumbs_up',
    'ok_sign',
    'victory',
    'rock_on', // Index and pinky extended
    'thumbs_down',

    // Counting Gestures
    'number_one',   // Already covered by point
    'number_two',   // Already covered by victory
    'number_three', // Index, middle, ring extended
    'number_four',  // All fingers except thumb extended
    'number_five',  // Already covered by open_hand

    // Special Gestures for HandiLearn
    'tickle',        // Wiggling fingers
    'rotate_cw',     // Clockwise rotation
    'rotate_ccw',    // Counter-clockwise rotation
    'swipe_left',
    'swipe_right',
    'swipe_up',
    'swipe_down'
  ];

  return (
    <div className="app">
      <header>
        <h1>HandiLearn ML Gesture Training</h1>
        <div className="tabs">
          <button
            className={activeTab === 'collect' ? 'active' : ''}
            onClick={() => setActiveTab('collect')}
          >
            Data Collection
          </button>
          <button
            className={activeTab === 'train' ? 'active' : ''}
            onClick={() => setActiveTab('train')}
          >
            Model Training
          </button>
        </div>
      </header>

      <main>
        {activeTab === 'collect' ? (
          <GestureDataCollector gestureLabels={gestureLabels} />
        ) : (
          <GestureModelTrainer />
        )}
      </main>

      <footer>
        <p>HandiLearn ML Gesture Training Tool - 2025</p>
      </footer>
    </div>
  );
}

export default App;
