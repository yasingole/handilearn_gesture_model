import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

// Initialize TensorFlow.js
import * as tf from '@tensorflow/tfjs';

// Set TensorFlow.js to use WebGL for better performance on M1 Macs
tf.setBackend('webgl');

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
