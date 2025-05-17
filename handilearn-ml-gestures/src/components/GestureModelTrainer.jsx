import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import './GestureModelTrainer.css';

function GestureModelTrainer() {
  const [dataFile, setDataFile] = useState(null);
  const [rawData, setRawData] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [splitData, setSplitData] = useState(null);
  const [model, setModel] = useState(null);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, processing, training, complete, error
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [modelParams, setModelParams] = useState({
    epochs: 30,
    batchSize: 16,
    learningRate: 0.001,
    validationSplit: 0.2,
    hiddenUnits: [64, 32]
  });
  const fileInputRef = useRef(null);
  const logContainerRef = useRef(null);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [trainingLogs]);

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setDataFile(file);
      const reader = new FileReader();

      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          setRawData(data);
          setTrainingLogs(prev => [...prev, `Loaded file: ${file.name}`]);
          processData(data);
        } catch (error) {
          setTrainingLogs(prev => [...prev, `Error parsing file: ${error.message}`]);
          setTrainingStatus('error');
        }
      };

      reader.readAsText(file);
    }
  };

  // Process the uploaded data
  const processData = (data) => {
    setTrainingStatus('processing');
    try {
      setTrainingLogs(prev => [...prev, 'Processing gesture data...']);

      if (!data || !data.gestures) {
        throw new Error('Invalid data format: missing gestures object');
      }

      // Check if there are enough samples
      const gestureCounts = {};
      let totalSamples = 0;

      Object.entries(data.gestures).forEach(([label, samples]) => {
        if (!Array.isArray(samples)) {
          throw new Error(`Invalid samples format for gesture: ${label}`);
        }

        gestureCounts[label] = samples.length;
        totalSamples += samples.length;

        if (samples.length < 5) {
          setTrainingLogs(prev => [...prev, `Warning: Only ${samples.length} samples for "${label}". Consider collecting more.`]);
        }
      });

      if (totalSamples < 20) {
        setTrainingLogs(prev => [...prev, `Warning: Only ${totalSamples} total samples. Model accuracy may be poor.`]);
      }

      setTrainingLogs(prev => [...prev, `Found ${totalSamples} samples across ${Object.keys(gestureCounts).length} gestures.`]);

      // Extract features and labels
      const gestures = Object.keys(data.gestures);
      const gestureToIndex = {};
      gestures.forEach((gesture, index) => {
        gestureToIndex[gesture] = index;
      });

      const features = [];
      const labels = [];

      gestures.forEach(gesture => {
        const samples = data.gestures[gesture];
        samples.forEach(sample => {
          if (!sample.landmarks || !Array.isArray(sample.landmarks)) {
            setTrainingLogs(prev => [...prev, `Warning: Skipping invalid sample for "${gesture}"`]);
            return;
          }

          // Normalize the landmarks
          const normalizedLandmarks = normalizeHandLandmarks(sample.landmarks);

          features.push(normalizedLandmarks);
          labels.push(gestureToIndex[gesture]);
        });
      });

      // Create processed data object
      const processed = {
        features,
        labels,
        labelMapping: gestureToIndex,
        gestures
      };

      setProcessedData(processed);

      setTrainingLogs(prev => [
        ...prev,
        `Processed ${processed.features.length} samples with ${processed.gestures.length} gesture classes`
      ]);

      // Split into train/test sets
      const split = splitTrainTest(processed, modelParams.validationSplit);
      setSplitData(split);

      setTrainingLogs(prev => [
        ...prev,
        `Split into ${split.train.features.length} training and ${split.test.features.length} testing samples`
      ]);

      setTrainingStatus('ready');
    } catch (error) {
      console.error('Error processing data:', error);
      setTrainingLogs(prev => [...prev, `Error processing data: ${error.message}`]);
      setTrainingStatus('error');
    }
  };

  // Create and train the model
  const trainModel = async () => {
    if (!splitData) return;

    setTrainingStatus('training');
    setTrainingLogs(prev => [...prev, 'Creating model...']);

    try {
      // Convert data to tensors
      const numClasses = Object.keys(splitData.labelMapping).length;
      const inputSize = splitData.train.features[0].length; // Flattened landmarks size

      // Create tensor datasets
      const trainFeatures = tf.tensor2d(splitData.train.features);
      const trainLabels = tf.oneHot(tf.tensor1d(splitData.train.labels, 'int32'), numClasses);

      const testFeatures = tf.tensor2d(splitData.test.features);
      const testLabels = tf.oneHot(tf.tensor1d(splitData.test.labels, 'int32'), numClasses);

      // Log model configuration
      setTrainingLogs(prev => [
        ...prev,
        `Creating neural network with input size ${inputSize}, ${modelParams.hiddenUnits.join('→')} hidden units, and ${numClasses} output classes`,
        `Training config: ${modelParams.epochs} epochs, batch size ${modelParams.batchSize}, learning rate ${modelParams.learningRate}`
      ]);

      // Create a model optimized for M1 Mac
      const model = tf.sequential();

      // Input layer
      model.add(tf.layers.dense({
        inputShape: [inputSize],
        units: modelParams.hiddenUnits[0],
        activation: 'relu',
        kernelInitializer: 'heNormal'
      }));

      // Add dropout to reduce overfitting
      model.add(tf.layers.dropout({ rate: 0.25 }));

      // Add hidden layers
      for (let i = 1; i < modelParams.hiddenUnits.length; i++) {
        model.add(tf.layers.dense({
          units: modelParams.hiddenUnits[i],
          activation: 'relu',
          kernelInitializer: 'heNormal'
        }));

        model.add(tf.layers.dropout({ rate: 0.2 }));
      }

      // Output layer (softmax for multi-class classification)
      model.add(tf.layers.dense({
        units: numClasses,
        activation: 'softmax'
      }));

      // Compile the model with Adam optimizer
      model.compile({
        optimizer: tf.train.adam(modelParams.learningRate),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      // Show model summary
      model.summary();
      setTrainingLogs(prev => [...prev, 'Model created and compiled. Starting training...']);

      // Train the model
      await model.fit(trainFeatures, trainLabels, {
        epochs: modelParams.epochs,
        batchSize: modelParams.batchSize,
        validationData: [testFeatures, testLabels],
        shuffle: true,
        callbacks: {
          onEpochBegin: (epoch) => {
            setTrainingLogs(prev => [...prev, `Starting epoch ${epoch + 1}/${modelParams.epochs}`]);
          },
          onEpochEnd: (epoch, logs) => {
            setTrainingLogs(prev => [
              ...prev,
              `Epoch ${epoch + 1}/${modelParams.epochs}: loss=${logs.loss.toFixed(4)}, accuracy=${(logs.acc * 100).toFixed(2)}%, val_loss=${logs.val_loss.toFixed(4)}, val_acc=${(logs.val_acc * 100).toFixed(2)}%`
            ]);
          }
        }
      });

      setTrainingLogs(prev => [...prev, 'Training complete. Evaluating model...']);

      // Evaluate the model on test data
      const evalResult = await model.evaluate(testFeatures, testLabels);

      const evalLoss = evalResult[0].dataSync()[0];
      const evalAcc = evalResult[1].dataSync()[0];

      // Calculate per-class accuracy
      const predictions = model.predict(testFeatures);
      const predLabels = predictions.argMax(1).dataSync();
      const trueLabels = splitData.test.labels;

      const confusionMatrix = {};
      const labelAccuracy = {};

      // Initialize confusion matrix
      for (let i = 0; i < numClasses; i++) {
        confusionMatrix[i] = {};
        for (let j = 0; j < numClasses; j++) {
          confusionMatrix[i][j] = 0;
        }
      }

      // Fill confusion matrix
      for (let i = 0; i < predLabels.length; i++) {
        const predicted = predLabels[i];
        const actual = trueLabels[i];
        confusionMatrix[actual][predicted] = (confusionMatrix[actual][predicted] || 0) + 1;
      }

      // Calculate per-class accuracy
      Object.keys(confusionMatrix).forEach(actual => {
        const total = Object.values(confusionMatrix[actual]).reduce((a, b) => a + b, 0);
        const correct = confusionMatrix[actual][actual] || 0;
        labelAccuracy[actual] = correct / total;
      });

      // Store evaluation results
      setEvaluationResults({
        loss: evalLoss,
        accuracy: evalAcc,
        perClassAccuracy: labelAccuracy,
        confusionMatrix
      });

      // Log detailed evaluation
      setTrainingLogs(prev => [
        ...prev,
        `Final evaluation: loss=${evalLoss.toFixed(4)}, accuracy=${(evalAcc * 100).toFixed(2)}%`,
        'Per-class accuracy:'
      ]);

      // Log per-class accuracy
      Object.entries(splitData.labelMapping).forEach(([gesture, index]) => {
        const accuracy = labelAccuracy[index] || 0;
        setTrainingLogs(prev => [
          ...prev,
          `- ${gesture}: ${(accuracy * 100).toFixed(2)}%`
        ]);
      });

      // Save the model reference
      setModel(model);
      setTrainingStatus('complete');

      // Clean up tensors
      trainFeatures.dispose();
      trainLabels.dispose();
      testFeatures.dispose();
      testLabels.dispose();
      predictions.dispose();

    } catch (error) {
      console.error('Error training model:', error);
      setTrainingLogs(prev => [...prev, `Error training model: ${error.message}`]);
      setTrainingStatus('error');
    }
  };

  // Handle parameter changes
  const handleParamChange = (param, value) => {
    setModelParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  // Save the trained model
  const saveModel = async () => {
    if (!model) return;

    try {
      setTrainingLogs(prev => [...prev, 'Saving model...']);

      // Create a metadata file with gesture labels
      const metadata = {
        gestures: processedData.gestures,
        labelMapping: processedData.labelMapping,
        inputSize: processedData.features[0].length,
        modelType: 'handilearn-gesture-recognizer',
        version: '1.0.0',
        date: new Date().toISOString()
      };

      // Save metadata
      const metadataStr = JSON.stringify(metadata, null, 2);
      const metadataBlob = new Blob([metadataStr], {type: 'application/json'});
      const metadataUrl = URL.createObjectURL(metadataBlob);
      const metadataLink = document.createElement('a');
      metadataLink.href = metadataUrl;
      metadataLink.download = 'handilearn-gesture-model-metadata.json';
      document.body.appendChild(metadataLink);

      // Save the model using TensorFlow.js built-in functions
      // This will automatically create model.json and weight files
      await model.save('downloads://handilearn-gesture-model');

      // Wait a bit to allow model to save first
      setTimeout(() => {
        metadataLink.click();
        document.body.removeChild(metadataLink);
        URL.revokeObjectURL(metadataUrl);
      }, 1000);

      setTrainingLogs(prev => [...prev, 'Model and metadata saved successfully!']);
    } catch (error) {
      console.error('Error saving model:', error);
      setTrainingLogs(prev => [...prev, `Error saving model: ${error.message}`]);
    }
  };

  // Try inference on a test sample
  const testModel = async () => {
    if (!model || !splitData || !splitData.test.features.length) return;

    try {
      // Get a random test sample
      const randomIndex = Math.floor(Math.random() * splitData.test.features.length);
      const testSample = splitData.test.features[randomIndex];
      const trueLabel = splitData.test.labels[randomIndex];

      // Convert to tensor
      const inputTensor = tf.tensor2d([testSample]);

      // Make prediction
      const prediction = await model.predict(inputTensor);

      // Get predicted class
      const predictedClass = prediction.argMax(1).dataSync()[0];

      // Get label names
      const labelNames = Object.entries(splitData.labelMapping)
        .reduce((acc, [label, index]) => {
          acc[index] = label;
          return acc;
        }, {});

      // Get confidence scores
      const scores = prediction.dataSync();

      // Log the top 3 predictions
      const topPredictions = [];
      for (let i = 0; i < scores.length; i++) {
        topPredictions.push({
          label: labelNames[i],
          score: scores[i]
        });
      }

      // Sort by score descending
      topPredictions.sort((a, b) => b.score - a.score);

      // Log the result
      setTrainingLogs(prev => [
        ...prev,
        `Test inference: True gesture: "${labelNames[trueLabel]}", Predicted: "${labelNames[predictedClass]}" with ${(scores[predictedClass] * 100).toFixed(2)}% confidence`,
        'Top 3 predictions:'
      ]);

      // Log top 3 predictions
      topPredictions.slice(0, 3).forEach((pred, idx) => {
        setTrainingLogs(prev => [
          ...prev,
          `${idx + 1}. ${pred.label}: ${(pred.score * 100).toFixed(2)}%`
        ]);
      });

      // Clean up
      inputTensor.dispose();
      prediction.dispose();

    } catch (error) {
      console.error('Error testing model:', error);
      setTrainingLogs(prev => [...prev, `Error testing model: ${error.message}`]);
    }
  };

  // Format gesture name for display
  const formatGestureName = (name) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <div className="gesture-model-trainer">
      <h2>Gesture Model Trainer</h2>

      <div className="trainer-container">
        <div className="trainer-panel">
          <div className="upload-section">
            <h3>1. Upload Gesture Data</h3>
            <input
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              ref={fileInputRef}
              style={{ display: 'none' }}
            />
            <button
              className="upload-button"
              onClick={() => fileInputRef.current.click()}
              disabled={trainingStatus === 'processing' || trainingStatus === 'training'}
            >
              Choose Gesture Data File
            </button>
            {dataFile && (
              <div className="file-info">
                Loaded: <strong>{dataFile.name}</strong> ({Math.round(dataFile.size / 1024)} KB)
              </div>
            )}

            {rawData && rawData.gestures && (
              <div className="data-summary">
                <h4>Dataset Summary:</h4>
                <div className="gesture-list">
                  {Object.entries(rawData.gestures).map(([label, samples]) => (
                    <div key={label} className="gesture-item">
                      <span className="label">{formatGestureName(label)}</span>
                      <span className="count">{samples.length}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="model-params">
            <h3>2. Configure Model</h3>
            <div className="param-group">
              <label htmlFor="epochs">Epochs:</label>
              <input
                type="number"
                id="epochs"
                value={modelParams.epochs}
                onChange={(e) => handleParamChange('epochs', parseInt(e.target.value))}
                min="5"
                max="100"
                disabled={trainingStatus === 'training'}
              />
              <span className="param-help">Number of training cycles (higher = better accuracy but slower)</span>
            </div>

            <div className="param-group">
              <label htmlFor="batchSize">Batch Size:</label>
              <input
                type="number"
                id="batchSize"
                value={modelParams.batchSize}
                onChange={(e) => handleParamChange('batchSize', parseInt(e.target.value))}
                min="1"
                max="64"
                disabled={trainingStatus === 'training'}
              />
              <span className="param-help">Samples processed at once (lower for limited memory)</span>
            </div>

            <div className="param-group">
              <label htmlFor="learningRate">Learning Rate:</label>
              <select
                id="learningRate"
                value={modelParams.learningRate}
                onChange={(e) => handleParamChange('learningRate', parseFloat(e.target.value))}
                disabled={trainingStatus === 'training'}
              >
                <option value="0.01">0.01 (Fast)</option>
                <option value="0.001">0.001 (Default)</option>
                <option value="0.0001">0.0001 (Slow)</option>
              </select>
              <span className="param-help">How quickly the model adapts</span>
            </div>

            <div className="param-group">
              <label htmlFor="validationSplit">Validation Split:</label>
              <select
                id="validationSplit"
                value={modelParams.validationSplit}
                onChange={(e) => handleParamChange('validationSplit', parseFloat(e.target.value))}
                disabled={trainingStatus === 'training' || trainingStatus === 'processing'}
              >
                <option value="0.1">10%</option>
                <option value="0.2">20%</option>
                <option value="0.3">30%</option>
              </select>
              <span className="param-help">Percentage of data used for validation</span>
            </div>

            <div className="param-group">
              <label htmlFor="modelSize">Model Size:</label>
              <select
                id="modelSize"
                value={modelParams.hiddenUnits.join(',')}
                onChange={(e) => handleParamChange('hiddenUnits', e.target.value.split(',').map(Number))}
                disabled={trainingStatus === 'training'}
              >
                <option value="32,16">Small (32→16 units)</option>
                <option value="64,32">Medium (64→32 units)</option>
                <option value="128,64">Large (128→64 units)</option>
              </select>
              <span className="param-help">Model complexity (smaller = faster, larger = more accurate)</span>
            </div>
          </div>

          <div className="train-section">
            <h3>3. Train Model</h3>
            <button
              className="train-button"
              onClick={trainModel}
              disabled={trainingStatus !== 'ready' || !splitData}
            >
              {trainingStatus === 'training' ? 'Training...' : 'Train Model'}
            </button>

            {evaluationResults && (
              <div className="evaluation-results">
                <h4>Evaluation Results:</h4>
                <div className="result-item">
                  <span>Accuracy:</span>
                  <span className="accuracy">{(evaluationResults.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div className="result-item">
                  <span>Loss:</span>
                  <span>{evaluationResults.loss.toFixed(4)}</span>
                </div>
              </div>
            )}

            {trainingStatus === 'complete' && (
              <div className="model-actions">
                <button
                  className="save-button"
                  onClick={saveModel}
                >
                  Save Model
                </button>
                <button
                  className="test-button"
                  onClick={testModel}
                >
                  Test Inference
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="log-section">
          <h3>Training Log</h3>
          <div className="log-container" ref={logContainerRef}>
            {trainingLogs.length === 0 ? (
              <div className="log-empty">
                No logs yet. Upload data and start training to see progress.
              </div>
            ) : (
              trainingLogs.map((log, index) => (
                <div key={index} className="log-entry">
                  {log}
                </div>
              ))
            )}
            {trainingStatus === 'training' && (
              <div className="log-entry training-indicator">
                Training in progress...
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="instructions">
        <h3>Instructions:</h3>
        <ol>
          <li>Upload the gesture data JSON file created with the Gesture Data Collector</li>
          <li>Configure model parameters (defaults are optimized for M1 Macs with 8GB RAM)</li>
          <li>Click "Train Model" to train a neural network</li>
          <li>Monitor the training progress in the log section</li>
          <li>Once training is complete, you can test the model on random samples</li>
          <li>Save the model files to use in your HandiLearn application</li>
        </ol>
        <p><strong>Note:</strong> Training may take several minutes depending on your device and dataset size.</p>
      </div>
    </div>
  );
}

// Normalize hand landmarks to be position and scale invariant
function normalizeHandLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 63) {
    throw new Error('Invalid landmarks format. Expected 63 values (21 landmarks × 3 coordinates)');
  }

  // Reshape into [x,y,z] format for easier processing
  const reshapedLandmarks = [];
  for (let i = 0; i < landmarks.length; i += 3) {
    reshapedLandmarks.push({
      x: landmarks[i],
      y: landmarks[i + 1],
      z: landmarks[i + 2]
    });
  }

  // Get wrist coordinates (first landmark)
  const wrist = reshapedLandmarks[0];

  // Subtract wrist position from all landmarks to center at origin
  const centeredLandmarks = reshapedLandmarks.map(landmark => ({
    x: landmark.x - wrist.x,
    y: landmark.y - wrist.y,
    z: landmark.z - wrist.z
  }));

  // Calculate scale factor based on distance from wrist to middle finger tip
  const middleFingerTip = centeredLandmarks[12]; // Index 12 corresponds to middle finger tip
  const distanceToMiddleTip = Math.sqrt(
    middleFingerTip.x * middleFingerTip.x +
    middleFingerTip.y * middleFingerTip.y +
    middleFingerTip.z * middleFingerTip.z
  );

  // Scale factor to make distance to middle finger tip = 1
  const scaleFactor = distanceToMiddleTip > 0 ? 1 / distanceToMiddleTip : 1;

  // Scale all landmarks
  const normalizedLandmarks = centeredLandmarks.map(landmark => ({
    x: landmark.x * scaleFactor,
    y: landmark.y * scaleFactor,
    z: landmark.z * scaleFactor
  }));

  // Flatten back to 1D array
  const flattenedNormalized = normalizedLandmarks.flatMap(landmark => [
    landmark.x, landmark.y, landmark.z
  ]);

  return flattenedNormalized;
}

// Split dataset into training and testing sets
function splitTrainTest(processedData, testRatio = 0.2) {
  const { features, labels, labelMapping, gestures } = processedData;

  if (!features || !labels || features.length !== labels.length) {
    throw new Error('Invalid processed data format');
  }

  // Create index array and shuffle it
  const indices = Array.from({ length: features.length }, (_, i) => i);
  shuffleArray(indices);

  // Calculate split point
  const splitPoint = Math.floor(features.length * (1 - testRatio));

  // Split into train and test sets
  const trainIndices = indices.slice(0, splitPoint);
  const testIndices = indices.slice(splitPoint);

  // Create train and test sets
  const trainFeatures = trainIndices.map(i => features[i]);
  const trainLabels = trainIndices.map(i => labels[i]);
  const testFeatures = testIndices.map(i => features[i]);
  const testLabels = testIndices.map(i => labels[i]);

  return {
    train: {
      features: trainFeatures,
      labels: trainLabels
    },
    test: {
      features: testFeatures,
      labels: testLabels
    },
    labelMapping,
    gestures
  };
}

// Shuffle an array in-place
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

export default GestureModelTrainer;
