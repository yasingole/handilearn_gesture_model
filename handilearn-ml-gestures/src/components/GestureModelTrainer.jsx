import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import GestureDataProcessor from '../utils/gestureDataProcessor';
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
  const [datasetAnalysis, setDatasetAnalysis] = useState(null);
  const [modelParams, setModelParams] = useState({
    epochs: 50,             // Increased for distance-based gestures
    batchSize: 16,
    learningRate: 0.001,
    validationSplit: 0.2,
    hiddenUnits: [64, 32],
    useDistanceAugmentation: true,  // Enable by default for TV-based detection
    augmentationFactor: 2,
    dropoutRate: 0.3        // Increased to help with distance generalization
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
      setTrainingLogs(prev => [...prev, 'Processing gesture data for TV-based detection...']);

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

        // For TV detection, we need more samples
        if (samples.length < 20) {
          setTrainingLogs(prev => [...prev, `Warning: Only ${samples.length} samples for "${label}". For TV-based detection, aim for at least 50 samples.`]);
        } else if (samples.length < 50) {
          setTrainingLogs(prev => [...prev, `Note: ${samples.length} samples for "${label}" may be insufficient for reliable TV-based detection. Consider collecting more.`]);
        }
      });

      if (totalSamples < 50) {
        setTrainingLogs(prev => [...prev, `Warning: Only ${totalSamples} total samples. For reliable TV-based detection, you should have at least 200-300 total samples.`]);
      }

      setTrainingLogs(prev => [...prev, `Found ${totalSamples} samples across ${Object.keys(gestureCounts).length} gestures.`]);

      // Process the dataset - using distance-optimized normalization
      const processed = GestureDataProcessor.processDataset(data);
      setProcessedData(processed);

      // Analyze the dataset for distance suitability
      const analysis = GestureDataProcessor.analyzeDataset(processed);
      setDatasetAnalysis(analysis);

      setTrainingLogs(prev => [
        ...prev,
        `Processed ${processed.features.length} samples with ${processed.gestures.length} gesture classes`,
        `Distance readiness score: ${analysis.distanceReadiness.readinessScore}/10 (${analysis.distanceReadiness.overallReadiness})`,
        `Recommendation: ${analysis.distanceReadiness.recommendation}`
      ]);

      // Apply distance augmentation if enabled
      let augmentedData = processed;
      if (modelParams.useDistanceAugmentation) {
        augmentedData = GestureDataProcessor.distanceAugmentDataset(
          processed,
          modelParams.augmentationFactor
        );

        const additionalSamples = augmentedData.features.length - processed.features.length;
        setTrainingLogs(prev => [
          ...prev,
          `Applied distance augmentation: Generated ${additionalSamples} additional samples to improve distance robustness`
        ]);
      }

      // Split into train/test sets
      const split = GestureDataProcessor.splitTrainTest(augmentedData, modelParams.validationSplit);
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
    setTrainingLogs(prev => [...prev, 'Creating model optimized for TV-based distance detection...']);

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
        `Distance optimization: ${modelParams.useDistanceAugmentation ? 'Enabled' : 'Disabled'}, Dropout: ${modelParams.dropoutRate}`,
        `Training config: ${modelParams.epochs} epochs, batch size ${modelParams.batchSize}, learning rate ${modelParams.learningRate}`
      ]);

      // Create a model optimized for distance detection
      const model = tf.sequential();

      // Input layer with extra regularization for distance robustness
      model.add(tf.layers.dense({
        inputShape: [inputSize],
        units: modelParams.hiddenUnits[0],
        activation: 'relu',
        kernelInitializer: 'heNormal',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) // Add L2 regularization
      }));

      // Add dropout to reduce overfitting
      model.add(tf.layers.dropout({ rate: modelParams.dropoutRate }));

      // Add hidden layers
      for (let i = 1; i < modelParams.hiddenUnits.length; i++) {
        model.add(tf.layers.dense({
          units: modelParams.hiddenUnits[i],
          activation: 'relu',
          kernelInitializer: 'heNormal',
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) // Add L2 regularization
        }));

        model.add(tf.layers.dropout({ rate: modelParams.dropoutRate * 0.8 })); // Gradually reduce dropout
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
      setTrainingLogs(prev => [...prev, 'Model created and compiled with distance optimizations. Starting training...']);

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

      setTrainingLogs(prev => [...prev, 'Training complete. Evaluating model for distance performance...']);

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
      const labelToGesture = {};
      Object.entries(splitData.labelMapping).forEach(([gesture, index]) => {
        labelToGesture[index] = gesture;
      });

      Object.entries(labelAccuracy).forEach(([labelIndex, accuracy]) => {
        const gesture = labelToGesture[labelIndex];
        setTrainingLogs(prev => [
          ...prev,
          `- ${gesture}: ${(accuracy * 100).toFixed(2)}%`
        ]);
      });

      // Distance-specific evaluation
      setTrainingLogs(prev => [
        ...prev,
        'TV-Based Distance Performance Assessment:',
        `- Overall accuracy ${(evalAcc * 100).toFixed(2)}% - ${evalAcc > 0.85 ? 'Good' : (evalAcc > 0.75 ? 'Adequate' : 'Needs improvement')} for distance detection`,
        `- ${Object.values(labelAccuracy).filter(acc => acc > 0.8).length} out of ${numClasses} gestures have >80% accuracy`
      ]);

      // Saving recommendations for TV usage
      setTrainingLogs(prev => [
        ...prev,
        'Distance Usage Recommendations:',
        '- For optimal performance, position webcam at eye level',
        '- Ensure good front lighting when using gestures',
        '- For deployment, set GestureRecognizer confidence threshold to 0.5',
        '- Use at least 4-5 frame temporal smoothing (windowSize: 5) in GestureRecognizer'
      ]);

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

  // Toggle the distance augmentation
  const handleToggleAugmentation = () => {
    setModelParams(prev => ({
      ...prev,
      useDistanceAugmentation: !prev.useDistanceAugmentation
    }));

    // Re-process the data if it exists
    if (rawData) {
      processData(rawData);
    }
  };

  // Save the trained model
  const saveModel = async () => {
    if (!model) return;

    try {
      setTrainingLogs(prev => [...prev, 'Saving model optimized for TV-based detection...']);

      // Create a metadata file with gesture labels and TV-specific settings
      const metadata = {
        gestures: processedData.gestures,
        labelMapping: processedData.labelMapping,
        inputSize: processedData.features[0].length,
        modelType: 'handilearn-gesture-recognizer',
        version: '1.0.0',
        date: new Date().toISOString(),
        distanceOptimized: true,
        recommendedSettings: {
          confidenceThreshold: 0.5,
          windowSize: 7,
          minHoldFrames: 5,
          useDoubleSmoothing: true,
          stabilityBonus: 0.1
        }
      };

      // Save metadata
      const metadataStr = JSON.stringify(metadata, null, 2);
      const metadataBlob = new Blob([metadataStr], {type: 'application/json'});
      const metadataUrl = URL.createObjectURL(metadataBlob);
      const metadataLink = document.createElement('a');
      metadataLink.href = metadataUrl;
      metadataLink.download = 'handilearn-tv-gesture-model-metadata.json';
      document.body.appendChild(metadataLink);

      // Save the model using TensorFlow.js built-in functions
      // This will automatically create model.json and weight files
      await model.save('downloads://handilearn-tv-gesture-model');

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
      <h2>TV-Based Gesture Model Trainer</h2>
      <p className="trainer-description">
        Optimized for detecting gestures at 4-5 feet distance on TV screens
      </p>

      <div className="trainer-container">
        <div className="trainer-panel">
          <div className="upload-section">
            <h3>1. Upload Gesture Data</h3>
            <p className="section-tip">
              For TV-based detection, collect samples at 4-5 feet distance with good lighting
            </p>
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

            {datasetAnalysis && (
              <div className="data-summary">
                <h4>Dataset Analysis:</h4>
                <div className="dataset-stats">
                  <div className="stat-item">
                    <span className="stat-label">Gestures:</span>
                    <span className="stat-value">{datasetAnalysis.uniqueClasses}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Total Samples:</span>
                    <span className="stat-value">{datasetAnalysis.totalSamples}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Distance Readiness:</span>
                    <span className={`stat-value readiness-${datasetAnalysis.distanceReadiness.overallReadiness}`}>
                      {datasetAnalysis.distanceReadiness.readinessScore}/10
                    </span>
                  </div>
                </div>

                <div className="gesture-list">
                  {Object.entries(datasetAnalysis.gestureCounts).map(([label, count]) => {
                    // Get readiness assessment for this gesture
                    const assessment = datasetAnalysis.distanceReadiness.gestureAssessment[label];
                    const readinessClass = assessment ? assessment.readiness : 'insufficient';

                    return (
                      <div key={label} className={`gesture-item readiness-${readinessClass}`}>
                        <span className="label">{formatGestureName(label)}</span>
                        <span className="count">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          <div className="model-params">
            <h3>2. Configure Distance-Optimized Model</h3>

            <div className="distance-augmentation">
              <label>
                <input
                  type="checkbox"
                  checked={modelParams.useDistanceAugmentation}
                  onChange={handleToggleAugmentation}
                  disabled={trainingStatus === 'training'}
                />
                Enable Distance Augmentation (Recommended)
              </label>
              <span className="param-help">Generates additional training data with distance-like noise patterns</span>
            </div>

            {modelParams.useDistanceAugmentation && (
              <div className="param-group">
                <label htmlFor="augFactor">Augmentation Factor:</label>
                <select
                  id="augFactor"
                  value={modelParams.augmentationFactor}
                  onChange={(e) => handleParamChange('augmentationFactor', parseInt(e.target.value))}
                  disabled={trainingStatus === 'training'}
                >
                  <option value="1">Light (1x original data)</option>
                  <option value="2">Medium (2x original data)</option>
                  <option value="3">Heavy (3x original data)</option>
                </select>
                <span className="param-help">Higher values generate more augmented samples</span>
              </div>
            )}

            <div className="param-group">
              <label htmlFor="epochs">Epochs:</label>
              <select
                id="epochs"
                value={modelParams.epochs}
                onChange={(e) => handleParamChange('epochs', parseInt(e.target.value))}
                disabled={trainingStatus === 'training'}
              >
                <option value="30">30 (Faster Training)</option>
                <option value="50">50 (Recommended for Distance)</option>
                <option value="75">75 (Better Distance Accuracy)</option>
                <option value="100">100 (Maximum Accuracy)</option>
              </select>
              <span className="param-help">More epochs improve distance accuracy but take longer</span>
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
                <option value="128,64,32">Extra Large (128→64→32 units)</option>
              </select>
              <span className="param-help">Larger models work better for distance detection</span>
            </div>

            <div className="param-group">
              <label htmlFor="dropoutRate">Regularization:</label>
              <select
                id="dropoutRate"
                value={modelParams.dropoutRate}
                onChange={(e) => handleParamChange('dropoutRate', parseFloat(e.target.value))}
                disabled={trainingStatus === 'training'}
              >
                <option value="0.2">Light (0.2)</option>
                <option value="0.3">Medium (0.3)</option>
                <option value="0.4">Heavy (0.4)</option>
              </select>
              <span className="param-help">Higher values help with distance generalization</span>
            </div>

            <div className="param-group">
              <label htmlFor="batchSize">Batch Size:</label>
              <select
                id="batchSize"
                value={modelParams.batchSize}
                onChange={(e) => handleParamChange('batchSize', parseInt(e.target.value))}
                disabled={trainingStatus === 'training'}
              >
                <option value="8">8 (Lower Memory Usage)</option>
                <option value="16">16 (Balanced)</option>
                <option value="32">32 (Faster Training)</option>
              </select>
              <span className="param-help">Use smaller batch size for limited memory</span>
            </div>
          </div>

          <div className="train-section">
            <h3>3. Train Distance-Optimized Model</h3>
            <button
              className="train-button"
              onClick={trainModel}
              disabled={trainingStatus !== 'ready' || !splitData}
            >
              {trainingStatus === 'training' ? 'Training...' : 'Train Model for TV Distance'}
            </button>

            {evaluationResults && (
              <div className="evaluation-results">
                <h4>Evaluation Results:</h4>
                <div className="result-item">
                  <span>Overall Accuracy:</span>
                  <span className={`accuracy ${evaluationResults.accuracy > 0.85 ? 'excellent' : (evaluationResults.accuracy > 0.75 ? 'good' : 'insufficient')}`}>
                    {(evaluationResults.accuracy * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="result-item">
                  <span>Distance Performance:</span>
                  <span className={`accuracy ${evaluationResults.accuracy > 0.85 ? 'excellent' : (evaluationResults.accuracy > 0.75 ? 'good' : 'insufficient')}`}>
                    {evaluationResults.accuracy > 0.85 ? 'Excellent' : (evaluationResults.accuracy > 0.75 ? 'Good' : 'Needs Improvement')}
                  </span>
                </div>
              </div>
            )}

            {trainingStatus === 'complete' && (
              <div className="model-actions">
                <button
                  className="save-button"
                  onClick={saveModel}
                >
                  Save TV-Optimized Model
                </button>
                <button
                  className="test-button"
                  onClick={testModel}
                >
                  Test Distance Inference
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
                Training in progress... (this may take several minutes)
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="instructions">
        <h3>TV-Based Detection Instructions:</h3>
        <ol>
          <li>Collect gesture samples at <strong>4-5 feet distance</strong> (TV viewing distance)</li>
          <li>Aim for <strong>50+ samples per gesture</strong> for reliable distance detection</li>
          <li>Ensure samples are collected with <strong>good front lighting</strong></li>
          <li>Use <strong>Distance Augmentation</strong> to improve model performance</li>
          <li>Choose <strong>larger model size</strong> and <strong>more epochs</strong> for better distance accuracy</li>
          <li>After training, save the model and <strong>test thoroughly</strong> at your target distance</li>
        </ol>
        <p className="distance-tip"><strong>Tip:</strong> For best results, position the camera at eye level, centered under/above the TV, and ensure it has a clear view of the interaction space.</p>
      </div>
    </div>
  );
}

export default GestureModelTrainer;
