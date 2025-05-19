// src/components/ModelUploader.jsx
import React, { useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import ModelTester from './ModelTester';
import './ModelUploader.css';

function ModelUploader() {
  const [model, setModel] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [loadingStatus, setLoadingStatus] = useState('idle'); // idle, loading, success, error
  const [error, setError] = useState(null);

  const modelJsonInputRef = useRef(null);
  const modelWeightsInputRef = useRef(null);
  const metadataInputRef = useRef(null);

  // Handle model JSON file upload
  const handleModelJsonUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Store the file for later use
    setLoadingStatus('waiting_for_weights');

    // Auto-click the weights upload if we have a JSON
    setTimeout(() => {
      if (modelWeightsInputRef.current) {
        modelWeightsInputRef.current.click();
      }
    }, 500);
  };

  // Handle model weights file upload
  const handleModelWeightsUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoadingStatus('waiting_for_metadata');

    // Auto-click the metadata upload
    setTimeout(() => {
      if (metadataInputRef.current) {
        metadataInputRef.current.click();
      }
    }, 500);
  };

  // Handle metadata file upload
  const handleMetadataUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const metadataJson = JSON.parse(e.target.result);
        setMetadata(metadataJson);
        setLoadingStatus('loading_model');

        // Now load the model files
        loadModelFiles();
      } catch (error) {
        setError(`Error parsing metadata: ${error.message}`);
        setLoadingStatus('error');
      }
    };

    reader.readAsText(file);
  };

  // Load model from the user's files
  const loadModelFiles = async () => {
    try {
      setLoadingStatus('loading_model');

      // Get model JSON and weights files
      const jsonFile = modelJsonInputRef.current.files[0];
      const weightsFile = modelWeightsInputRef.current.files[0];

      if (!jsonFile || !weightsFile) {
        throw new Error('Both model JSON and weights files are required');
      }

      // Create a URL for the model JSON file
      const jsonUrl = URL.createObjectURL(jsonFile);

      // Create a custom IOHandler to handle model and weights
      const weightsHandler = {
        load: async () => {
          // Read the weights file
          const arrayBuffer = await weightsFile.arrayBuffer();

          // Parse the model JSON
          const jsonReader = new FileReader();
          const modelJSON = await new Promise((resolve, reject) => {
            jsonReader.onload = e => resolve(JSON.parse(e.target.result));
            jsonReader.onerror = reject;
            jsonReader.readAsText(jsonFile);
          });

          // Return model configuration and weights
          return {
            modelTopology: modelJSON.modelTopology,
            weightSpecs: modelJSON.weightsManifest[0].weights,
            weightData: arrayBuffer,
          };
        }
      };

      // Load the model using a URL for the JSON file and our custom handler for weights
      const loadedModel = await tf.loadLayersModel(weightsHandler);

      // Log model summary
      loadedModel.summary();

      setModel(loadedModel);
      setLoadingStatus('success');
      setError(null);
    } catch (error) {
      console.error('Error loading model:', error);
      setError(`Error loading model: ${error.message}`);
      setLoadingStatus('error');
    }
  };

  // Reset the uploader state
  const resetUploader = () => {
    setModel(null);
    setMetadata(null);
    setLoadingStatus('idle');
    setError(null);

    // Reset file inputs
    if (modelJsonInputRef.current) modelJsonInputRef.current.value = '';
    if (modelWeightsInputRef.current) modelWeightsInputRef.current.value = '';
    if (metadataInputRef.current) metadataInputRef.current.value = '';
  };

  // Simplified upload all at once
  const handleAllFilesUpload = async (event) => {
    event.preventDefault();

    try {
      const jsonFile = modelJsonInputRef.current.files[0];
      const weightsFile = modelWeightsInputRef.current.files[0];
      const metadataFile = metadataInputRef.current.files[0];

      if (!jsonFile || !weightsFile || !metadataFile) {
        throw new Error('All three files (model.json, .weights.bin, and metadata.json) are required');
      }

      setLoadingStatus('loading');

      // 1. Load metadata first
      const metadataReader = new FileReader();
      const metadataJson = await new Promise((resolve, reject) => {
        metadataReader.onload = e => resolve(JSON.parse(e.target.result));
        metadataReader.onerror = reject;
        metadataReader.readAsText(metadataFile);
      });

      setMetadata(metadataJson);

      // 2. Create a URL for the model JSON file
      const jsonUrl = URL.createObjectURL(jsonFile);

      // 3. Create a custom IOHandler to handle model and weights
      const weightsHandler = {
        load: async () => {
          // Read the weights file
          const arrayBuffer = await weightsFile.arrayBuffer();

          // Parse the model JSON
          const jsonReader = new FileReader();
          const modelJSON = await new Promise((resolve, reject) => {
            jsonReader.onload = e => resolve(JSON.parse(e.target.result));
            jsonReader.onerror = reject;
            jsonReader.readAsText(jsonFile);
          });

          // Return model configuration and weights
          return {
            modelTopology: modelJSON.modelTopology,
            weightSpecs: modelJSON.weightsManifest[0].weights,
            weightData: arrayBuffer,
          };
        }
      };

      // Load the model
      const loadedModel = await tf.loadLayersModel(weightsHandler);

      // Log model summary
      loadedModel.summary();

      setModel(loadedModel);
      setLoadingStatus('success');
      setError(null);
    } catch (error) {
      console.error('Error loading model files:', error);
      setError(`Error loading model: ${error.message}`);
      setLoadingStatus('error');
    }
  };

  return (
    <div className="model-uploader">
      <h2>Test Pre-Trained Model</h2>
      <p className="uploader-description">
        Upload your previously trained model and metadata files to test them
      </p>

      {loadingStatus !== 'success' && (
        <form className="upload-form" onSubmit={handleAllFilesUpload}>
          <div className="file-upload-group">
            <label>
              <span>1. Model Structure (model.json)</span>
              <input
                type="file"
                ref={modelJsonInputRef}
                accept=".json"
                onChange={handleModelJsonUpload}
                required
              />
            </label>

            <label>
              <span>2. Model Weights (.weights.bin)</span>
              <input
                type="file"
                ref={modelWeightsInputRef}
                accept=".bin"
                onChange={handleModelWeightsUpload}
                required
              />
            </label>

            <label>
              <span>3. Model Metadata (metadata.json)</span>
              <input
                type="file"
                ref={metadataInputRef}
                accept=".json"
                onChange={handleMetadataUpload}
                required
              />
            </label>
          </div>

          <button
            type="submit"
            className="load-model-button"
            disabled={loadingStatus === 'loading'}
          >
            {loadingStatus === 'loading' ? 'Loading Model...' : 'Load & Test Model'}
          </button>
        </form>
      )}

      {loadingStatus === 'loading' && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <span>Loading model...</span>
        </div>
      )}

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {loadingStatus === 'success' && model && metadata && (
        <div className="model-test-container">
          <div className="model-info-panel">
            <h3>Model Loaded Successfully</h3>
            <div className="model-details">
              <div className="detail-item">
                <span className="label">Gesture Classes:</span>
                <span className="value">{metadata.gestures?.length || 'Unknown'}</span>
              </div>
              <div className="detail-item">
                <span className="label">Input Size:</span>
                <span className="value">{metadata.inputSize || 'Unknown'}</span>
              </div>
              <div className="detail-item">
                <span className="label">Created:</span>
                <span className="value">{metadata.date ? new Date(metadata.date).toLocaleDateString() : 'Unknown'}</span>
              </div>
              {metadata.distanceOptimized && (
                <div className="detail-item">
                  <span className="label">Distance Optimized:</span>
                  <span className="value">Yes</span>
                </div>
              )}
            </div>

            {metadata.gestures && (
              <div className="gestures-list">
                <h4>Available Gestures:</h4>
                <div className="gestures-grid">
                  {metadata.gestures.map((gesture, index) => (
                    <div key={index} className="gesture-item">
                      {gesture.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <ModelTester model={model} metadata={metadata} />

          <button
            className="reset-button"
            onClick={resetUploader}
          >
            Upload a Different Model
          </button>
        </div>
      )}
    </div>
  );
}

export default ModelUploader;
