// src/utils/modelLoader.js
import * as tf from '@tensorflow/tfjs';

/**
 * Utility functions for loading TensorFlow.js models from files
 */
const ModelLoader = {
  /**
   * Load a model from uploaded files
   * @param {File} jsonFile - The model.json file
   * @param {File} weightsFile - The weights.bin file
   * @returns {Promise<tf.LayersModel>} - The loaded model
   */
  loadModelFromFiles: async (jsonFile, weightsFile) => {
    if (!jsonFile || !weightsFile) {
      throw new Error('Both model JSON and weights files are required');
    }

    try {
      // Create a custom IOHandler for loading from files
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
      const model = await tf.loadLayersModel(weightsHandler);
      return model;
    } catch (error) {
      console.error('Error loading model from files:', error);
      throw error;
    }
  },

  /**
   * Load metadata from a file
   * @param {File} metadataFile - The metadata JSON file
   * @returns {Promise<Object>} - The parsed metadata
   */
  loadMetadataFromFile: async (metadataFile) => {
    if (!metadataFile) {
      throw new Error('Metadata file is required');
    }

    try {
      const reader = new FileReader();
      const metadata = await new Promise((resolve, reject) => {
        reader.onload = e => resolve(JSON.parse(e.target.result));
        reader.onerror = reject;
        reader.readAsText(metadataFile);
      });

      return metadata;
    } catch (error) {
      console.error('Error loading metadata from file:', error);
      throw error;
    }
  }
};

export default ModelLoader;
