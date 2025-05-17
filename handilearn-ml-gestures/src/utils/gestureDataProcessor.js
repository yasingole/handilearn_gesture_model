/**
 * Utilities for processing hand gesture data for ML training
 */

class GestureDataProcessor {
  /**
   * Normalize a single hand's landmarks
   * @param {Array} landmarks - Flattened array of x,y,z coordinates [x1,y1,z1,x2,y2,z2,...]
   * @returns {Array} - Normalized flattened array
   */
  static normalizeLandmarks(landmarks) {
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

  /**
   * Process a dataset of gesture samples
   * @param {Object} gestureData - Object containing gesture samples by label
   * @returns {Object} - Processed dataset ready for model training
   */
  static processDataset(gestureData) {
    if (!gestureData || !gestureData.gestures) {
      throw new Error('Invalid gesture data format');
    }

    const processed = {
      features: [],
      labels: [],
      labelMapping: {},
      gestures: []
    };

    // Create a mapping of label names to numeric indices
    const labelNames = Object.keys(gestureData.gestures);
    processed.gestures = labelNames;

    labelNames.forEach((label, index) => {
      processed.labelMapping[label] = index;
    });

    // Process each gesture sample
    for (const [label, samples] of Object.entries(gestureData.gestures)) {
      if (!samples || !Array.isArray(samples)) continue;

      for (const sample of samples) {
        try {
          // Normalize the landmarks
          const normalizedFeatures = this.normalizeLandmarks(sample.landmarks);

          // Add to dataset
          processed.features.push(normalizedFeatures);
          processed.labels.push(processed.labelMapping[label]);
        } catch (error) {
          console.warn(`Error processing sample for ${label}:`, error);
          continue;
        }
      }
    }

    return processed;
  }

  /**
   * Split dataset into training and testing sets
   * @param {Object} processedData - Data processed by processDataset()
   * @param {Number} testRatio - Ratio of data to use for testing (0-1)
   * @returns {Object} - Object containing train and test sets
   */
  static splitTrainTest(processedData, testRatio = 0.2) {
    const { features, labels, labelMapping, gestures } = processedData;

    if (!features || !labels || features.length !== labels.length) {
      throw new Error('Invalid processed data format');
    }

    // Create index array and shuffle it
    const indices = Array.from({ length: features.length }, (_, i) => i);
    this.shuffleArray(indices);

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

  /**
   * Shuffle an array in-place
   * @param {Array} array - Array to shuffle
   */
  static shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  /**
   * Format data for TensorFlow.js
   * @param {Object} splitData - Data from splitTrainTest()
   * @returns {Object} - Data formatted for TensorFlow.js
   */
  static formatForTensorflow(splitData) {
    const { train, test, labelMapping, gestures } = splitData;

    // We need to convert to the format TensorFlow.js expects
    return {
      train: {
        xs: train.features, // x input features
        ys: train.labels    // y output labels
      },
      test: {
        xs: test.features,
        ys: test.labels
      },
      metadata: {
        labels: gestures,
        labelMapping: labelMapping
      }
    };
  }

  /**
   * Augment a dataset to create additional training samples
   * @param {Object} processedData - Data processed by processDataset()
   * @param {Number} augmentationFactor - How many variants to create per sample
   * @returns {Object} - Augmented dataset
   */
  static augmentDataset(processedData, augmentationFactor = 2) {
    const { features, labels, labelMapping, gestures } = processedData;

    const augmentedFeatures = [...features];
    const augmentedLabels = [...labels];

    for (let i = 0; i < features.length; i++) {
      const originalFeatures = features[i];
      const label = labels[i];

      // Create multiple augmented versions
      for (let j = 0; j < augmentationFactor; j++) {
        // Apply random noise to each landmark
        const augmentedFeature = originalFeatures.map(val => {
          // Add small random noise (±5%)
          const noise = (Math.random() - 0.5) * 0.1;
          return val + noise;
        });

        augmentedFeatures.push(augmentedFeature);
        augmentedLabels.push(label);
      }
    }

    return {
      features: augmentedFeatures,
      labels: augmentedLabels,
      labelMapping,
      gestures
    };
  }

  /**
   * Analyze a dataset to check for class imbalance
   * @param {Object} processedData - Data processed by processDataset()
   * @returns {Object} - Analysis results
   */
  static analyzeDataset(processedData) {
    const { labels, labelMapping, gestures } = processedData;

    // Count occurrences of each class
    const classCounts = {};
    labels.forEach(label => {
      classCounts[label] = (classCounts[label] || 0) + 1;
    });

    // Convert numeric indices to gesture names
    const gestureToIndex = Object.entries(labelMapping).reduce((acc, [gesture, index]) => {
      acc[index] = gesture;
      return acc;
    }, {});

    const gestureCounts = {};
    Object.entries(classCounts).forEach(([index, count]) => {
      gestureCounts[gestureToIndex[index]] = count;
    });

    // Calculate statistics
    const totalSamples = labels.length;
    const uniqueClasses = Object.keys(classCounts).length;
    const minCount = Math.min(...Object.values(classCounts));
    const maxCount = Math.max(...Object.values(classCounts));
    const avgCount = totalSamples / uniqueClasses;

    // Check for imbalance
    const imbalanceRatio = maxCount / minCount;
    const isImbalanced = imbalanceRatio > 1.5;

    return {
      totalSamples,
      uniqueClasses,
      gestureCounts,
      minCount,
      maxCount,
      avgCount,
      imbalanceRatio,
      isImbalanced
    };
  }
}

export default GestureDataProcessor;
