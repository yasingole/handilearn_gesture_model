/**
 * Utilities for processing hand gesture data for ML training
 * Optimized for TV-based detection at 4-5 feet distance
 */

class GestureDataProcessor {
  /**
   * Normalize a single hand's landmarks with distance optimization
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

    // Calculate hand bounding box for more robust scaling at distance
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    centeredLandmarks.forEach(point => {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      minZ = Math.min(minZ, point.z);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
      maxZ = Math.max(maxZ, point.z);
    });

    // Get the maximum range across all dimensions for uniform scaling
    const rangeX = maxX - minX || 0.001; // Avoid division by zero
    const rangeY = maxY - minY || 0.001;
    const rangeZ = maxZ - minZ || 0.001;
    const maxRange = Math.max(rangeX, rangeY, rangeZ);

    // Scale to normalize the hand size (more robust than single finger distance)
    const scaleFactor = maxRange > 0 ? 2.0 / maxRange : 1.0;

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
          // Normalize the landmarks with distance optimization
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

    // Create index array
    const indices = Array.from({ length: features.length }, (_, i) => i);

    // Group indices by label for stratified splitting
    const labelIndices = {};
    labels.forEach((label, index) => {
      if (!labelIndices[label]) {
        labelIndices[label] = [];
      }
      labelIndices[label].push(indices[index]);
    });

    // Create stratified train/test split
    const trainIndices = [];
    const testIndices = [];

    Object.values(labelIndices).forEach(indexGroup => {
      // Shuffle indices for this class
      this.shuffleArray(indexGroup);

      // Calculate split point for this class
      const splitPoint = Math.floor(indexGroup.length * (1 - testRatio));

      // Add to train/test sets
      trainIndices.push(...indexGroup.slice(0, splitPoint));
      testIndices.push(...indexGroup.slice(splitPoint));
    });

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
   * Create distance-simulating augmentations for better TV-based detection
   * @param {Object} processedData - Data processed by processDataset()
   * @param {Number} augmentationFactor - How many variants to create per sample
   * @returns {Object} - Augmented dataset
   */
  static distanceAugmentDataset(processedData, augmentationFactor = 2) {
    const { features, labels, labelMapping, gestures } = processedData;

    const augmentedFeatures = [...features];
    const augmentedLabels = [...labels];

    for (let i = 0; i < features.length; i++) {
      const originalFeatures = features[i];
      const label = labels[i];

      // Create multiple augmented versions
      for (let j = 0; j < augmentationFactor; j++) {
        // Create distance-simulating noise levels
        const distanceNoiseLevel = 0.15 + (j * 0.05); // Increasing noise for "farther" simulations

        // Apply distance-simulating noise
        const augmentedFeature = this.applyDistanceSimulation(originalFeatures, distanceNoiseLevel);

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
   * Apply distance-simulating noise to landmarks
   * @param {Array} features - Original features
   * @param {Number} noiseLevel - Amount of noise to add
   * @returns {Array} - Distance-simulated features
   */
  static applyDistanceSimulation(features, noiseLevel = 0.15) {
    // For distance simulation, we apply different noise patterns:
    // 1. More noise on z-coordinates (depth perception degrades with distance)
    // 2. Correlated noise on finger joints (distant hands have less precise joint tracking)

    const result = [...features];

    // Apply different noise levels to different coordinates
    for (let i = 0; i < features.length; i += 3) {
      // Position-based noise variance (fingertips have more jitter at distance)
      const jointIndex = Math.floor(i / 3);
      const isFingertip = [4, 8, 12, 16, 20].includes(jointIndex);
      const isWrist = jointIndex === 0;

      // Base noise factor varies by joint type
      const baseFactor = isFingertip ? 1.5 : (isWrist ? 0.5 : 1.0);

      // Apply noise to x/y coordinates (screen plane)
      const xyNoise = noiseLevel * baseFactor;
      result[i] += (Math.random() - 0.5) * xyNoise;     // x
      result[i + 1] += (Math.random() - 0.5) * xyNoise; // y

      // Apply more noise to z coordinate (depth is less accurate at distance)
      const zNoise = noiseLevel * baseFactor * 2.0;
      result[i + 2] += (Math.random() - 0.5) * zNoise;  // z
    }

    // Apply correlated noise to finger segments
    // This simulates how finger tracking errors are correlated at distance
    this.applyCorrelatedFingerNoise(result, noiseLevel * 0.5);

    return result;
  }

  /**
   * Apply correlated noise to finger segments to simulate distance effects
   * @param {Array} features - Features to modify in-place
   * @param {Number} amount - Amount of correlated noise
   */
  static applyCorrelatedFingerNoise(features, amount = 0.1) {
    // Define finger segments (wrist to fingertip indices)
    const fingerSegments = [
      [0, 1, 2, 3, 4],       // Thumb
      [0, 5, 6, 7, 8],       // Index
      [0, 9, 10, 11, 12],    // Middle
      [0, 13, 14, 15, 16],   // Ring
      [0, 17, 18, 19, 20]    // Pinky
    ];

    // For each finger, apply a correlated rotation/translation
    fingerSegments.forEach(segment => {
      // Generate random values for this finger's correlated movement
      const xShift = (Math.random() - 0.5) * amount;
      const yShift = (Math.random() - 0.5) * amount;
      const zShift = (Math.random() - 0.5) * amount * 2; // More z noise

      // Apply to all joints in this finger (skip wrist - index 0)
      for (let i = 1; i < segment.length; i++) {
        const jointIdx = segment[i] * 3;
        features[jointIdx] += xShift;
        features[jointIdx + 1] += yShift;
        features[jointIdx + 2] += zShift;
      }
    });
  }

  /**
   * Analyze a dataset to check for class imbalance and data quality
   * @param {Object} processedData - Data processed by processDataset()
   * @returns {Object} - Analysis results
   */
  static analyzeDataset(processedData) {
    const { features, labels, labelMapping, gestures } = processedData;

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

    // Check for potential distance-related issues
    const distanceReadiness = this.assessDistanceReadiness(gestureCounts);

    return {
      totalSamples,
      uniqueClasses,
      gestureCounts,
      minCount,
      maxCount,
      avgCount,
      imbalanceRatio,
      isImbalanced,
      distanceReadiness
    };
  }

  /**
   * Assess dataset readiness for distance-based detection
   * @param {Object} gestureCounts - Count of samples per gesture
   * @returns {Object} - Assessment results
   */
  static assessDistanceReadiness(gestureCounts) {
    // For TV-based detection, we need more samples and good diversity
    const minRecommended = 50; // 50 samples per gesture minimum for distance
    const idealRecommended = 100; // 100+ samples per gesture ideal for robust distance detection

    // Check each gesture for sufficient samples
    const gestureAssessment = {};
    let totalReady = 0;

    Object.entries(gestureCounts).forEach(([gesture, count]) => {
      let readiness = 'insufficient';
      if (count >= idealRecommended) {
        readiness = 'excellent';
        totalReady += 2;
      } else if (count >= minRecommended) {
        readiness = 'adequate';
        totalReady += 1;
      }

      gestureAssessment[gesture] = {
        count,
        readiness,
        minRecommended,
        idealRecommended,
        additionalNeeded: count < minRecommended ? minRecommended - count : 0
      };
    });

    // Overall readiness score (0-10)
    const maxScore = Object.keys(gestureCounts).length * 2; // 2 points per gesture
    const readinessScore = Math.round((totalReady / maxScore) * 10);

    // Overall assessment
    let overallReadiness = 'insufficient';
    if (readinessScore >= 8) {
      overallReadiness = 'excellent';
    } else if (readinessScore >= 5) {
      overallReadiness = 'adequate';
    }

    return {
      gestureAssessment,
      readinessScore,
      overallReadiness,
      recommendation: this.getDistanceRecommendation(readinessScore)
    };
  }

  /**
   * Get recommendation based on distance readiness score
   * @param {Number} score - Readiness score (0-10)
   * @returns {String} - Recommendation
   */
  static getDistanceRecommendation(score) {
    if (score >= 8) {
      return "Dataset looks good for distance-based detection. Consider applying distance augmentation to further improve robustness.";
    } else if (score >= 5) {
      return "Dataset may work for distance-based detection, but consider collecting more samples for better results. Definitely use distance augmentation.";
    } else {
      return "Dataset likely insufficient for reliable distance-based detection. Collect more samples at the actual usage distance (4-5 feet). Aim for 50+ samples per gesture.";
    }
  }
}

export default GestureDataProcessor;
