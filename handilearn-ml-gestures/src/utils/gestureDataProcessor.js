/**
 * Utility functions for processing gesture data
 */

/**
 * Process raw gesture data to prepare for model training
 * @param {Array} recordings - Array of recorded gesture data
 * @returns {Object} - Processed data ready for training
 */
export const processGestureData = (recordings) => {
  if (!recordings || !recordings.length) {
    return { samples: [], labels: [] }
  }

  // Group recordings by gesture name
  const groupedByGesture = recordings.reduce((groups, record) => {
    if (!groups[record.name]) {
      groups[record.name] = []
    }
    groups[record.name].push(record)
    return groups
  }, {})

  // Convert data to training format
  // In a real implementation, this would normalize and augment the data
  const samples = []
  const labels = []

  Object.entries(groupedByGesture).forEach(([gestureName, gestureRecordings]) => {
    gestureRecordings.forEach(recording => {
      // In a real app, this would process actual hand landmark data
      // For this template, we're just creating placeholder data
      samples.push({
        id: recording.id,
        features: preprocessHandLandmarks(recording.frames)
      })

      labels.push(gestureName)
    })
  })

  return {
    samples,
    labels,
    classNames: Object.keys(groupedByGesture),
    stats: {
      totalSamples: samples.length,
      samplesPerClass: Object.fromEntries(
        Object.entries(groupedByGesture).map(([name, data]) => [name, data.length])
      )
    }
  }
}

/**
 * Preprocess hand landmarks data for the model
 * @param {Array} frames - Raw frames of hand landmarks
 * @returns {Array} - Preprocessed feature data
 */
const preprocessHandLandmarks = (frames) => {
  // This is a placeholder function
  // In a real implementation, this would:
  // - Normalize coordinates
  // - Extract relevant features
  // - Handle temporal aspects of the gesture

  // Return mock processed data for template
  return [
    // This would be the processed features
    0.5, 0.2, 0.7, 0.3, 0.1, 0.9, 0.4, 0.6
  ]
}

/**
 * Calculate statistics about the collected gesture data
 * @param {Array} recordings - Array of recorded gesture data
 * @returns {Object} - Statistics about the data
 */
export const calculateDataStatistics = (recordings) => {
  if (!recordings || !recordings.length) {
    return {
      totalRecordings: 0,
      uniqueGestures: 0,
      recordingsPerGesture: {},
      averageFramesPerRecording: 0
    }
  }

  // Count recordings per gesture
  const recordingsPerGesture = recordings.reduce((counts, record) => {
    counts[record.name] = (counts[record.name] || 0) + 1
    return counts
  }, {})

  // Calculate average frames per recording
  const totalFrames = recordings.reduce((sum, record) => sum + (record.frames?.length || 0), 0)
  const averageFrames = totalFrames / recordings.length

  return {
    totalRecordings: recordings.length,
    uniqueGestures: Object.keys(recordingsPerGesture).length,
    recordingsPerGesture,
    averageFramesPerRecording: averageFrames
  }
}
