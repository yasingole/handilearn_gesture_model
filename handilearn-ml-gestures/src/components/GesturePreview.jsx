import React, { useRef, useEffect } from 'react';

/**
 * Component to visualize a 3D hand model
 */
function GesturePreview({ landmarks, label }) {
  const canvasRef = useRef(null);

  // Format gesture name
  const formatGestureName = (name) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // Draw the 3D hand preview
  useEffect(() => {
    if (!canvasRef.current || !landmarks || landmarks.length !== 21) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Set up coordinate system centered in canvas
    // We'll use a more zoomed in view for better detail
    const scale = Math.min(width, height) * 0.4;
    const centerX = width / 2;
    const centerY = height / 2;

    // Define connections between landmarks (finger joints)
    const connections = [
      // Thumb
      [0, 1], [1, 2], [2, 3], [3, 4],
      // Index finger
      [0, 5], [5, 6], [6, 7], [7, 8],
      // Middle finger
      [0, 9], [9, 10], [10, 11], [11, 12],
      // Ring finger
      [0, 13], [13, 14], [14, 15], [15, 16],
      // Pinky
      [0, 17], [17, 18], [18, 19], [19, 20],
      // Palm
      [0, 5], [5, 9], [9, 13], [13, 17]
    ];

    // Calculate bounds of landmarks for normalization
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    landmarks.forEach(landmark => {
      minX = Math.min(minX, landmark.x);
      maxX = Math.max(maxX, landmark.x);
      minY = Math.min(minY, landmark.y);
      maxY = Math.max(maxY, landmark.y);
    });

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const normalizeScale = 0.8 / Math.max(rangeX, rangeY);

    // Adjust the center to the middle of the hand
    const centerOffsetX = (minX + maxX) / 2;
    const centerOffsetY = (minY + maxY) / 2;

    // Draw connections
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 3;

    for (const [i, j] of connections) {
      const startPoint = landmarks[i];
      const endPoint = landmarks[j];

      const startX = centerX + (startPoint.x - centerOffsetX) * scale * normalizeScale;
      const startY = centerY + (startPoint.y - centerOffsetY) * scale * normalizeScale;
      const endX = centerX + (endPoint.x - centerOffsetX) * scale * normalizeScale;
      const endY = centerY + (endPoint.y - centerOffsetY) * scale * normalizeScale;

      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }

    // Draw landmarks with different colors for each finger
    const fingerColors = [
      '#E74C3C', // Thumb (red)
      '#3498DB', // Index (blue)
      '#2ECC71', // Middle (green)
      '#F39C12', // Ring (orange)
      '#9B59B6'  // Pinky (purple)
    ];

    // Helper to get finger index from landmark index
    const getFingerIndex = (i) => {
      if (i >= 1 && i <= 4) return 0; // Thumb
      if (i >= 5 && i <= 8) return 1; // Index
      if (i >= 9 && i <= 12) return 2; // Middle
      if (i >= 13 && i <= 16) return 3; // Ring
      if (i >= 17 && i <= 20) return 4; // Pinky
      return -1; // Wrist
    };

    // Draw landmarks
    landmarks.forEach((landmark, i) => {
      const fingerIndex = getFingerIndex(i);
      ctx.fillStyle = fingerIndex >= 0 ? fingerColors[fingerIndex] : '#FFFFFF';

      const x = centerX + (landmark.x - centerOffsetX) * scale * normalizeScale;
      const y = centerY + (landmark.y - centerOffsetY) * scale * normalizeScale;

      ctx.beginPath();

      // Draw larger circle for fingertips and wrist
      const isWrist = i === 0;
      const isFingertip = [4, 8, 12, 16, 20].includes(i);
      const radius = isWrist ? 8 : (isFingertip ? 6 : 4);

      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fill();

      // Add landmark labels for key points
      if (isWrist || isFingertip) {
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(i.toString(), x, y);
      }
    });

    // Draw depth indicators (z values as colors)
    if (landmarks[0].z !== undefined) {
      // Draw in a separate area
      const depthIndicatorSize = 10;
      const padding = 5;
      const depthDisplayWidth = (landmarks.length * depthIndicatorSize) + ((landmarks.length - 1) * padding);
      const depthDisplayX = (width - depthDisplayWidth) / 2;
      const depthDisplayY = height - 30;

      // Find min/max Z values for normalization
      let minZ = Infinity, maxZ = -Infinity;
      landmarks.forEach(landmark => {
        minZ = Math.min(minZ, landmark.z);
        maxZ = Math.max(maxZ, landmark.z);
      });

      const zRange = maxZ - minZ || 1;

      // Draw depth indicator legend
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Z-Depth (lighter = closer to camera)', width / 2, depthDisplayY - 15);

      // Draw depth indicators
      landmarks.forEach((landmark, i) => {
        const fingerIndex = getFingerIndex(i);
        const baseColor = fingerIndex >= 0 ? fingerColors[fingerIndex] : '#FFFFFF';

        // Normalize Z value to 0-1 range
        const normalizedZ = (landmark.z - minZ) / zRange;

        // Use alpha to indicate depth (closer to camera = more opaque)
        const alpha = 1 - normalizedZ;
        const x = depthDisplayX + (i * (depthIndicatorSize + padding));

        ctx.fillStyle = baseColor;
        ctx.globalAlpha = alpha;
        ctx.fillRect(x, depthDisplayY, depthIndicatorSize, depthIndicatorSize);
        ctx.globalAlpha = 1.0;
      });
    }

    // Draw the gesture name
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(formatGestureName(label), width / 2, 20);

  }, [landmarks, label]);

  return (
    <div className="gesture-preview-container">
      <canvas
        ref={canvasRef}
        width={300}
        height={200}
        style={{
          backgroundColor: '#333',
          borderRadius: '5px',
          width: '100%',
          height: 'auto'
        }}
      />
    </div>
  );
}

export default GesturePreview;
