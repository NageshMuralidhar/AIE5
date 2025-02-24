import React, { useEffect, useRef } from 'react';

const WaveCanvas = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    let animationFrameId;

    // Set canvas size
    const setCanvasSize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    // Initial setup
    setCanvasSize();

    // Wave parameters
    const waves = [
      { amplitude: 120, frequency: 0.001, speed: 0.0005, phase: 0 },
      { amplitude: 80, frequency: 0.002, speed: 0.0004, phase: 2 }
    ];

    // Get theme colors
    const getColors = () => {
      const isDarkTheme = document.body.getAttribute('data-theme') === 'dark';
      return {
        firstWave: isDarkTheme 
          ? { r: 147, g: 51, b: 234 }  // Purple for dark theme
          : { r: 18, g: 163, b: 176 },  // #12A3B0 Bright teal
        secondWave: isDarkTheme
          ? { r: 59, g: 130, b: 246 }   // Blue for dark theme
          : { r: 1, g: 73, b: 81 }      // #014951 Medium teal
      };
    };

    // Draw function
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Center the waves vertically
      const centerY = canvas.height * 0.5;
      const colors = getColors();

      waves.forEach((wave, index) => {
        // Update wave phase
        wave.phase += wave.speed;

        // Create gradient with theme-aware colors
        const gradient = ctx.createLinearGradient(0, centerY - wave.amplitude, 0, centerY + wave.amplitude);
        const color = index === 0 ? colors.firstWave : colors.secondWave;
        
        if (document.body.getAttribute('data-theme') === 'dark') {
          gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);
          gradient.addColorStop(0.5, `rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`);
          gradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);
        } else {
          // Light theme gradient with additional color stops
          gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);
          gradient.addColorStop(0.2, `rgba(1, 49, 53, 0.4)`);  // #013135
          gradient.addColorStop(0.5, `rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`);
          gradient.addColorStop(0.8, `rgba(175, 221, 229, 0.4)`);  // #AFDDE5
          gradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);
        }

        // Begin drawing wave
        ctx.beginPath();
        
        // Start from bottom left
        ctx.moveTo(0, canvas.height);
        ctx.lineTo(0, centerY);

        // Draw wave path
        for (let x = 0; x <= canvas.width; x++) {
          const y = centerY + 
                   Math.sin(x * wave.frequency + wave.phase) * wave.amplitude;
          ctx.lineTo(x, y);
        }

        // Complete the path to bottom right
        ctx.lineTo(canvas.width, centerY);
        ctx.lineTo(canvas.width, canvas.height);
        ctx.closePath();

        // Fill with gradient
        ctx.fillStyle = gradient;
        ctx.fill();
      });

      animationFrameId = requestAnimationFrame(draw);
    };

    // Start animation
    draw();

    // Handle resize and theme changes
    const handleResize = () => {
      setCanvasSize();
    };

    window.addEventListener('resize', handleResize);

    // Watch for theme changes
    const observer = new MutationObserver(() => {
      draw(); // Redraw when theme changes
    });
    
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-theme']
    });

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      observer.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="wave-canvas" />;
};

export default WaveCanvas; 