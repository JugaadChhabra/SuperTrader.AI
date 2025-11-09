import React, { useState, useEffect } from "react";
import PixelBlast from "./PixelBlast";

function Header() {
  const [currentColorIndex, setCurrentColorIndex] = useState(0);
  
  const colorTransitions = [
    { color: "#27ae60", hue: 120, name: "Finance Green" },
    { color: "#e74c3c", hue: 0, name: "Alert Red" },
    { color: "#2980b9", hue: 210, name: "Tech Blue" },
    { color: "#f1c40f", hue: 50, name: "Gold" },
    { color: "#8e44ad", hue: 270, name: "Innovation Purple" },
    { color: "#16a085", hue: 170, name: "Analytics Teal" },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentColorIndex(prevIndex => (prevIndex + 1) % colorTransitions.length);
    }, 8000); // Increased interval to reduce transitions
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Optimized PixelBlast with performance settings */}
      <div
        className="absolute inset-0 w-full h-full transition-all duration-3000 ease-in-out"
        style={{
          filter: `hue-rotate(${colorTransitions[currentColorIndex].hue}deg) saturate(1.1) brightness(1.05)`,
          // Add will-change for better GPU acceleration
          willChange: 'filter',
          // Enable hardware acceleration
          transform: 'translateZ(0)',
        }}
      >
        <PixelBlast
          variant="circle"
          color="#4285F4"
          // Performance optimization props (adjust these based on PixelBlast API)
          size={2}           // Smaller particles
          density={30}       // Fewer particles
          speed={0.5}        // Slower animation
          maxParticles={100} // Limit total particles
        >
          <div className="w-full h-full"></div>
        </PixelBlast>
      </div>
    </div>
  );
}

export default Header;