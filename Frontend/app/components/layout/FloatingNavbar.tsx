import React, { useState } from "react";
import AnimatedText from "../ui/AnimatedText";
import "./FloatingNavbar.css";

interface FloatingNavbarProps {
  activeTab?: string;
  onTabChange?: (tab: string) => void;
}

const FloatingNavbar: React.FC<FloatingNavbarProps> = ({
  activeTab = "Dashboard",
  onTabChange
}) => {
  const [currentTab, setCurrentTab] = useState(activeTab);

  const navItems = ["Dashboard", "News", "Settings"];

  const handleTabClick = (tab: string) => {
    setCurrentTab(tab);
    if (onTabChange) {
      onTabChange(tab);
    }
  };

  return (
    <nav className="floating-navbar">
      <div className="floating-navbar-container">
        {/* Navigation Items - directly in main panel */}
        {navItems.map((item) => (
          <button
            key={item}
            onClick={() => handleTabClick(item)}
            className={`nav-item ${currentTab === item ? 'active' : ''}`}
          >
            <AnimatedText>{item}</AnimatedText>
          </button>
        ))}
      </div>
    </nav>
  );
};

export default FloatingNavbar;
