import React from "react";
import { Link, useLocation } from "react-router";
import "./Navbar.css";

const Navbar: React.FC = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        {/* Logo/Brand */}
        <div className="navbar-brand">
          <Link to="/" className="brand-link">
            SuperTrader.AI
          </Link>
        </div>

        {/* Navigation Links */}
        <div className="navbar-nav">
          <Link 
            to="/dashboard" 
            className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
          >
            Dashboard
          </Link>
          <Link 
            to="/news" 
            className={`nav-link ${isActive('/news') ? 'active' : ''}`}
          >
            News
          </Link>
          <Link 
            to="/settings" 
            className={`nav-link ${isActive('/settings') ? 'active' : ''}`}
          >
            Settings
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
