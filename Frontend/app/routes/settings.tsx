import React from "react";
import ProtectedRoute from "../components/auth/ProtectedRoute";
import FloatingNavbar from "../components/layout/FloatingNavbar";

const SettingsPage: React.FC = () => {
  return (
    <ProtectedRoute>
      <FloatingNavbar activeTab="Settings" />
      <div style={{ 
        minHeight: "100vh", 
        padding: "2rem",
        paddingTop: "6rem", // Space for floating navbar
        color: "white"
      }}>
        <h1 style={{ 
          fontSize: "2.5rem", 
          fontWeight: "600", 
          margin: "0 0 2rem 0"
        }}>
          Settings
        </h1>

        <div style={{ 
          display: "grid", 
          gap: "2rem",
          maxWidth: "800px"
        }}>
          {/* Account Settings */}
          <div style={{
            padding: "2rem",
            background: "rgba(255, 255, 255, 0.1)",
            borderRadius: "12px",
            border: "1px solid rgba(255, 255, 255, 0.2)"
          }}>
            <h2 style={{ margin: "0 0 1rem 0", color: "#00ff88" }}>Account Settings</h2>
            <p style={{ color: "rgba(255, 255, 255, 0.7)", margin: "0 0 1.5rem 0" }}>
              Manage your account preferences and profile information.
            </p>
            <button style={{
              padding: "0.75rem 1.5rem",
              background: "rgba(0, 255, 136, 0.1)",
              border: "1px solid #00ff88",
              color: "#00ff88",
              borderRadius: "6px",
              cursor: "pointer"
            }}>
              Edit Profile
            </button>
          </div>

          {/* Trading Settings */}
          <div style={{
            padding: "2rem",
            background: "rgba(255, 255, 255, 0.1)",
            borderRadius: "12px",
            border: "1px solid rgba(255, 255, 255, 0.2)"
          }}>
            <h2 style={{ margin: "0 0 1rem 0", color: "#0066ff" }}>Trading Preferences</h2>
            <p style={{ color: "rgba(255, 255, 255, 0.7)", margin: "0 0 1.5rem 0" }}>
              Configure your trading parameters and risk management settings.
            </p>
            <button style={{
              padding: "0.75rem 1.5rem",
              background: "rgba(0, 102, 255, 0.1)",
              border: "1px solid #0066ff",
              color: "#0066ff",
              borderRadius: "6px",
              cursor: "pointer"
            }}>
              Configure Trading
            </button>
          </div>

          {/* Notifications */}
          <div style={{
            padding: "2rem",
            background: "rgba(255, 255, 255, 0.1)",
            borderRadius: "12px",
            border: "1px solid rgba(255, 255, 255, 0.2)"
          }}>
            <h2 style={{ margin: "0 0 1rem 0", color: "#ff6b35" }}>Notifications</h2>
            <p style={{ color: "rgba(255, 255, 255, 0.7)", margin: "0 0 1.5rem 0" }}>
              Manage your notification preferences and alerts.
            </p>
            <button style={{
              padding: "0.75rem 1.5rem",
              background: "rgba(255, 107, 53, 0.1)",
              border: "1px solid #ff6b35",
              color: "#ff6b35",
              borderRadius: "6px",
              cursor: "pointer"
            }}>
              Manage Notifications
            </button>
          </div>
        </div>
      </div>
    </ProtectedRoute>
  );
};

export default SettingsPage;
