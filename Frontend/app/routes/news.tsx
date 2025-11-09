import React from "react";
import ProtectedRoute from "../components/auth/ProtectedRoute";
import FloatingNavbar from "../components/layout/FloatingNavbar";

const NewsPage: React.FC = () => {
  return (
    <ProtectedRoute>
      <FloatingNavbar activeTab="News" />
      <div style={{ 
        minHeight: "100vh", 
        padding: "2rem",
        paddingTop: "6rem", // Space for floating navbar
        color: "white",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: "2rem"
      }}>
        <h1 style={{ 
          fontSize: "2.5rem", 
          fontWeight: "600", 
          margin: "0",
          textAlign: "center"
        }}>
          Market News
        </h1>
        <p style={{ 
          fontSize: "1.2rem", 
          color: "rgba(255, 255, 255, 0.7)",
          textAlign: "center",
          maxWidth: "600px",
          lineHeight: "1.6"
        }}>
          Stay updated with the latest market news, analysis, and insights. 
          This section will feature real-time financial news, market trends, and trading opportunities.
        </p>
        <div style={{
          padding: "2rem",
          background: "rgba(255, 255, 255, 0.1)",
          borderRadius: "12px",
          border: "1px solid rgba(255, 255, 255, 0.2)",
          textAlign: "center",
          maxWidth: "500px",
          width: "100%"
        }}>
          <h3 style={{ margin: "0 0 1rem 0", color: "#00ff88" }}>Coming Soon</h3>
          <p style={{ margin: "0", color: "rgba(255, 255, 255, 0.8)" }}>
            Real-time market news integration will be available here.
          </p>
        </div>
      </div>
    </ProtectedRoute>
  );
};

export default NewsPage;
