import React from "react";
// import ProtectedRoute from "../components/auth/ProtectedRoute";
// import FloatingNavbar from "../components/layout/FloatingNavbar";
import IntradayTradingDashboard from "../components/dashboard/IntradayTradingDashboard";

const DashboardPage: React.FC = () => {
  return (
    <div>
      {/* Temporarily disabled auth for testing */}
      <IntradayTradingDashboard />
    </div>
  );
};

export default DashboardPage;
