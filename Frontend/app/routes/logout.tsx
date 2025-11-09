import React from "react";
import { Navigate } from "react-router";
import { useAuth } from "../contexts/AuthContext";

const Logout: React.FC = () => {
  const { logout } = useAuth();

  React.useEffect(() => {
    const handleLogout = async () => {
      try {
        await logout();
      } catch (error) {
        console.error("Logout failed:", error);
      }
    };

    handleLogout();
  }, [logout]);

  return <Navigate to="/login" replace />;
};

export default Logout;