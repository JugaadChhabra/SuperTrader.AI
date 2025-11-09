import React from "react";
import { Navigate } from "react-router";
import { useAuth } from "../contexts/AuthContext";
import SignupForm from "../components/auth/SignupForm";

const SignupPage: React.FC = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: '#fff',
        fontSize: '1.2rem'
      }}>
        Loading...
      </div>
    );
  }

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <SignupForm />
    </div>
  );
};

export default SignupPage;
