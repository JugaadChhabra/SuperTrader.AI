import React from "react";
import ProtectedRoute from "../components/auth/ProtectedRoute";
import ProfileCard from "../components/auth/ProfileCard";

const ProfilePage: React.FC = () => (
  <ProtectedRoute>
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <ProfileCard />
    </div>
  </ProtectedRoute>
);

export default ProfilePage;
