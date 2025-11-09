import React from "react";
import { useNavigate } from "react-router";
import { useAuth } from "../../contexts/AuthContext";
import Card from "../ui/Card";
import Button from "../ui/Button";
import "./ProfileCard.css";

const ProfileCard: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout();
      navigate("/login");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  if (!user) {
    return null;
  }

  // Get first letter of email for avatar
  const avatarLetter = user.email?.charAt(0).toUpperCase() || "U";

  return (
    <Card className="profile-card">
      <div className="profile-info">
        <div className="profile-avatar">{avatarLetter}</div>
        <div className="profile-details">
          <div className="profile-name">{user.displayName || "User"}</div>
          <div className="profile-email">{user.email}</div>
        </div>
      </div>
      <Button onClick={handleLogout}>Logout</Button>
    </Card>
  );
};

export default ProfileCard;
