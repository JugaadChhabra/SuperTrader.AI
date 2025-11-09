import React from "react";
import Card from "../ui/Card";
import "./StatsCard.css";

interface StatsCardProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
}

const StatsCard: React.FC<StatsCardProps> = ({ label, value, icon }) => (
  <Card className="stats-card">
    <div className="stats-content">
      {icon && <span className="stats-icon">{icon}</span>}
      <div>
        <div className="stats-label">{label}</div>
        <div className="stats-value">{value}</div>
      </div>
    </div>
  </Card>
);

export default StatsCard;
