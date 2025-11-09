import React, { useState } from "react";
import Card from "../ui/Card";
import PerformanceReport from "./PerformanceReport";
import OverviewSection from "./OverviewSection";
import TradesAnalysis from "./TradesAnalysis";
import "./StrategyOverview.css";

interface PerformanceMetric {
  label: string;
  value: string | number;
  percentage?: number;
  isPositive?: boolean;
}

interface PerformanceSection {
  title: string;
  data: {
    all?: PerformanceMetric;
    long?: PerformanceMetric;
    short?: PerformanceMetric;
  };
}

interface StrategyOverviewProps {
  strategyName: string;
  invested: string;
  current: string;
  dateRange: string;
  performanceData: PerformanceSection[];
}

const StrategyOverview: React.FC<StrategyOverviewProps> = ({
  strategyName,
  invested,
  current,
  dateRange,
  performanceData
}) => {
  const [activeTab, setActiveTab] = useState<"Overview" | "Performance" | "Trades analysis">("Performance");

  const renderTabContent = () => {
    switch (activeTab) {
      case "Overview":
        return <OverviewSection />;
      case "Performance":
        return (
          <PerformanceReport
            sections={performanceData}
          />
        );
      case "Trades analysis":
        return <TradesAnalysis />;
      default:
        return null;
    }
  };

  return (
    <Card className="strategy-overview">
      {/* Strategy Header */}
      <div className="strategy-header">
        <div className="strategy-info">
          <h2>{strategyName || "Supertrend Strategy"}</h2>
          <div className="investment-summary">
            <div className="investment-line">Invested - {invested}</div>
            <div className="investment-line">Current - {current}</div>
          </div>
        </div>
        <div className="date-range">{dateRange}</div>
      </div>

      {/* Tab Navigation */}
      <div className="strategy-tabs">
        {(["Overview", "Performance", "Trades analysis"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="strategy-content">
        {renderTabContent()}
      </div>
    </Card>
  );
};

export default StrategyOverview;
