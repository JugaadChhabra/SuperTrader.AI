import React from "react";
import "./OverviewSection.css";

interface OverviewMetric {
  label: string;
  value: string;
  subValue?: string;
  percentage?: string;
  isPositive?: boolean;
}

interface OverviewSectionProps {
  metrics?: OverviewMetric[];
}

const OverviewSection: React.FC<OverviewSectionProps> = ({ metrics }) => {
  // Default metrics based on the image
  const defaultMetrics: OverviewMetric[] = [
    {
      label: "Total P&L",
      value: "+1,424,764.42",
      subValue: "INR",
      percentage: "+285.13%",
      isPositive: true
    },
    {
      label: "Max equity drawdown",
      value: "248,373.21",
      subValue: "INR",
      percentage: "18.16%",
      isPositive: false
    },
    {
      label: "Total trades",
      value: "89",
      isPositive: true
    },
    {
      label: "Profitable trades",
      value: "42.70%",
      subValue: "39/89",
      isPositive: true
    },
    {
      label: "Profit factor",
      value: "2.522",
      isPositive: true
    }
  ];

  const displayMetrics = metrics || defaultMetrics;

  return (
    <div className="overview-section">
      {/* Metrics Row */}
      <div className="overview-metrics">
        {displayMetrics.map((metric, index) => (
          <div key={index} className="overview-metric">
            <div className="metric-label">
              {metric.label}
              <span className="info-icon">ⓘ</span>
            </div>
            <div className="metric-value">
              <span className={`value ${metric.isPositive ? 'positive' : 'negative'}`}>
                {metric.value}
              </span>
              {metric.subValue && (
                <span className="sub-value">{metric.subValue}</span>
              )}
              {metric.percentage && (
                <span className={`percentage ${metric.isPositive ? 'positive' : 'negative'}`}>
                  {metric.percentage}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Chart Section */}
      <div className="overview-chart">
        <div className="chart-header">
          <div className="chart-title">
            <span className="stock-name">Redington Limited</span>
            <div className="stock-info">
              <span>NSE</span>
              <span>Real-time</span>
              <span>Market closed</span>
            </div>
          </div>
          <div className="chart-values">
            <div className="value-item">1,600,000.00</div>
            <div className="value-item">800,000.00</div>
            <div className="value-item">400,000.00</div>
            <div className="value-item">0</div>
            <div className="value-item">-400,000.00</div>
            <div className="value-item">-800,000.00</div>
            <div className="value-item">-1,200,000.00</div>
            <div className="value-item">-1,600,000.00</div>
          </div>
        </div>
        
        {/* Chart Placeholder */}
        <div className="chart-container">
          <div className="chart-placeholder">
            <div className="chart-line"></div>
            <div className="chart-bars">
              {Array.from({ length: 20 }, (_, i) => (
                <div key={i} className="chart-bar" style={{ height: `${Math.random() * 60 + 20}%` }}></div>
              ))}
            </div>
          </div>
        </div>

        {/* Timeline */}
        <div className="chart-timeline">
          <span>2007</span>
          <span>2009</span>
          <span>2011</span>
          <span>2013</span>
          <span>2015</span>
          <span>2017</span>
          <span>2018</span>
          <span>2022</span>
          <span>Jul</span>
          <span>2024</span>
          <span>Feb</span>
        </div>

        {/* Chart Controls */}
        <div className="chart-controls">
          <div className="control-group">
            <label className="checkbox-label">
              <input type="checkbox" />
              <span>Buy & hold</span>
            </label>
            <span className="chart-icon">📈</span>
          </div>
          
          <div className="control-group">
            <label className="checkbox-label checked">
              <input type="checkbox" defaultChecked />
              <span>Trades run-up & drawdown</span>
            </label>
            <span className="chart-icon">📊</span>
          </div>

          <div className="view-toggle">
            <button className="toggle-btn active">Absolute</button>
            <button className="toggle-btn">Percentage</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OverviewSection;
