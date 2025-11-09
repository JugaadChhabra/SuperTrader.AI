import React from "react";
import "./PerformanceReport.css";

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

interface PerformanceReportProps {
  sections: PerformanceSection[];
}

const PerformanceReport: React.FC<PerformanceReportProps> = ({
  sections
}) => {
  const formatValue = (value: string | number) => {
    if (typeof value === 'number') {
      return value.toLocaleString();
    }
    return value;
  };

  return (
    <div className="performance-table">
      <div className="performance-table-header">
        <span>METRIC</span>
        <span>ALL</span>
        <span>LONG</span>
        <span>SHORT</span>
      </div>
      
      <div className="performance-holdings">
        {sections.map((section, sectionIndex) => (
          <div key={sectionIndex} className="performance-row">
            <span className="metric-name">{section.title}</span>
            
            <span className="metric-all">
              {section.data.all && (
                <>
                  <div className="value">{formatValue(section.data.all.value)}</div>
                  {section.data.all.percentage !== undefined && (
                    <div className={`change ${section.data.all.isPositive !== undefined ? (section.data.all.isPositive ? 'positive' : 'negative') : (section.data.all.percentage >= 0 ? 'positive' : 'negative')}`}>
                      {section.data.all.percentage >= 0 ? '+' : ''}{section.data.all.percentage.toFixed(2)}%
                    </div>
                  )}
                </>
              )}
            </span>
            
            <span className="metric-long">
              {section.data.long && (
                <>
                  <div className="value">{formatValue(section.data.long.value)}</div>
                  {section.data.long.percentage !== undefined && (
                    <div className={`change ${section.data.long.isPositive !== undefined ? (section.data.long.isPositive ? 'positive' : 'negative') : (section.data.long.percentage >= 0 ? 'positive' : 'negative')}`}>
                      {section.data.long.percentage >= 0 ? '+' : ''}{section.data.long.percentage.toFixed(2)}%
                    </div>
                  )}
                </>
              )}
            </span>
            
            <span className="metric-short">
              {section.data.short && (
                <>
                  <div className="value">{formatValue(section.data.short.value)}</div>
                  {section.data.short.percentage !== undefined && (
                    <div className={`change ${section.data.short.isPositive !== undefined ? (section.data.short.isPositive ? 'positive' : 'negative') : (section.data.short.percentage >= 0 ? 'positive' : 'negative')}`}>
                      {section.data.short.percentage >= 0 ? '+' : ''}{section.data.short.percentage.toFixed(2)}%
                    </div>
                  )}
                </>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PerformanceReport;
