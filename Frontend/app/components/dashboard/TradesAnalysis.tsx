import React from "react";
import "./TradesAnalysis.css";

interface Trade {
  id: number;
  type: "Long" | "Short";
  action: "Entry" | "Exit";
  date: string;
  signal: string;
  price: number;
  positionSize: number;
  netPL: number;
  netPLPercentage: number;
  runUp: number;
  runUpPercentage: number;
  drawdown: number;
  drawdownPercentage: number;
  cumulativePL: number;
  cumulativePLPercentage: number;
}

interface TradesAnalysisProps {
  trades?: Trade[];
}

const TradesAnalysis: React.FC<TradesAnalysisProps> = ({ trades }) => {
  // Default trades data based on the image
  const defaultTrades: Trade[] = [
    {
      id: 90,
      type: "Short",
      action: "Exit",
      date: "Sep 08, 2025",
      signal: "Open",
      price: 6797.50,
      positionSize: 29,
      netPL: -3183.18,
      netPLPercentage: -1.62,
      runUp: 7720.45,
      runUpPercentage: 3.92,
      drawdown: -5126.55,
      drawdownPercentage: -2.60,
      cumulativePL: 1422799.20,
      cumulativePLPercentage: 284.56
    },
    {
      id: 90,
      type: "Short",
      action: "Entry",
      date: "Sep 03, 2025",
      signal: "My Short Entry Id",
      price: 6755.50,
      positionSize: 29,
      netPL: 0,
      netPLPercentage: 0,
      runUp: 0,
      runUpPercentage: 0,
      drawdown: 0,
      drawdownPercentage: 0,
      cumulativePL: 0,
      cumulativePLPercentage: 0
    },
    {
      id: 89,
      type: "Long",
      action: "Exit",
      date: "Sep 03, 2025",
      signal: "My Short Entry Id",
      price: 6755.50,
      positionSize: 41,
      netPL: 77357.48,
      netPLPercentage: 39.02,
      runUp: 111952.37,
      runUpPercentage: 56.48,
      drawdown: -8003.38,
      drawdownPercentage: -4.04,
      cumulativePL: 1426962.00,
      cumulativePLPercentage: 285.39
    },
    {
      id: 89,
      type: "Long",
      action: "Entry",
      date: "Mar 21, 2025",
      signal: "My Long Entry Id",
      price: 4810.90,
      positionSize: 41,
      netPL: 0,
      netPLPercentage: 0,
      runUp: 0,
      runUpPercentage: 0,
      drawdown: 0,
      drawdownPercentage: 0,
      cumulativePL: 0,
      cumulativePLPercentage: 0
    },
    {
      id: 88,
      type: "Short",
      action: "Exit",
      date: "Mar 21, 2025",
      signal: "My Long Entry Id",
      price: 4810.90,
      positionSize: 45,
      netPL: -21512.18,
      netPLPercentage: -10.86,
      runUp: 6667.02,
      runUpPercentage: 3.37,
      drawdown: -21995.73,
      drawdownPercentage: -11.11,
      cumulativePL: 1349604.50,
      cumulativePLPercentage: 269.92
    },
    {
      id: 88,
      type: "Short",
      action: "Entry",
      date: "Feb 28, 2025",
      signal: "My Short Entry Id",
      price: 4378.80,
      positionSize: 45,
      netPL: 0,
      netPLPercentage: 0,
      runUp: 0,
      runUpPercentage: 0,
      drawdown: 0,
      drawdownPercentage: 0,
      cumulativePL: 0,
      cumulativePLPercentage: 0
    },
    {
      id: 87,
      type: "Long",
      action: "Exit",
      date: "Feb 28, 2025",
      signal: "My Short Entry Id",
      price: 4378.80,
      positionSize: 41,
      netPL: -18225.80,
      netPLPercentage: -9.26,
      runUp: 7196.00,
      runUpPercentage: 3.66,
      drawdown: -18502.80,
      drawdownPercentage: -9.40,
      cumulativePL: 1371116.60,
      cumulativePLPercentage: 274.22
    },
    {
      id: 87,
      type: "Long",
      action: "Entry",
      date: "Jan 23, 2025",
      signal: "My Long Entry Id",
      price: 4777.55,
      positionSize: 41,
      netPL: 0,
      netPLPercentage: 0,
      runUp: 0,
      runUpPercentage: 0,
      drawdown: 0,
      drawdownPercentage: 0,
      cumulativePL: 0,
      cumulativePLPercentage: 0
    },
    {
      id: 86,
      type: "Short",
      action: "Exit",
      date: "Jan 23, 2025",
      signal: "My Long Entry Id",
      price: 4777.55,
      positionSize: 45,
      netPL: -21121.06,
      netPLPercentage: -10.73,
      runUp: 0,
      runUpPercentage: 0.00,
      drawdown: -22386.12,
      drawdownPercentage: -11.37,
      cumulativePL: 1389342.50,
      cumulativePLPercentage: 277.87
    },
    {
      id: 86,
      type: "Short",
      action: "Entry",
      date: "Jan 13, 2025",
      signal: "My Short Entry Id",
      price: 4353.85,
      positionSize: 45,
      netPL: 0,
      netPLPercentage: 0,
      runUp: 0,
      runUpPercentage: 0,
      drawdown: 0,
      drawdownPercentage: 0,
      cumulativePL: 0,
      cumulativePLPercentage: 0
    }
  ];

  const displayTrades = trades || defaultTrades;

  const formatCurrency = (value: number) => {
    return value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  const formatPercentage = (value: number) => {
    if (value === 0) return "0.00%";
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className="trades-analysis">
      <div className="trades-table">
        <div className="trades-header">
          <span>Trade # ↓</span>
          <span>Type</span>
          <span>Date/Time</span>
          <span>Signal</span>
          <span>Price</span>
          <span>Position size</span>
          <span>Net P&L</span>
          <span>Run-up</span>
          <span>Drawdown</span>
          <span>Cumulative P&L</span>
        </div>

        <div className="trades-body">
          {displayTrades.map((trade, index) => (
            <div key={index} className="trade-row">
              <span className="trade-id">
                {index === 0 || trade.id !== displayTrades[index - 1]?.id ? (
                  <>
                    {trade.id} <span className={`trade-type ${trade.type.toLowerCase()}`}>{trade.type}</span>
                  </>
                ) : (
                  ""
                )}
              </span>
              
              <span className="trade-action">{trade.action}</span>
              <span className="trade-date">{trade.date}</span>
              <span className="trade-signal">{trade.signal}</span>
              <span className="trade-price">
                {formatCurrency(trade.price)} <span className="currency">INR</span>
              </span>
              <span className="trade-position">
                {trade.positionSize}
                <br />
                <span className="position-value">{formatCurrency(trade.positionSize * 1000)} <span className="currency">INR</span></span>
              </span>
              <span className={`trade-pnl ${trade.netPL > 0 ? 'positive' : trade.netPL < 0 ? 'negative' : 'neutral'}`}>
                {trade.netPL !== 0 && (
                  <>
                    {trade.netPL > 0 ? '+' : ''}{formatCurrency(trade.netPL)} <span className="currency">INR</span>
                    <br />
                    <span className="percentage">{formatPercentage(trade.netPLPercentage)}</span>
                  </>
                )}
              </span>
              <span className={`trade-runup ${trade.runUp > 0 ? 'positive' : 'neutral'}`}>
                {trade.runUp !== 0 && (
                  <>
                    {formatCurrency(trade.runUp)} <span className="currency">INR</span>
                    <br />
                    <span className="percentage">{formatPercentage(trade.runUpPercentage)}</span>
                  </>
                )}
              </span>
              <span className={`trade-drawdown ${trade.drawdown < 0 ? 'negative' : 'neutral'}`}>
                {trade.drawdown !== 0 && (
                  <>
                    {formatCurrency(trade.drawdown)} <span className="currency">INR</span>
                    <br />
                    <span className="percentage">{formatPercentage(trade.drawdownPercentage)}</span>
                  </>
                )}
              </span>
              <span className={`trade-cumulative ${trade.cumulativePL > 0 ? 'positive' : trade.cumulativePL < 0 ? 'negative' : 'neutral'}`}>
                {trade.cumulativePL !== 0 && (
                  <>
                    {formatCurrency(trade.cumulativePL)} <span className="currency">INR</span>
                    <br />
                    <span className="percentage">{formatPercentage(trade.cumulativePLPercentage)}</span>
                  </>
                )}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TradesAnalysis;
