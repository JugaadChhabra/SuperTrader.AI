import React from "react";
import Card from "../ui/Card";
import "./PortfolioCard.css";

interface PortfolioHolding {
  symbol: string;
  change: number;
  ltp: number; // Last Traded Price
}

interface PortfolioCardProps {
  holdings: PortfolioHolding[];
  totalValue?: number;
  totalChange?: number;
}

const PortfolioCard: React.FC<PortfolioCardProps> = ({ 
  holdings, 
  totalValue, 
  totalChange 
}) => {
  return (
    <Card className="portfolio-card">
      <div className="portfolio-header">
        <h2>PORTFOLIO</h2>
        {totalValue && (
          <div className="portfolio-summary">
            <div className="total-value">₹{totalValue.toLocaleString()}</div>
            {totalChange && (
              <div className={`total-change ${totalChange >= 0 ? 'positive' : 'negative'}`}>
                {totalChange >= 0 ? '+' : ''}{totalChange.toFixed(2)}%
              </div>
            )}
          </div>
        )}
      </div>
      
      <div className="portfolio-table">
        <div className="portfolio-table-header">
          <span>SYMBOL</span>
          <span>%change</span>
          <span>LTP</span>
        </div>
        
        <div className="portfolio-holdings">
          {holdings.map((holding, index) => (
            <div key={index} className="portfolio-row">
              <span className="symbol">{holding.symbol}</span>
              <span className={`change ${holding.change >= 0 ? 'positive' : 'negative'}`}>
                {holding.change >= 0 ? '+' : ''}{holding.change.toFixed(2)}%
              </span>
              <span className="ltp">{holding.ltp.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};

export default PortfolioCard;
