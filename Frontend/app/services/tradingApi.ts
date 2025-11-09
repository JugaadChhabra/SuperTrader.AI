// Trading API Service for SuperTrader.AI Backend

const API_BASE_URL = 'http://localhost:8000/api';

export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

export interface CompletedTrade {
  symbol: string;
  entry_time?: string;
  entry_date?: string;
  exit_time?: string;
  exit_date?: string;
  quantity: number;
  entry_price: number;
  exit_price: number;
  pnl?: number;
  net_pnl?: number;
  pnl_pct?: number;
  return_percent?: number;
  trade_id?: string;
}

export interface PortfolioSummary {
  portfolio_value: number;
  cash_balance: number;
  total_invested: number;
  open_positions: Position[];
  unrealized_pnl: number;
  realized_pnl: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  average_pnl: number;
}

export interface TradingStatus {
  portfolio_value: number;
  cash_balance: number;
  open_positions: number;
  total_trades: number;
  models_initialized: boolean;
  dqn_available: boolean;
}

export interface AIDecision {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reason: string;
  current_price: number;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  backend_initialized: boolean;
  dqn_model_available: boolean;
  market_status: 'OPEN' | 'CLOSED';
  message: string;
}

export interface MarketData {
  symbol: string;
  ltp: number;
  change: number;
  change_pct: number;
  volume: number;
  timestamp: string;
}

export interface ExecuteTradeRequest {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  quantity?: number;
}

export interface ExecuteTradeResponse {
  success: boolean;
  message: string;
  position?: Position;
  trade?: CompletedTrade;
}

export interface BacktestSummary {
  initial_capital: number;
  final_value: number;
  total_return: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  total_trades: number;
  open_positions: number;
  cash_balance: number;
}

export interface BacktestResults {
  summary: BacktestSummary;
  trades: CompletedTrade[];
  positions: any;
  ai_info: any;
  generated_at: string;
}

// API Functions
export const tradingApi = {
  // Health check
  async getHealth(): Promise<HealthStatus> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('Failed to fetch health status');
    return response.json();
  },

  // Get current trading status
  async getTradingStatus(): Promise<TradingStatus> {
    const response = await fetch(`${API_BASE_URL}/trading/status`);
    if (!response.ok) throw new Error('Failed to fetch trading status');
    return response.json();
  },

  // Get AI decision for a symbol
  async getAIDecision(symbol: string): Promise<AIDecision> {
    const response = await fetch(`${API_BASE_URL}/trading/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol })
    });
    if (!response.ok) throw new Error('Failed to get AI decision');
    return response.json();
  },

  // Get portfolio summary
  async getPortfolioSummary(): Promise<PortfolioSummary> {
    const response = await fetch(`${API_BASE_URL}/portfolio/summary`);
    if (!response.ok) throw new Error('Failed to fetch portfolio summary');
    return response.json();
  },

  // Get trade history
  async getTradeHistory(): Promise<CompletedTrade[]> {
    const response = await fetch(`${API_BASE_URL}/trades/history`);
    if (!response.ok) throw new Error('Failed to fetch trade history');
    return response.json();
  },

  // Execute trade
  async executeTrade(request: ExecuteTradeRequest): Promise<ExecuteTradeResponse> {
    const response = await fetch(`${API_BASE_URL}/trading/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    if (!response.ok) throw new Error('Failed to execute trade');
    return response.json();
  },

  // Get market data for symbols
  async getMarketData(symbols: string[]): Promise<MarketData[]> {
    const symbolsQuery = symbols.join(',');
    const response = await fetch(`${API_BASE_URL}/market/data?symbols=${symbolsQuery}`);
    if (!response.ok) throw new Error('Failed to fetch market data');
    return response.json();
  },

  // Get backtest/validation results
  async getBacktestResults(): Promise<BacktestResults> {
    const response = await fetch(`${API_BASE_URL}/backtest/results`);
    if (!response.ok) throw new Error('Failed to fetch backtest results');
    return response.json();
  }
};

// Utility function to format currency
export const formatCurrency = (value: number): string => {
  const sign = value >= 0 ? '+' : '';
  return `${sign}₹${value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

// Utility function to format percentage
export const formatPercentage = (value: number): string => {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
};

// Utility function to check if market is open (9:30 AM - 3:15 PM)
export const isMarketOpen = (): boolean => {
  const now = new Date();
  const dayOfWeek = now.getDay(); // 0 = Sunday, 6 = Saturday
  
  // Market closed on weekends
  if (dayOfWeek === 0 || dayOfWeek === 6) {
    return false;
  }
  
  const hours = now.getHours();
  const minutes = now.getMinutes();
  const currentTime = hours * 60 + minutes;
  
  const marketOpen = 9 * 60 + 30; // 9:30 AM
  const marketClose = 15 * 60 + 15; // 3:15 PM
  
  return currentTime >= marketOpen && currentTime <= marketClose;
};

// Utility function to get time until market close
export const getTimeUntilMarketClose = (): string => {
  const now = new Date();
  const dayOfWeek = now.getDay();
  
  // If weekend, show when market opens Monday
  if (dayOfWeek === 0 || dayOfWeek === 6) {
    const daysUntilMonday = dayOfWeek === 0 ? 1 : 2;
    return `Market opens Monday 9:30 AM`;
  }
  
  const marketClose = new Date();
  marketClose.setHours(15, 15, 0, 0);
  
  const diff = marketClose.getTime() - now.getTime();
  if (diff < 0) return 'Market Closed';
  
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  
  return `${hours}h ${minutes}m until close`;
};

export default tradingApi;
