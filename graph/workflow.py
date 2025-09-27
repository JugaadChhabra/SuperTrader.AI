from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging 
from langgraph.graph import StateGraph, Graph
from langgraph.prebuilt import ToolNode
import asyncio
import numpy as np
import pandas as pd

from agents.data_agent import DataAgent
from agents.news_agent import NewsAgent
from agents.rl_strategy_agent import RLStrategyAgent
from agents.execution_agent import ExecutionAgent
from utils.config import load_config
from utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class TradingState:
    "global state for all agents in the workflow"

    symbols: List[str]
    timestamp: datetime
    market_data: Dict[str, pd.DataFrame]  

    technical_features: Dict[str, np.ndarray]
    sentiment_features: Dict[str, np.ndarray]

    rl_state: Optional[np.ndarray] 
    raw_actions: Dict[str, float]
    position_sizes: Dict[str, float]

    portfolio_exposure: float
    current_positions: Dict[str, float]
    risk_metrics: Dict[str, Any]

    pending_orders: List[Dict[str, Any]]
    executed_trades: List[Dict[str, Any]]

    data_ready: bool = False
    news_ready: bool = False
    rl_ready: bool = False
    execution_ready: bool = False
    risk_approved: bool = False

    errors: List[str] = None


class TradingWorkflow:
    """
    Main workflow orchestator for managing the sequential execution of agents.
    """

    def __init__(self, config_path: str = "configs/"):
        self.config = load_config(config_path)
        self.graph = None

        self.data_agent = DataAgent(self.config.get('data', {}))
        self.news_agent = NewsAgent(self.config.get('news', {}))
        self.rl_agent = RLStrategyAgent(self.config.get('rl', {}))
        self.execution_agent = ExecutionAgent(self.config.get('execution', {}))

        self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TradingState)

        workflow.add_node("data_agent", self._data_node)
        workflow.add_node("news_agent", self._news_node)
        workflow.add_node("rl_agent", self._rl_node)
        workflow.add_node("execution_agent", self._execution_node)
        workflow.add_node("risk_guard", self._risk_guard_node)

        workflow.add_edge("data_agent", "news_agent")
        workflow.add_edge("news_agent", "rl_agent")
        workflow.add_edge("rl_agent", "execution_agent")
        workflow.add_edge("execution_agent", "risk_guard")

        workflow.set_entry_point("data_agent")

        workflow.add_conditional_edges(
            "risk_guard",
            self._should_continue,
            {
                "continue": "data_agent",
                "stop": "__end__"
            }
        )

        self.graph = workflow.compile()


    async def _data_node(self, state: TradingState) -> TradingState:
        """
        data agent node for fetchng and processing market data
        """

        try:
            logger.info(f"DataAgent processing for symbols: {state.symbols}")

            market_data = {}
            for symbol in state.symbols:
                df = await self.data_agent.fetch_ohlcv(
                    symbol = symbol,
                    interval="5m",
                    lookback_period=100
                )
                
                if self.data_agent.validate_data(df):
                    market_data[symbol] = df
                else:
                    state.errors.append(f"Invalid data for {symbol}")
                    continue
            
            state.market_data = market_data

            technical_features = {}
            for symbol, df in market_data.items():
                features = self.data_agent.compute_indicators(df)
                feature_array = self.data_agent.build_feature_frame(df, features)
                technical_features[symbol] = feature_array

            state.technical_features = technical_features
            state.data_ready = True
            
            logger.info(f"DataAgent completed, processed {len(market_data)}")
        
        except Exception as e:
            logger.error(f"DataAgent error: {str(e)}")
            state.errors.append(f"DataAgent: {str(e)}")

        return state
    
    async def _news_node(self, state: TradingState) -> TradingState:
        """
        News agent node that fetches and analyzes sentiment
        """

        try:
            if not state.data_ready:
                raise ValueError("Data not ready for processing")
            
            logger.info("NewsAgent analyzing sentiments")

            # Fetch news for all symbols
            news_data = await self.news_agent.fetch_news_blobs(
                symbols=state.symbols,
                lookback_hours=24
            )
            
            sentiment_features = {}
            for symbol in state.symbols:
                symbol_sentiment = self.news_agent.score_sentiment(
                    symbol = symbol,
                    news_data = news_data
                )
                sentiment_features[symbol] = self.news_agent.build_sentiment_features(
                    symbol=symbol,
                    sentiment_scores=symbol_sentiment,
                    timestamp=state.timestamp
                )

                state.sentiment_features = sentiment_features
                state.news_ready = True

                logger.info("NewsAgent completed sentiment analysis")

        except Exception as e:
            logger.error(f"NewsAgent error: {str(e)}")
            state.errors.append(f"NewsAgent: {str(e)}")
            
        return state
    
    async def _rl_node(self, state: TradingState) -> TradingState:
        """RL Strategy Agent Node - Generate trading decisions"""
        try:
            if not (state.data_ready and state.news_ready):
                raise ValueError("Data and news not ready for RL processing")
                
            logger.info("RLAgent generating trading decisions")
            
            # Build state representations and generate actions
            raw_actions = {}
            position_sizes = {}
            
            for symbol in state.symbols:
                # Combine technical and sentiment features
                rl_state = self.rl_agent.build_state_representation(
                    price_features=state.technical_features[symbol],
                    tech_features=state.technical_features[symbol],
                    sentiment_features=state.sentiment_features[symbol]
                )
                
                # Sample action from policy
                raw_action = self.rl_agent.sample_action(
                    state=rl_state,
                    mode="inference"
                )
                
                # Compute position size with risk controls
                position_size = self.rl_agent.compute_position_size(
                    state=rl_state,
                    raw_action=raw_action,
                    risk_params=self.rl_agent.risk_params
                )
                
                raw_actions[symbol] = raw_action
                position_sizes[symbol] = position_size
            
            state.raw_actions = raw_actions
            state.position_sizes = position_sizes
            state.rl_ready = True
            
            logger.info(f"RLAgent completed. Generated {len(raw_actions)} decisions")
            
        except Exception as e:
            logger.error(f"RLAgent error: {str(e)}")
            state.errors.append(f"RLAgent: {str(e)}")
            
        return state
    
    async def _execution_node(self, state: TradingState) -> TradingState:
        """Execution Agent Node - Prepare and route orders"""
        try:
            if not state.rl_ready:
                raise ValueError("RL decisions not ready for execution")
                
            logger.info("ExecutionAgent preparing orders")
            
            # Get current portfolio state
            current_portfolio = await self.execution_agent.get_portfolio_state()
            state.current_positions = current_portfolio.get('positions', {})
            
            # Prepare orders for each symbol
            pending_orders = []
            
            for symbol, target_position in state.position_sizes.items():
                current_position = state.current_positions.get(symbol, 0.0)
                
                # Calculate required trade size
                trade_size = target_position - current_position
                
                if abs(trade_size) > 0.001:  # Minimum trade threshold
                    # Build order
                    order = await self.execution_agent.build_order(
                        symbol=symbol,
                        size=trade_size,
                        order_type="MARKET",
                        time_in_force="IOC"
                    )
                    
                    # Pre-trade risk checks
                    if self.execution_agent.pre_trade_checks(order, current_portfolio):
                        pending_orders.append(order)
                    else:
                        state.errors.append(f"Pre-trade check failed for {symbol}")
            
            state.pending_orders = pending_orders
            state.execution_ready = True
            
            logger.info(f"ExecutionAgent prepared {len(pending_orders)} orders")
            
        except Exception as e:
            logger.error(f"ExecutionAgent error: {str(e)}")
            state.errors.append(f"ExecutionAgent: {str(e)}")
            
        return state
    
    async def _risk_guard_node(self, state: TradingState) -> TradingState:
        """Risk Guard Node - Final risk validation and execution"""
        try:
            if not state.execution_ready:
                raise ValueError("Orders not ready for risk validation")
                
            logger.info("RiskGuard performing final validation")
            
            # Calculate portfolio-level risk metrics
            risk_metrics = self._calculate_risk_metrics(state)
            state.risk_metrics = risk_metrics
            
            # Portfolio exposure check
            total_exposure = sum(abs(pos) for pos in state.position_sizes.values())
            state.portfolio_exposure = total_exposure
            
            max_exposure = self.config.get('risk', {}).get('max_portfolio_exposure', 1.0)
            
            if total_exposure <= max_exposure:
                # Execute approved orders
                executed_trades = []
                
                for order in state.pending_orders:
                    try:
                        trade_result = await self.execution_agent.place_order(order)
                        executed_trades.append(trade_result)
                        logger.info(f"Executed order: {order['symbol']} {order['size']}")
                        
                    except Exception as e:
                        logger.error(f"Order execution failed: {str(e)}")
                        state.errors.append(f"Execution failed: {str(e)}")
                
                state.executed_trades = executed_trades
                state.risk_approved = True
                
                logger.info(f"RiskGuard approved and executed {len(executed_trades)} trades")
                
            else:
                state.errors.append(f"Portfolio exposure {total_exposure:.2%} exceeds limit {max_exposure:.2%}")
                logger.warning(f"Portfolio exposure limit exceeded: {total_exposure:.2%}")
            
        except Exception as e:
            logger.error(f"RiskGuard error: {str(e)}")
            state.errors.append(f"RiskGuard: {str(e)}")
            
        return state
    
    def _calculate_risk_metrics(self, state: TradingState) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        
        # Get current positions and market data
        positions = state.position_sizes
        market_data = state.market_data
        
        # Calculate metrics
        risk_metrics = {
            'exposure_pct': sum(abs(pos) for pos in positions.values()),
            'long_exposure': sum(max(0, pos) for pos in positions.values()),
            'short_exposure': sum(min(0, pos) for pos in positions.values()),
            'net_exposure': sum(positions.values()),
            'num_positions': len([p for p in positions.values() if abs(p) > 0.001]),
            'max_position': max(abs(pos) for pos in positions.values()) if positions else 0,
            'timestamp': state.timestamp.isoformat()
        }
        
        # Add sector/asset class concentration if available
        # This would be enhanced with actual sector classification
        risk_metrics['sector_concentration'] = self._calculate_sector_concentration(positions)
        
        return risk_metrics
    
    def _calculate_sector_concentration(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector concentration (placeholder implementation)"""
        # In a real implementation, you would map symbols to sectors
        # and calculate concentration metrics
        return {
            'max_sector_exposure': 0.3,  # Placeholder
            'sector_count': len(positions),
            'herfindahl_index': 0.15  # Placeholder concentration index
        }
    
    def _should_continue(self, state: TradingState) -> str:
        """Determine if workflow should continue or stop"""
        
        # Check for critical errors
        critical_errors = [error for error in state.errors 
                          if any(keyword in error.lower() 
                                for keyword in ['critical', 'fatal', 'connection'])]
        
        if critical_errors:
            logger.error(f"Critical errors detected: {critical_errors}")
            return "stop"
        
        # Check if we should continue trading
        current_hour = state.timestamp.hour
        market_hours = self.config.get('trading', {}).get('market_hours', [9, 16])
        
        if market_hours[0] <= current_hour <= market_hours[1]:
            return "continue"
        else:
            logger.info("Outside market hours, stopping workflow")
            return "stop"
    
    async def run_single_cycle(self, symbols: List[str], timestamp: datetime = None) -> TradingState:
        """Run a single trading cycle"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize state
        initial_state = TradingState(
            symbols=symbols,
            timestamp=timestamp,
            market_data={},
            technical_features={},
            sentiment_features={},
            rl_state=None,
            raw_actions={},
            position_sizes={},
            portfolio_exposure=0.0,
            current_positions={},
            risk_metrics={},
            pending_orders=[],
            executed_trades=[]
        )
        
        logger.info(f"Starting trading cycle for {len(symbols)} symbols at {timestamp}")
        
        # Execute workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            # Log final results
            if final_state.risk_approved:
                logger.info(f"Trading cycle completed successfully. "
                           f"Executed {len(final_state.executed_trades)} trades")
            else:
                logger.warning(f"Trading cycle completed with errors: {final_state.errors}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            initial_state.errors.append(f"Workflow: {str(e)}")
            return initial_state
    
    async def run_continuous(self, symbols: List[str], interval_minutes: int = 5):
        """Run continuous trading with specified interval"""
        
        logger.info(f"Starting continuous trading for {symbols} every {interval_minutes} minutes")
        
        while True:
            try:
                # Run trading cycle
                result = await self.run_single_cycle(symbols)
                
                # Log results
                if result.errors:
                    logger.warning(f"Cycle completed with errors: {result.errors}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Continuous trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Continuous trading error: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


# Example usage
async def main():
    """Example of how to use the trading workflow"""
    
    # Initialize workflow
    workflow = TradingWorkflow()
    
    # Define trading universe
    symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFC"]
    
    # Run single cycle
    result = await workflow.run_single_cycle(symbols)
    
    print(f"Trading cycle completed:")
    print(f"Errors: {result.errors}")
    print(f"Executed trades: {len(result.executed_trades)}")
    print(f"Portfolio exposure: {result.portfolio_exposure:.2%}")
    
    # For continuous trading (uncomment to run)
    # await workflow.run_continuous(symbols, interval_minutes=5)

if __name__ == "__main__":
    asyncio.run(main())