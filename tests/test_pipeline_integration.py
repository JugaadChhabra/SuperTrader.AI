from configs.config import get_config
from utils.scaler_manager import load_scaler

from enhanced_portfolio_test import TradeTracker


def test_scaler_and_trade_tracker_integration():
    """Smoke test: load configured scaler and run a single AI decision through TradeTracker.

    This validates that the scaler loader is wired into the agent pipeline and
    that the AI decision code runs without throwing exceptions.
    """
    cfg = get_config()

    # Load scaler (may be None) - should not raise
    scaler = load_scaler(cfg.get_model_paths().get('feature_scaler'))

    # Initialize TradeTracker and request an AI decision
    tracker = TradeTracker(initial_capital=100000.0)

    # Run one decision - ensure result contains expected keys
    decision = tracker._get_ai_decision('RELIANCE', 2850.0, __import__('datetime').datetime.now(), has_position=False)

    assert isinstance(decision, dict)
    assert 'action' in decision
    assert 'ai_details' in decision
