"""Run the pipeline integration smoke test interactively.

Usage: PYTHONPATH=. python3 scripts/run_pipeline_test.py
"""
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_pipeline_integration import test_scaler_and_trade_tracker_integration

if __name__ == '__main__':
    try:
        test_scaler_and_trade_tracker_integration()
        print('PIPELINE_TEST_OK')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('PIPELINE_TEST_FAILED:', e)
        raise
