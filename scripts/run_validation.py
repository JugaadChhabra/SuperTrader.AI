"""Validation Evaluation Script

Runs the trained model (Keras or PyTorch) over the validation dataset
(`data/processed_data/validate.csv`) and produces performance metrics.

Outputs JSON + CSV report under `validation_output/`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import datetime as dt

from configs.config import get_config
from utils.scaler_manager import load_scaler
from utils.model_loader import load_model
from data.state_builder import IntradayStateBuilder


def load_validation_dataframe() -> pd.DataFrame:
    cfg = get_config()
    # Path is fixed relative to project root
    csv_path = Path(cfg.rl.dqn_model_dir).parent.parent / "data" / "processed_data" / "validate.csv"
    if not csv_path.exists():
        # Fallback to workspace path
        csv_path = Path("data/processed_data/validate.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation file not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["datetime"])  # retains engineered columns
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_state_sequence(df_window: pd.DataFrame, builder: IntradayStateBuilder) -> np.ndarray:
    # Time feature dictionary for builder
    last_row = df_window.iloc[-1]
    time_features = {
        'minutes_since_open': float(last_row.get('minutes_since_open', 0)),
        'minutes_to_close': float(last_row.get('minutes_to_close', 0)),
        'session_phase': _session_phase_from_row(last_row)
    }
    state_matrix = builder.build_state_vector(
        df_price=df_window[['open','high','low','close','volume']].copy(),
        time_features=time_features,
        sentiment_data={'market_sentiment_5min': 0.0}  # placeholder; could extend
    )  # shape (lookback, features)
    return state_matrix


def _session_phase_from_row(row: pd.Series) -> str:
    phase_val = row.get('session')
    if isinstance(phase_val, str):
        # Map numeric-coded string to textual phase if needed
        return 'morning'  # Simplified; extend mapping if phases present
    return 'morning'


def evaluate():
    cfg = get_config()
    model, model_desc = load_model()
    print(f"Model loader: {model_desc}")
    if model is None:
        print("No model available. Exiting validation early.")
        return

    scaler = load_scaler(cfg.rl.feature_scaler_path)
    builder = IntradayStateBuilder(lookback=30)

    df = load_validation_dataframe()
    lookback = builder.lookback
    position = 0  # -1 short, 0 flat, 1 long
    equity_curve: List[float] = [1.0]
    step_returns: List[float] = []  # realized per-bar return given current position
    trade_pnls: List[float] = []
    current_trade_returns: List[float] = []
    actions_taken: List[Dict[str, Any]] = []

    # Determine expected input size for model
    if hasattr(model, 'input') and hasattr(model.input, 'shape'):
        # Keras: extract feature dimension from input shape (batch, time, features)
        model_feature_size = model.input.shape[-1] if model.input.shape[-1] is not None else 20
    elif hasattr(model, 'num_features'):
        # PyTorch TradingAgent
        model_feature_size = model.num_features
    else:
        model_feature_size = 20  # Default fallback
    
    print(f"📊 Model expects {model_feature_size} features per timestep")
    print(f"📊 State builder produces {builder.lookback} timesteps")
    
    scaling_warned = False  # Warn once about scaling issues

    # Iterate over sliding windows
    for idx in range(lookback, len(df)):
        window = df.iloc[idx - lookback: idx]
        state_seq = build_state_sequence(window, builder)  # (30, 32)
        
        # Adapt state to model's expected input size
        if state_seq.shape[1] > model_feature_size:
            # Slice to first N features (price + technical + time features priority)
            state_seq = state_seq[:, :model_feature_size]
        elif state_seq.shape[1] < model_feature_size:
            # Pad with zeros if needed
            padding = np.zeros((state_seq.shape[0], model_feature_size - state_seq.shape[1]))
            state_seq = np.hstack([state_seq, padding])

        # Keras expects (batch, time, features); PyTorch agent expects (time, features)
        if hasattr(model, 'predict'):
            # Keras path
            input_tensor = state_seq[None, :, :]  # Add batch dimension
            
            # Skip scaling if mismatch (Keras model likely has normalization built-in or was trained on raw features)
            if scaler is not None and not scaling_warned:
                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != state_seq.shape[1]:
                    print(f"⚠️ Scaler mismatch ({scaler.n_features_in_} vs {state_seq.shape[1]} features) - using raw features")
                    scaling_warned = True
            
            q_values = model.predict(input_tensor, verbose=0)[0]
            action = int(np.argmax(q_values))  # 0 short,1 hold,2 long
            
        elif hasattr(model, 'decide_action'):
            # PyTorch agent path
            seq_for_agent = state_seq  # shape (30, features)
            
            # Apply scaling if dimensions match
            if scaler is not None and not scaling_warned:
                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == seq_for_agent.shape[1]:
                    try:
                        seq_for_agent = scaler.transform(seq_for_agent)
                    except Exception as e:
                        print(f"⚠️ Scaling failed: {e}")
                        scaling_warned = True
                else:
                    if not scaling_warned:
                        print(f"⚠️ Scaler mismatch ({scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'unknown'} vs {seq_for_agent.shape[1]} features) - using raw features")
                        scaling_warned = True
            
            action, info = model.decide_action(seq_for_agent, mode='eval', minutes_to_close=float(df.iloc[idx].get('minutes_to_close', 180)))
            q_values = info.get('q_values', [0,0,0])
        else:
            print("Unknown model type; aborting evaluation.")
            return

        price_return = df.iloc[idx]['returns'] if 'returns' in df.columns else 0.0
        step_ret = position * price_return
        step_returns.append(step_ret)
        new_equity = equity_curve[-1] * (1 + step_ret)
        equity_curve.append(new_equity)

        # Update position based on action (mapping unify: 0 short, 1 hold, 2 long)
        new_position = {0: -1, 1: 0, 2: 1}.get(action, 0)
        position_changed = new_position != position
        if position_changed and position != 0:
            # Realize trade PnL of current trade
            trade_pnls.append(sum(current_trade_returns))
            current_trade_returns = []
        position = new_position
        if position != 0:
            current_trade_returns.append(step_ret)

        actions_taken.append({
            'index': idx,
            'datetime': df.iloc[idx]['datetime'],
            'action': action,
            'position': position,
            'q_values': q_values,
            'return': step_ret
        })

    # Close final open trade
    if current_trade_returns:
        trade_pnls.append(sum(current_trade_returns))

    metrics = compute_metrics(step_returns, trade_pnls, equity_curve)
    report_dir = Path('validation_output')
    report_dir.mkdir(exist_ok=True)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = report_dir / f'validation_report_{timestamp}.json'
    csv_path = report_dir / f'validation_actions_{timestamp}.csv'
    json_path.write_text(json.dumps({'model': model_desc, 'metrics': metrics}, indent=2))
    pd.DataFrame(actions_taken).to_csv(csv_path, index=False)
    print(f"Validation complete. Metrics saved to {json_path} | Actions -> {csv_path}")
    print(json.dumps(metrics, indent=2))


def compute_metrics(step_returns: List[float], trade_pnls: List[float], equity_curve: List[float]) -> Dict[str, Any]:
    arr = np.array(step_returns, dtype=float)
    trades = np.array(trade_pnls, dtype=float) if trade_pnls else np.array([])
    total_ret = equity_curve[-1] - 1.0
    mean_ret = arr.mean() if arr.size else 0.0
    std_ret = arr.std(ddof=1) if arr.size > 1 else 0.0
    sharpe = (mean_ret / std_ret) * np.sqrt(len(arr)) if std_ret > 0 else 0.0
    win_rate_steps = float(np.sum(arr > 0) / len(arr)) if arr.size else 0.0
    win_rate_trades = float(np.sum(trades > 0) / len(trades)) if trades.size else 0.0
    max_dd = _max_drawdown(np.array(equity_curve))
    avg_trade = trades.mean() if trades.size else 0.0
    return {
        'total_return': total_ret,
        'sharpe_intraday': sharpe,
        'steps': len(arr),
        'win_rate_steps': win_rate_steps,
        'num_trades': int(len(trades)),
        'win_rate_trades': win_rate_trades,
        'avg_trade_return': avg_trade,
        'max_drawdown': max_dd,
        'equity_final': equity_curve[-1]
    }


def _max_drawdown(equity: np.ndarray) -> float:
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    return float(drawdowns.min())


if __name__ == "__main__":
    evaluate()
