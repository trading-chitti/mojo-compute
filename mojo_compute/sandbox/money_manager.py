"""ML Money Manager - Signal selection and position sizing.

Uses XGBoost binary classifier to decide TRADE vs SKIP,
then Mojo Kelly Criterion for position sizing.

Stage 1 (bootstrap): Uses signal confidence + basic features.
Stage 2 (trained): Uses full 25-feature XGBoost model after training.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import config
from .mojo_wrappers import kelly_position_size_pct, current_drawdown

logger = logging.getLogger(__name__)

# Try to load trained model
_model = None
_model_features: list = []

def _load_model():
    """Load the trained XGBoost model if available."""
    global _model, _model_features
    try:
        import joblib
        model_dir = config.ML_MODEL_DIR
        # Find the most recent model file
        model_files = sorted(
            [f for f in os.listdir(model_dir) if f.startswith("money_manager") and f.endswith(".joblib")],
            reverse=True,
        )
        if model_files:
            path = os.path.join(model_dir, model_files[0])
            data = joblib.load(path)
            _model = data.get("model")
            _model_features = data.get("features", [])
            logger.info("✅ Loaded ML money manager: %s (%d features)", model_files[0], len(_model_features))
        else:
            logger.info("No trained money manager model found, using rule-based fallback")
    except Exception as e:
        logger.warning("Could not load ML model: %s", e)

_load_model()


class MoneyManager:
    """ML-powered trade selection and position sizing.

    Decision pipeline:
    1. Extract features from signal + session state
    2. ML model predicts TRADE probability (or rule-based fallback)
    3. If score > threshold -> TRADE, else -> SKIP
    4. Position size via Mojo Kelly Criterion
    """

    def __init__(self):
        self.trade_threshold = config.ML_TRADE_THRESHOLD

    def evaluate_signal(
        self,
        signal: Dict[str, Any],
        session: Dict[str, Any],
        open_positions: int,
        allocated_capital: float,
    ) -> Tuple[str, float, float, float, Dict[str, Any], Dict[str, Any]]:
        """Evaluate a signal and decide TRADE or SKIP.

        Args:
            signal: Signal dict from intraday-engine.
            session: Active session dict.
            open_positions: Number of currently open trades.
            allocated_capital: Capital currently in open positions.

        Returns:
            (decision, ml_score, position_pct, leverage, features_dict, reasoning_dict)
            decision: "TRADE" or "SKIP"
            ml_score: ML model's probability score
            position_pct: Recommended position size as % of capital
            leverage: Dynamic leverage multiplier for this trade
            features_dict: Feature values used for decision
            reasoning_dict: Human-readable reasoning
        """
        features = self._extract_features(signal, session, open_positions, allocated_capital)

        # Get ML score
        if _model is not None and _model_features:
            ml_score = self._ml_predict(features)
        else:
            ml_score = self._rule_based_score(features)

        # Decision
        decision = "TRADE" if ml_score > self.trade_threshold else "SKIP"

        # Position sizing + leverage (only if TRADE)
        position_pct = 0.0
        leverage = 0.0
        if decision == "TRADE":
            position_pct = self._calculate_position_size(features, ml_score, session)
            leverage = self._calculate_leverage(features, ml_score, session)

        # Build reasoning
        reasoning = self._build_reasoning(features, ml_score, decision, position_pct, leverage)

        return decision, ml_score, position_pct, leverage, features, reasoning

    def _extract_features(
        self,
        signal: Dict[str, Any],
        session: Dict[str, Any],
        open_positions: int,
        allocated_capital: float,
    ) -> Dict[str, Any]:
        """Extract 25 features from signal and session state."""
        current_capital = float(session.get("current_capital", 0))
        initial_capital = float(session.get("initial_capital", 0))
        peak_capital = float(session.get("peak_capital", 0))

        entry = float(signal.get("entry_price", 0))
        target = float(signal.get("target_price", 0))
        stop_loss = float(signal.get("stop_loss", 0))
        confidence = float(signal.get("confidence", 0.5))

        # Risk:reward
        risk_reward = 2.0
        if entry > 0 and stop_loss > 0 and target > 0:
            if signal.get("signal_type") == "CALL":
                reward = target - entry
                risk = entry - stop_loss
            else:
                reward = entry - target
                risk = stop_loss - entry
            if risk > 0:
                risk_reward = reward / risk

        # Drawdown
        dd_pct = current_drawdown(current_capital, peak_capital) if peak_capital > 0 else 0.0

        # Capital remaining
        available = current_capital - allocated_capital
        capital_remaining_pct = (available / initial_capital * 100.0) if initial_capital > 0 else 100.0

        # Session win/loss stats
        total_trades = session.get("total_trades", 0)
        wins = session.get("winning_trades", 0)
        losses = session.get("losing_trades", 0)
        wr = (wins / total_trades * 100.0) if total_trades > 0 else 50.0

        # Signal metadata
        metadata = signal.get("metadata", {}) or {}
        rsi = float(metadata.get("rsi", signal.get("rsi", 50)))
        macd = float(metadata.get("macd", signal.get("macd", 0)))
        volume_ratio = float(metadata.get("volume_ratio", signal.get("volume_ratio", 1.0)))

        # Prediction features from intraday engine
        pred_features = signal.get("prediction_features", {}) or {}
        bb_position = float(pred_features.get("bb_position", 0.5))
        atr_pct = float(pred_features.get("atr_pct", 1.5))

        # News sentiment
        sentiment = float(signal.get("recent_news_sentiment", signal.get("sentiment", 0)))

        # Time features
        now = datetime.now(config.IST)
        minutes_since_open = max(0, (now.hour - 9) * 60 + (now.minute - 15))

        # Is CALL or PUT
        is_call = 1 if signal.get("signal_type") == "CALL" else 0

        # SL and target as percentages
        sl_pct = abs(entry - stop_loss) / entry * 100.0 if entry > 0 else 1.5
        target_pct = abs(target - entry) / entry * 100.0 if entry > 0 else 3.0

        return {
            # Signal features
            "confidence": confidence,
            "rsi": rsi,
            "macd": macd,
            "bb_position": bb_position,
            "volume_ratio": volume_ratio,
            "atr_pct": atr_pct,
            "sentiment": sentiment,
            "is_call": is_call,
            "risk_reward": risk_reward,
            "sl_pct": sl_pct,
            "target_pct": target_pct,
            "entry_price": entry,
            # Session state features
            "capital_remaining_pct": capital_remaining_pct,
            "drawdown_pct": dd_pct,
            "open_positions": open_positions,
            "total_trades": total_trades,
            "win_rate": wr,
            "winning_streak": 0,  # TODO: track streaks
            "losing_streak": 0,
            # Time features
            "minutes_since_open": minutes_since_open,
            "is_first_hour": 1 if minutes_since_open <= 60 else 0,
            "is_last_hour": 1 if minutes_since_open >= 300 else 0,
            # Position context
            "pnl_pct": float(session.get("total_pnl_pct", 0)),
            "signals_received": session.get("signals_received", 0),
            "skip_rate": (session.get("signals_skipped", 0) / max(1, session.get("signals_received", 1))) * 100,
        }

    def _ml_predict(self, features: Dict[str, Any]) -> float:
        """Use trained XGBoost model to predict trade probability."""
        try:
            feature_values = [features.get(f, 0) for f in _model_features]
            X = np.array([feature_values])
            proba = _model.predict_proba(X)[0]
            return float(proba[1])  # probability of class 1 (TRADE)
        except Exception as e:
            logger.warning("ML prediction failed, falling back to rule-based: %s", e)
            return self._rule_based_score(features)

    def _rule_based_score(self, features: Dict[str, Any]) -> float:
        """Rule-based scoring when no ML model is available.

        Multi-factor scoring that differentiates CALL vs PUT quality,
        uses session win rate to adapt, and penalizes poor conditions.

        Score ranges:
          0.70+ → Strong trade (large position via Kelly)
          0.55-0.70 → Moderate trade (medium position)
          < 0.55 → SKIP (below threshold)
        """
        score = features["confidence"]
        is_call = features["is_call"]

        # ── Risk:Reward quality ──
        rr = features["risk_reward"]
        if rr >= 3.0:
            score += 0.08   # excellent R:R
        elif rr >= 2.5:
            score += 0.05
        elif rr >= 2.0:
            score += 0.02
        elif rr < 1.5:
            score -= 0.12   # poor R:R, aggressive penalty

        # ── CALL vs PUT asymmetry ──
        # PUT signals historically have lower success rate (~20% vs 35%)
        # Require higher confidence for PUTs
        if not is_call:
            score -= 0.05   # PUT penalty (need stronger signal)
            if features["confidence"] < 0.70:
                score -= 0.05  # extra penalty for low-confidence PUTs

        # ── Session performance feedback ──
        # If win rate is high, be slightly more aggressive
        # If losing, tighten the filter
        wr = features["win_rate"]
        total_trades = features["total_trades"]
        if total_trades >= 5:
            if wr >= 60:
                score += 0.05   # winning streak → be bolder
            elif wr < 35:
                score -= 0.08   # losing → be cautious

        # ── Drawdown-aware scaling ──
        dd = features["drawdown_pct"]
        if dd > 8.0:
            score -= 0.20   # near max drawdown → very conservative
        elif dd > 5.0:
            score -= 0.12
        elif dd > 3.0:
            score -= 0.05

        # ── Position concentration ──
        if features["open_positions"] >= 4:
            score -= 0.08   # almost at limit
        elif features["open_positions"] >= 3:
            score -= 0.03

        # ── Technical indicator confirmation ──
        rsi = features["rsi"]
        volume = features["volume_ratio"]
        bb = features["bb_position"]

        # Volume surge confirms signal
        if volume >= 2.5:
            score += 0.07
        elif volume >= 1.5:
            score += 0.03
        elif volume < 0.5:
            score -= 0.06  # no volume = unreliable

        # RSI alignment with signal direction
        if is_call:
            if rsi < 30:
                score += 0.07    # oversold → strong CALL
            elif rsi < 40:
                score += 0.03
            elif rsi > 75:
                score -= 0.08    # overbought → bad CALL
        else:
            if rsi > 70:
                score += 0.07    # overbought → strong PUT
            elif rsi > 60:
                score += 0.03
            elif rsi < 25:
                score -= 0.08    # oversold → bad PUT

        # Bollinger Band position
        if is_call and bb < 0.2:
            score += 0.04       # near lower band → good CALL entry
        elif not is_call and bb > 0.8:
            score += 0.04       # near upper band → good PUT entry

        # ── Sentiment alignment ──
        sentiment = features["sentiment"]
        if is_call and sentiment > 0.3:
            score += 0.04
        elif is_call and sentiment < -0.3:
            score -= 0.04       # negative sentiment → bad for CALL
        elif not is_call and sentiment < -0.3:
            score += 0.04
        elif not is_call and sentiment > 0.3:
            score -= 0.04       # positive sentiment → bad for PUT

        # ── Time-of-day effects ──
        mins = features["minutes_since_open"]
        if mins < 15:
            score -= 0.12       # opening volatility → avoid
        elif mins < 30:
            score -= 0.05       # still choppy
        elif 60 <= mins <= 240:
            score += 0.02       # prime trading hours
        elif mins > 345:
            score -= 0.08       # last 15 min → avoid

        # ── Stop loss too tight ──
        if features["sl_pct"] < 1.0:
            score -= 0.08       # SL will get hit by noise
        elif features["atr_pct"] > 0 and features["sl_pct"] < features["atr_pct"] * 0.8:
            score -= 0.05       # SL tighter than ATR → likely hit

        # ── Consecutive loss penalty (V2) ──
        consec_losses = features.get("consecutive_losses", 0)
        if consec_losses >= 3:
            score -= 0.20       # 3+ losses → strong penalty
        elif consec_losses >= 2:
            score -= 0.10       # 2 losses → moderate penalty

        # ── VWAP alignment (V2) ──
        vwap_distance = features.get("vwap_distance_pct", 0)
        if is_call and vwap_distance > 0.3:
            score += 0.05       # Price above VWAP confirms CALL
        elif is_call and vwap_distance < -0.5:
            score -= 0.05       # Price below VWAP contradicts CALL
        elif not is_call and vwap_distance < -0.3:
            score += 0.05       # Price below VWAP confirms PUT
        elif not is_call and vwap_distance > 0.5:
            score -= 0.05       # Price above VWAP contradicts PUT

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _calculate_position_size(
        self,
        features: Dict[str, Any],
        ml_score: float,
        session: Dict[str, Any],
    ) -> float:
        """Calculate position size using Mojo Kelly Criterion with volatility adjustment."""
        dd_pct = features["drawdown_pct"]
        confidence = features["confidence"]
        risk_reward = features["risk_reward"]
        max_dd = float(session.get("max_daily_drawdown_pct", config.DEFAULT_MAX_DAILY_DRAWDOWN_PCT))
        max_pos = float(session.get("max_position_pct", config.DEFAULT_MAX_POSITION_PCT))

        position_pct = kelly_position_size_pct(
            ml_score,
            confidence,
            risk_reward,
            dd_pct,
            max_dd,
            5.0,  # min 5%
            max_pos,
        )

        # Volatility regime adjustment (V2)
        vol_percentile = features.get("volatility_percentile", 50)
        if vol_percentile > 75:
            position_pct *= 0.6   # HIGH vol: reduce 40%
        elif vol_percentile < 25:
            position_pct = min(position_pct * 1.2, max_pos)  # LOW vol: increase 20%, capped

        return position_pct

    def _calculate_leverage(
        self,
        features: Dict[str, Any],
        ml_score: float,
        session: Dict[str, Any],
    ) -> float:
        """Calculate dynamic leverage based on ML score and conditions.

        Scaling logic:
          - ML score at threshold (0.55) → 30% of max leverage
          - ML score at 0.85+ → 100% of max leverage
          - Drawdown > 5% → leverage reduced by 40%
          - Losing streak (win rate < 35%) → leverage reduced by 30%
          - PUT signals → leverage capped at 70% of max (higher risk)
        """
        max_lev = float(session.get("leverage_multiplier", config.DEFAULT_LEVERAGE))
        min_lev = max(1.0, max_lev * 0.2)  # floor at 20% of max or 1x

        # Score-based scaling: maps [threshold, 1.0] → [0.3, 1.0]
        score_range = 1.0 - self.trade_threshold
        score_above = max(0.0, ml_score - self.trade_threshold)
        score_factor = min(1.0, score_above / score_range) if score_range > 0 else 0.5
        leverage_pct = 0.3 + 0.7 * (score_factor ** 0.8)

        # Drawdown dampening
        dd = features.get("drawdown_pct", 0)
        if dd > 7:
            leverage_pct *= 0.4   # severe drawdown → heavy reduction
        elif dd > 5:
            leverage_pct *= 0.6
        elif dd > 3:
            leverage_pct *= 0.8

        # Session win rate feedback
        if features.get("total_trades", 0) >= 5:
            wr = features.get("win_rate", 50)
            if wr < 35:
                leverage_pct *= 0.7   # losing → cut leverage
            elif wr >= 60:
                leverage_pct = min(1.0, leverage_pct * 1.1)  # winning → slight boost

        # PUT signals get capped leverage (historically riskier)
        if not features.get("is_call", 1):
            leverage_pct = min(leverage_pct, 0.70)

        # Risk:reward quality
        rr = features.get("risk_reward", 2.0)
        if rr < 1.5:
            leverage_pct *= 0.7   # poor R:R → don't amplify losses
        elif rr >= 3.0:
            leverage_pct = min(1.0, leverage_pct * 1.1)

        leverage = max(min_lev, min(max_lev, max_lev * leverage_pct))
        return round(leverage, 1)

    def _build_reasoning(
        self,
        features: Dict[str, Any],
        ml_score: float,
        decision: str,
        position_pct: float,
        leverage: float = 0.0,
    ) -> Dict[str, Any]:
        """Build human-readable reasoning for the ML decision."""
        factors = []

        # ML score
        if ml_score > 0.7:
            factors.append({"factor": "High ML score", "impact": "+", "value": f"{ml_score:.2f}"})
        elif ml_score < 0.45:
            factors.append({"factor": "Low ML score", "impact": "-", "value": f"{ml_score:.2f}"})

        # Signal confidence
        conf = features["confidence"]
        if conf >= 0.80:
            factors.append({"factor": "Strong signal confidence", "impact": "+", "value": f"{conf:.0%}"})
        elif conf < 0.60:
            factors.append({"factor": "Weak signal confidence", "impact": "-", "value": f"{conf:.0%}"})

        # Risk:Reward
        rr = features["risk_reward"]
        if rr >= 2.5:
            factors.append({"factor": "Excellent risk:reward", "impact": "+", "value": f"{rr:.1f}:1"})
        elif rr < 1.5:
            factors.append({"factor": "Poor risk:reward", "impact": "-", "value": f"{rr:.1f}:1"})

        # CALL/PUT type
        if not features["is_call"]:
            factors.append({"factor": "PUT signal (higher bar)", "impact": "-", "value": "PUT"})

        # Drawdown
        dd = features["drawdown_pct"]
        if dd > 5:
            factors.append({"factor": "Elevated drawdown", "impact": "-", "value": f"{dd:.1f}%"})

        # Volume
        vol = features["volume_ratio"]
        if vol >= 2.0:
            factors.append({"factor": "Strong volume confirmation", "impact": "+", "value": f"{vol:.1f}x"})
        elif vol < 0.5:
            factors.append({"factor": "Weak volume", "impact": "-", "value": f"{vol:.1f}x"})

        # RSI
        rsi = features["rsi"]
        is_call = features["is_call"]
        if is_call and rsi < 35:
            factors.append({"factor": "RSI oversold (bullish)", "impact": "+", "value": f"{rsi:.0f}"})
        elif not is_call and rsi > 65:
            factors.append({"factor": "RSI overbought (bearish)", "impact": "+", "value": f"{rsi:.0f}"})
        elif is_call and rsi > 75:
            factors.append({"factor": "RSI overbought (bad for CALL)", "impact": "-", "value": f"{rsi:.0f}"})

        # Win rate feedback
        if features["total_trades"] >= 5:
            wr = features["win_rate"]
            if wr >= 60:
                factors.append({"factor": "Session winning streak", "impact": "+", "value": f"{wr:.0f}%"})
            elif wr < 35:
                factors.append({"factor": "Session losing, reducing size", "impact": "-", "value": f"{wr:.0f}%"})

        # Time of day
        mins = features["minutes_since_open"]
        if mins < 15:
            factors.append({"factor": "Opening volatility (risky)", "impact": "-", "value": f"{mins}m"})
        elif 60 <= mins <= 240:
            factors.append({"factor": "Prime trading hours", "impact": "+", "value": f"{mins}m"})

        # Sentiment
        sentiment = features["sentiment"]
        if abs(sentiment) > 0.3:
            aligned = (is_call and sentiment > 0) or (not is_call and sentiment < 0)
            factors.append({
                "factor": "News sentiment " + ("aligned" if aligned else "against"),
                "impact": "+" if aligned else "-",
                "value": f"{sentiment:+.2f}",
            })

        # Leverage factor in reasoning
        if leverage > 0:
            max_lev = 10.0  # default, actual max from session not available here
            if leverage >= max_lev * 0.9:
                factors.append({"factor": "Full leverage (high conviction)", "impact": "+", "value": f"{leverage:.1f}x"})
            elif leverage <= max_lev * 0.4:
                factors.append({"factor": "Reduced leverage (caution)", "impact": "-", "value": f"{leverage:.1f}x"})

        return {
            "decision": decision,
            "ml_score": ml_score,
            "threshold": self.trade_threshold,
            "position_pct": position_pct,
            "leverage": leverage,
            "allocated_inr": position_pct,  # will be filled in server
            "model_type": "xgboost" if _model is not None else "rule_based",
            "signal_type": "CALL" if is_call else "PUT",
            "factors": factors,
        }
