from functools import reduce
from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.tools import calculate_bollinger_bands, calculate_macd, calculate_obv, calculate_rsi, get_financial_metrics, get_prices, prices_to_df

import argparse
from datetime import datetime
import json

llm = ChatOpenAI(model="gpt-4o")

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]

##### Market Data Agent #####
def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing market data"""
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    end_date = data["end_date"] or datetime.now().strftime('%Y-%m-%d')
    if not data["start_date"]:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = end_date_obj.replace(month=end_date_obj.month - 3) if end_date_obj.month > 3 else \
            end_date_obj.replace(year=end_date_obj.year - 1, month=end_date_obj.month + 9)
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]

    # Get the historical price data
    prices = get_prices(
        ticker=data["ticker"], 
        start_date=start_date, 
        end_date=end_date,
    )

    # Get the financial metrics
    financial_metrics = get_financial_metrics(
        ticker=data["ticker"], 
        report_period=end_date, 
        period='ttm', 
        limit=1,
    )

    return {
        "messages": messages,
        "data": {
            **data, 
            "prices": prices, 
            "start_date": start_date, 
            "end_date": end_date,
            "financial_metrics": financial_metrics,
        }
    }

##### Quantitative Agent #####
def quant_agent(state: AgentState):
    """Analyzes technical indicators and generates trading signals."""
    show_reasoning = state["metadata"]["show_reasoning"]

    data = state["data"]
    prices = data["prices"]
    prices_df = prices_to_df(prices)
    
    # Calculate indicators
    # 1. MACD (Moving Average Convergence Divergence)
    macd_line, signal_line = calculate_macd(prices_df)
    
    # 2. RSI (Relative Strength Index)
    rsi = calculate_rsi(prices_df)
    
    # 3. Bollinger Bands
    upper_band, lower_band = calculate_bollinger_bands(prices_df)
    
    # 4. OBV (On-Balance Volume)
    obv = calculate_obv(prices_df)
    
    # 5. Calculate trend strength
    sma_20 = prices_df['close'].rolling(window=20).mean()
    sma_50 = prices_df['close'].rolling(window=50).mean()
    atr = prices_df['high'].rolling(window=14).max() - prices_df['low'].rolling(window=14).min()
    
    # Generate individual signals
    signals = []
    signal_weights = []  # Add weights to give more importance to stronger signals
    
    # Trend Analysis
    if len(sma_20) > 0 and len(sma_50) > 0:
        trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / atr.iloc[-1]
        if trend_strength > 0.5:  # Strong uptrend
            signals.append('bullish')
            signal_weights.append(2.0)  # Higher weight for strong trend
        elif trend_strength < -0.5:  # Strong downtrend
            signals.append('bearish')
            signal_weights.append(2.0)
        else:
            signals.append('neutral')
            signal_weights.append(1.0)
    else:
        signals.append('neutral')
        signal_weights.append(1.0)
    
    # MACD signal - Enhanced to consider trend strength
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        macd_trend = macd_line.diff().iloc[-5:].mean()  # Look at MACD momentum
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_trend > 0:
            signals.append('bullish')
            signal_weights.append(1.5 if abs(macd_trend) > macd_line.std() else 1.0)
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_trend < 0:
            signals.append('bearish')
            signal_weights.append(1.5 if abs(macd_trend) > macd_line.std() else 1.0)
        else:
            signals.append('neutral')
            signal_weights.append(1.0)
    else:
        signals.append('neutral')
        signal_weights.append(1.0)
    
    # RSI signal - More dynamic thresholds based on trend
    rsi_value = rsi.iloc[-1] if len(rsi) > 0 else 50
    if len(sma_20) > 0 and len(sma_50) > 0:
        if sma_20.iloc[-1] > sma_50.iloc[-1]:  # Uptrend
            if rsi_value < 40:  # More aggressive in uptrend
                signals.append('bullish')
                signal_weights.append(1.5)
            elif rsi_value > 80:
                signals.append('bearish')
                signal_weights.append(1.0)
            else:
                signals.append('neutral')
                signal_weights.append(1.0)
        else:  # Downtrend
            if rsi_value < 20:
                signals.append('bullish')
                signal_weights.append(1.0)
            elif rsi_value > 60:  # More aggressive in downtrend
                signals.append('bearish')
                signal_weights.append(1.5)
            else:
                signals.append('neutral')
                signal_weights.append(1.0)
    else:
        signals.append('neutral')
        signal_weights.append(1.0)
    
    # Bollinger Bands signal - Consider trend context
    if len(upper_band) > 0 and len(lower_band) > 0:
        current_price = prices_df['close'].iloc[-1]
        bb_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / prices_df['close'].rolling(window=20).mean().iloc[-1]
        
        if current_price < lower_band.iloc[-1] and bb_width > 0.05:  # Wide bands = stronger signal
            signals.append('bullish')
            signal_weights.append(1.5)
        elif current_price > upper_band.iloc[-1] and bb_width > 0.05:
            signals.append('bearish')
            signal_weights.append(1.5)
        else:
            # Look for price moves within bands
            price_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            if price_position < 0.2:  # Close to lower band
                signals.append('bullish')
                signal_weights.append(1.0)
            elif price_position > 0.8:  # Close to upper band
                signals.append('bearish')
                signal_weights.append(1.0)
            else:
                signals.append('neutral')
                signal_weights.append(1.0)
    else:
        signals.append('neutral')
        signal_weights.append(1.0)
    
    # OBV signal - Enhanced with volume trend
    if len(obv) >= 5:
        obv_slope = obv.diff().iloc[-5:].mean()
        obv_std = obv.diff().iloc[-20:].std()
        
        if obv_slope > obv_std:  # Strong volume trend up
            signals.append('bullish')
            signal_weights.append(1.5)
        elif obv_slope < -obv_std:  # Strong volume trend down
            signals.append('bearish')
            signal_weights.append(1.5)
        else:
            signals.append('neutral')
            signal_weights.append(1.0)
    else:
        signals.append('neutral')
        signal_weights.append(1.0)
    
    # Weighted signal calculation
    bullish_score = sum(w for s, w in zip(signals, signal_weights) if s == 'bullish')
    bearish_score = sum(w for s, w in zip(signals, signal_weights) if s == 'bearish')
    total_weight = sum(signal_weights)
    
    if bullish_score > bearish_score:
        overall_signal = 'bullish'
        confidence = bullish_score / total_weight
    elif bearish_score > bullish_score:
        overall_signal = 'bearish'
        confidence = bearish_score / total_weight
    else:
        overall_signal = 'neutral'
        confidence = 0.5
    
    # Add reasoning collection
    reasoning = {
        "Trend": {
            "signal": signals[0],
            "details": f"Trend strength: {trend_strength:.2f}"
        },
        "MACD": {
            "signal": signals[1],
            "details": f"MACD Line crossed {'above' if signals[1] == 'bullish' else 'below' if signals[1] == 'bearish' else 'neither above nor below'} Signal Line"
        },
        "RSI": {
            "signal": signals[2],
            "details": f"RSI is {rsi_value:.2f} ({'oversold' if signals[2] == 'bullish' else 'overbought' if signals[2] == 'bearish' else 'neutral'})"
        },
        "Bollinger": {
            "signal": signals[3],
            "details": f"Price is {'below lower band' if signals[3] == 'bullish' else 'above upper band' if signals[3] == 'bearish' else 'within bands'}"
        },
        "OBV": {
            "signal": signals[4],
            "details": f"OBV slope is {obv_slope:.2f} ({signals[4]})"
        }
    }
    
    # Generate the message content
    message_content = {
        "signal": overall_signal,
        "confidence": round(confidence, 2),
        "reasoning": {
            "Trend": reasoning["Trend"],
            "MACD": reasoning["MACD"],
            "RSI": reasoning["RSI"],
            "Bollinger": reasoning["Bollinger"],
            "OBV": reasoning["OBV"]
        }
    }

    # Create the quant message
    message = HumanMessage(
        content=str(message_content),  # Convert dict to string for message content
        name="quant_agent",
    )

    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Quant Agent")
    
    return {
        "messages": [message],
        "data": data,
    }

##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals."""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]  # Get the most recent metrics
    ticker = data["ticker"]
    
    # Initialize signals list for different fundamental aspects
    signals = []
    reasoning = {}
    
    # For ETFs like GLD, focus on price momentum and market trends
    if ticker in ["GLD", "IAU", "SGOL"]:  # Gold ETFs
        prices_df = prices_to_df(data["prices"])
        
        # Calculate moving averages
        sma_50 = prices_df['close'].rolling(window=50).mean()
        sma_200 = prices_df['close'].rolling(window=200).mean()
        current_price = prices_df['close'].iloc[-1]
        
        # Price above moving averages suggests bullish trend
        if current_price > sma_50.iloc[-1] and current_price > sma_200.iloc[-1]:
            signals.append("bullish")
            reasoning["trend"] = "Price above both 50 and 200-day moving averages"
        elif current_price < sma_50.iloc[-1] and current_price < sma_200.iloc[-1]:
            signals.append("bearish")
            reasoning["trend"] = "Price below both 50 and 200-day moving averages"
        else:
            signals.append("neutral")
            reasoning["trend"] = "Mixed signals from moving averages"
        
        # Volume analysis with lower threshold
        avg_volume = prices_df['volume'].rolling(window=10).mean()
        if prices_df['volume'].iloc[-1] > avg_volume.iloc[-1] * 1.2:
            if current_price > prices_df['close'].iloc[-2]:
                signals.append("bullish")
                reasoning["volume"] = "Above-average volume on price increase"
            else:
                signals.append("bearish")
                reasoning["volume"] = "Above-average volume on price decrease"
        else:
            signals.append("neutral")
            reasoning["volume"] = "Normal trading volume"
        
        # Price momentum with longer window
        returns = prices_df['close'].pct_change()
        momentum = returns.rolling(window=20).mean()
        volatility = returns.rolling(window=20).std()
        
        # Adjust momentum threshold based on volatility
        momentum_threshold = volatility.iloc[-1] * 0.5
        
        if momentum.iloc[-1] > momentum_threshold:
            signals.append("bullish")
            reasoning["momentum"] = f"Strong positive momentum: {momentum.iloc[-1]:.2%}"
        elif momentum.iloc[-1] < -momentum_threshold:
            signals.append("bearish")
            reasoning["momentum"] = f"Strong negative momentum: {momentum.iloc[-1]:.2%}"
        else:
            signals.append("neutral")
            reasoning["momentum"] = "Neutral momentum"
            
    else:  # For regular stocks
        # Regular stock analysis (existing code)
        if metrics.get("returnOnEquity") and metrics["returnOnEquity"] > 0.15:
            signals.append("bullish")
            reasoning["profitability"] = f"Strong ROE of {metrics['returnOnEquity']*100:.1f}%"
        elif metrics.get("returnOnEquity") and metrics["returnOnEquity"] < 0.05:
            signals.append("bearish")
            reasoning["profitability"] = f"Weak ROE of {metrics['returnOnEquity']*100:.1f}%"
        else:
            signals.append("neutral")
            reasoning["profitability"] = "Average profitability metrics"
        
        if metrics.get("forwardPE") and metrics["forwardPE"] < 15:
            signals.append("bullish")
            reasoning["valuation"] = f"Attractive Forward P/E of {metrics['forwardPE']:.1f}"
        elif metrics.get("forwardPE") and metrics["forwardPE"] > 30:
            signals.append("bearish")
            reasoning["valuation"] = f"High Forward P/E of {metrics['forwardPE']:.1f}"
        else:
            signals.append("neutral")
            reasoning["valuation"] = "Average valuation metrics"
        
        if metrics.get("debtToEquity") and metrics["debtToEquity"] < 0.5:
            signals.append("bullish")
            reasoning["financial_health"] = f"Low Debt/Equity ratio of {metrics['debtToEquity']:.2f}"
        elif metrics.get("debtToEquity") and metrics["debtToEquity"] > 2:
            signals.append("bearish")
            reasoning["financial_health"] = f"High Debt/Equity ratio of {metrics['debtToEquity']:.2f}"
        else:
            signals.append("neutral")
            reasoning["financial_health"] = "Average financial health metrics"
    
    # Determine overall signal
    bullish_signals = signals.count("bullish")
    bearish_signals = signals.count("bearish")
    
    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"
    
    # Calculate confidence level based on the proportion of indicators agreeing
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals
    
    # Generate the message content
    message_content = {
        "signal": overall_signal,
        "confidence": round(confidence, 2),
        "reasoning": reasoning
    }

    # Create the fundamentals message
    message = HumanMessage(
        content=str(message_content),
        name="fundamentals_agent",
    )
    
    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Fundamentals Agent")
    
    return {
        "messages": [message],
        "data": data,
    }

##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Evaluates portfolio risk and sets position limits"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    prices = state["data"]["prices"]
    prices_df = prices_to_df(prices)
    
    # Calculate volatility and trend metrics
    returns = prices_df['close'].pct_change()
    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
    
    # Calculate trend strength
    sma_20 = prices_df['close'].rolling(window=20).mean()
    sma_50 = prices_df['close'].rolling(window=50).mean()
    atr = prices_df['high'].rolling(window=14).max() - prices_df['low'].rolling(window=14).min()
    
    if len(sma_20) > 0 and len(sma_50) > 0 and len(atr) > 0:
        trend_strength = abs((sma_20.iloc[-1] - sma_50.iloc[-1]) / atr.iloc[-1])
    else:
        trend_strength = 0
    
    # Find the quant message by looking for the message with name "quant_agent"
    quant_message = next(msg for msg in state["messages"] if msg.name == "quant_agent")
    fundamentals_message = next(msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    
    # Extract confidence levels from messages
    quant_data = eval(quant_message.content)
    fundamentals_data = eval(fundamentals_message.content)
    
    quant_confidence = quant_data.get('confidence', 0.5)
    fundamentals_confidence = fundamentals_data.get('confidence', 0.5)
    
    # Calculate base position size based on portfolio value
    total_value = portfolio['cash'] + (portfolio['stock'] * prices_df['close'].iloc[-1])
    
    # Adjust position size based on volatility
    vol_factor = max(0.2, min(1.0, 0.3 / volatility))  # Reduce position size when volatility is high
    
    # Adjust position size based on trend strength
    trend_factor = min(1.5, 1.0 + trend_strength)  # Increase position size in strong trends
    
    # Adjust position size based on signal confidence
    confidence_factor = (quant_confidence + fundamentals_confidence) / 2
    
    # Calculate final position size
    base_position = 0.5  # Start with 50% of portfolio
    max_position_size = base_position * vol_factor * trend_factor * confidence_factor
    max_position_size = min(0.95, max_position_size)  # Cap at 95% of portfolio
    
    # Calculate risk score (1-10)
    risk_factors = [
        volatility * 10,  # Higher volatility = higher risk
        (1 - vol_factor) * 5,  # Lower vol_factor = higher risk
        (trend_factor - 1) * 3,  # Stronger trend = slightly higher risk
        (1 - confidence_factor) * 2  # Lower confidence = higher risk
    ]
    risk_score = min(10, max(1, sum(risk_factors) / len(risk_factors) * 10))
    
    # Determine trading action based on signals and current position
    quant_signal = quant_data.get('signal', 'neutral')
    fundamentals_signal = fundamentals_data.get('signal', 'neutral')
    
    if quant_signal == fundamentals_signal:
        trading_action = quant_signal
    elif quant_confidence > fundamentals_confidence:
        trading_action = quant_signal
    else:
        trading_action = fundamentals_signal
    
    # Convert trading_action to buy/sell/hold
    if trading_action == 'bullish':
        trading_action = 'buy'
    elif trading_action == 'bearish':
        trading_action = 'sell'
    else:
        trading_action = 'hold'
    
    # Create the message content
    message_content = {
        "max_position_size": round(max_position_size, 2),
        "risk_score": round(risk_score, 1),
        "trading_action": trading_action,
        "reasoning": f"Position size adjusted for volatility ({vol_factor:.2f}), "
                    f"trend strength ({trend_factor:.2f}), and "
                    f"signal confidence ({confidence_factor:.2f})"
    }
    
    message = HumanMessage(
        content=str(message_content),
        name="risk_management_agent",
    )
    
    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Risk Management Agent")
    
    return {"messages": state["messages"] + [message]}

##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # Get the quant agent, fundamentals agent, and risk management agent messages
    quant_message = next(msg for msg in state["messages"] if msg.name == "quant_agent")
    fundamentals_message = next(msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    risk_message = next(msg for msg in state["messages"] if msg.name == "risk_management_agent")

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions.
                Your job is to make a trading decision based on the team's analysis.
                Provide the following in your output:
                - "action": "buy" | "sell" | "hold",
                - "quantity": <positive integer>
                - "reasoning": <concise explanation of the decision>
                Only buy if you have available cash.
                The quantity that you buy must be less than or equal to the max position size.
                Only sell if you have shares in the portfolio to sell.
                The quantity that you sell must be less than or equal to the current position."""
            ),
            (
                "human",
                """Based on the team's analysis below, make your trading decision.

                Quant Analysis Trading Signal: {quant_message}
                Fundamental Analysis Trading Signal: {fundamentals_message}
                Risk Management Trading Signal: {risk_message}

                Here is the current portfolio:
                Portfolio:
                Cash: {portfolio_cash}
                Current Position: {portfolio_stock} shares

                Only include the action, quantity, and reasoning in your output as JSON.  Do not include any JSON markdown.

                Remember, the action must be either buy, sell, or hold.
                You can only buy if you have available cash.
                You can only sell if you have shares in the portfolio to sell.
                """
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "quant_message": quant_message.content, 
            "fundamentals_message": fundamentals_message.content,
            "risk_message": risk_message.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"]
        }
    )
    # Invoke the LLM
    result = llm.invoke(prompt)

    # Create the portfolio management message
    message = HumanMessage(
        content=result.content,
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}

def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    if isinstance(output, (dict, list)):
        # If output is already a dictionary or list, just pretty print it
        print(json.dumps(output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            print(output)
    print("=" * 48)

##### Run the Hedge Fund #####
def run_hedge_fund(ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False):
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            }
        },
    )
    return final_state["messages"][-1].content

# Define the new workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("quant_agent", quant_agent)
workflow.add_node("fundamentals_agent", fundamentals_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("portfolio_management_agent", portfolio_management_agent)

# Define the workflow
workflow.set_entry_point("market_data_agent")
workflow.add_edge("market_data_agent", "quant_agent")
workflow.add_edge("market_data_agent", "fundamentals_agent")
workflow.add_edge("quant_agent", "risk_management_agent")
workflow.add_edge("fundamentals_agent", "risk_management_agent")
workflow.add_edge("risk_management_agent", "portfolio_management_agent")
workflow.add_edge("portfolio_management_agent", END)

app = workflow.compile()

# Add this at the bottom of the file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD). Defaults to 3 months before end date')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD). Defaults to today')
    parser.add_argument('--show-reasoning', action='store_true', help='Show reasoning from each agent')
    
    args = parser.parse_args()
    
    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")
    
    if args.end_date:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")
    
    # Sample portfolio - you might want to make this configurable too
    portfolio = {
        "cash": 100000.0,  # $100,000 initial cash
        "stock": 0         # No initial stock position
    }
    
    result = run_hedge_fund(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning
    )
    print("\nFinal Result:")
    print(result)