import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from flask import Flask, render_template
import matplotlib
matplotlib.use('Agg')  # Required for running in Flask
import matplotlib.pyplot as plt
from matplotlib import gridspec
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

# --- Configuration ---
TICKER = "BABA"
MIN_DATA_POINTS = 400
PREDICTION_DAYS = 20  # Predict price 20 days ahead

def create_plot(stock_data, future_prices, model_r2):
    """Create matplotlib plot and return as base64 encoded image"""
    plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # Price and Moving Averages
    ax0 = plt.subplot(gs[0])
    ax0.plot(stock_data.index, stock_data['Close'], label='Price', color='blue')
    ax0.plot(stock_data.index, stock_data['EMA_50'], label='50-EMA', color='orange')
    ax0.plot(stock_data.index, stock_data['SMA_200'], label='200-SMA', color='red')
    
    # Plot future price prediction if available
    if len(future_prices) > 0:
        future_dates = pd.date_range(start=stock_data.index[-1], periods=PREDICTION_DAYS+1)[1:]
        ax0.plot(future_dates, future_prices, '--', label=f'Predicted (next {PREDICTION_DAYS} days)', color='purple')
        ax0.plot(future_dates[-1], future_prices[-1], 'o', markersize=8, color='purple')
        ax0.text(0.02, 0.02, f'Model R²: {model_r2:.3f}', transform=ax0.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    ax0.set_title(f"{TICKER} Technical Analysis with Price Prediction")
    ax0.legend()
    ax0.grid()
    
    # RSI
    ax1 = plt.subplot(gs[1])
    ax1.plot(stock_data.index, stock_data['RSI'], label='RSI(14)', color='purple')
    ax1.axhline(70, color='red', linestyle='--')
    ax1.axhline(30, color='green', linestyle='--')
    ax1.legend()
    ax1.grid()
    
    # MACD
    ax2 = plt.subplot(gs[2])
    ax2.plot(stock_data.index, stock_data['MACD'], label='MACD', color='blue')
    ax2.plot(stock_data.index, stock_data['Signal'], label='Signal', color='red')
    ax2.legend()
    ax2.grid()
    
    # Prediction details
    ax3 = plt.subplot(gs[3])
    if len(future_prices) > 0:
        days = np.arange(1, PREDICTION_DAYS+1)
        ax3.bar(days, future_prices - stock_data['Close'].iloc[-1], 
               color=['green' if x > 0 else 'red' for x in (future_prices - stock_data['Close'].iloc[-1])])
        ax3.axhline(0, color='black')
        ax3.set_title(f"Predicted Price Change Over Next {PREDICTION_DAYS} Days (R² = {model_r2:.2f})")
        ax3.set_xlabel('Days from now')
        ax3.set_ylabel('Price Change ($)')
        ax3.grid()
    
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    # Encode plot to base64 for HTML
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    return plot_data

# --- Technical Indicators ---
def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# --- Price Prediction ---
def predict_future_prices(data, days=10):
    """Predict future prices using linear regression on recent trend"""
    recent_data = data['Close'].tail(30).values.reshape(-1, 1)  # Use last 30 days
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R² score for the model
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    future_days = np.arange(len(recent_data), len(recent_data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    return predictions.flatten(), r2

# --- Fetch Stock Data ---
def get_stock_data(ticker):
    try:
        data = yf.download(ticker, period="2y")  # Extended period for analysis
        if len(data) < MIN_DATA_POINTS:
            raise ValueError(f"Insufficient data points ({len(data)} < {MIN_DATA_POINTS})")
        
        data['EMA_50'] = calculate_ema(data, 50)
        data['SMA_200'] = data['Close'].rolling(200).mean()
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['Signal'] = calculate_macd(data)
        
        # Add future price predictions and R² score
        future_prices, r2_score = predict_future_prices(data, PREDICTION_DAYS)
        return data.dropna(), future_prices, r2_score
    except Exception as e:
        print(f"Data download error: {e}")
        return pd.DataFrame(), [], 0

def generate_recommendation(latest_close, ema_50, sma_200, current_rsi, current_macd, current_signal, price_change, model_r2):
    """Generate trading recommendation based on analysis"""
    recommendation = {
        'current_price': f"{latest_close:.2f}",
        'predicted_price': f"{latest_close * (1 + price_change/100):.2f}" if price_change else "N/A",
        'price_change': f"{price_change:.1f}%" if price_change else "N/A",
        'rsi': f"{current_rsi:.1f}",
        'macd': f"{current_macd:.2f}",
        'macd_signal': "Bullish" if current_macd > current_signal else "Bearish",
        'trend': "Bullish" if latest_close > ema_50 > sma_200 else "Bearish",
        'ema_50': f"{ema_50:.2f}",
        'sma_200': f"{sma_200:.2f}",
        'model_r2': f"{model_r2:.4f}",
        'confidence': "High" if model_r2 > 0.7 else ("Medium" if model_r2 > 0.4 else "Low"),
        'action': "HOLD",
        'action_color': "warning",
        'action_icon': "bi-arrow-repeat",
        'reason': "Standard technical analysis",
        'prediction_available': bool(price_change)
    }
    
    if price_change and model_r2 > 0.4:  # Only consider predictions if model has reasonable fit
        recommendation['reason'] = f"Expected price change of {price_change:.1f}% in {PREDICTION_DAYS} days"
        
        if price_change > 5:  # Strong expected gain
            recommendation.update({
                'action': "STRONG BUY",
                'action_color': "success",
                'action_icon': "bi-arrow-up-circle"
            })
        elif price_change > 2:
            recommendation.update({
                'action': "BUY",
                'action_color': "success",
                'action_icon': "bi-arrow-up"
            })
        elif price_change < -5:  # Strong expected loss
            recommendation.update({
                'action': "STRONG SELL",
                'action_color': "danger",
                'action_icon': "bi-arrow-down-circle"
            })
        elif price_change < -2:
            recommendation.update({
                'action': "SELL",
                'action_color': "danger",
                'action_icon': "bi-arrow-down"
            })
    else:
        if latest_close > ema_50 > sma_200:
            recommendation.update({
                'action': "BUY",
                'action_color': "success",
                'action_icon': "bi-arrow-up",
                'reason': "Price above both 50EMA and 200SMA"
            })
        elif latest_close < ema_50:
            recommendation.update({
                'action': "SELL",
                'action_color': "danger",
                'action_icon': "bi-arrow-down",
                'reason': "Price below 50EMA"
            })
    
    return recommendation

@app.route('/')
def stock_analysis():
    try:
        stock_data, future_prices, model_r2 = get_stock_data(TICKER)
        
        if stock_data.empty:
            raise ValueError("No stock data available")
        
        # Technical indicators
        latest_close = float(stock_data['Close'].iloc[-1])
        ema_50 = float(stock_data['EMA_50'].iloc[-1])
        sma_200 = float(stock_data['SMA_200'].iloc[-1])
        current_rsi = float(stock_data['RSI'].iloc[-1])
        current_macd = float(stock_data['MACD'].iloc[-1])
        current_signal = float(stock_data['Signal'].iloc[-1])
        
        # Prediction results
        if len(future_prices) > 0:
            predicted_price = future_prices[-1]  # Price at the end of prediction period
            price_change = (predicted_price - latest_close) / latest_close * 100
        else:
            price_change = 0
        
        # Generate recommendation
        recommendation = generate_recommendation(
            latest_close, ema_50, sma_200, current_rsi, 
            current_macd, current_signal, price_change, model_r2
        )
        
        # Create plot
        plot_data = create_plot(stock_data, future_prices, model_r2)
        
        # Last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return render_template(
            'index.html',
            ticker=TICKER,
            plot_data=plot_data,
            recommendation=recommendation,
            last_updated=last_updated,
            prediction_days=PREDICTION_DAYS
        )
        
    except Exception as e:
        error_message = f"Error: {str(e)}. Possible fixes: 1) Check ticker symbol 2) Verify internet connection 3) Try again later"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
