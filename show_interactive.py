import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Loading data for interactive plot...")
    # Load the processed raw data
    df = pd.read_csv("data/raw/merged_raw.csv", index_col=0, parse_dates=True)
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot ETF Price
    color = 'tab:green'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Gold ETF Price (₹)', color=color)
    ax1.plot(df.index, df['Target_Price'], color=color, linewidth=2, label='GOLDBEES.NS')
    ax1.tick_params(axis='y', labelcolor=color)
    

    import re
    import numpy as np
    try:
        with open("outputs/reports/diwali_2026_gold_forecast.md", "r") as f:
            text = f.read()
            p10_match = re.search(r'\* P10: ₹([0-9.]+)', text)
            p25_match = re.search(r'\* P25: ₹([0-9.]+)', text)
            p50_match = re.search(r'\* P50: ₹([0-9.]+)', text)
            p75_match = re.search(r'\* P75: ₹([0-9.]+)', text)
            p90_match = re.search(r'\* P90: ₹([0-9.]+)', text)
            
            if p50_match:
                p10 = float(p10_match.group(1))
                p25 = float(p25_match.group(1))
                p50 = float(p50_match.group(1))
                p75 = float(p75_match.group(1))
                p90 = float(p90_match.group(1))
                
                target_date = pd.to_datetime('2026-11-08')
                last_date = df.index[-1]
                last_price = df['Target_Price'].iloc[-1]
                
                # Generate future dates for the cone
                future_dates = pd.date_range(start=last_date, end=target_date, freq='D')
                days = np.arange(len(future_dates))
                total_days = len(future_dates) - 1
                
                # Spread out as sqrt(time)
                spread_factor = np.sqrt(days / total_days)
                
                # Calculate bounds
                path_50_straight = last_price + (p50 - last_price) * (days / total_days)
                path_10 = path_50_straight - (path_50_straight[-1] - p10) * spread_factor
                path_25 = path_50_straight - (path_50_straight[-1] - p25) * spread_factor
                path_75 = path_50_straight + (p75 - path_50_straight[-1]) * spread_factor
                path_90 = path_50_straight + (p90 - path_50_straight[-1]) * spread_factor
                
                # Generate a realistic jagged Monte Carlo path trending toward the median
                daily_vol = df['Target_Price'].pct_change().std()
                daily_drift = (p50 / last_price) ** (1.0 / max(1, total_days)) - 1.0
                
                sim_path = [last_price]
                np.random.seed(42) # Keep it reproducible
                for _ in range(total_days):
                    random_shock = np.random.normal(0, daily_vol)
                    next_price = sim_path[-1] * (1 + daily_drift + random_shock)
                    sim_path.append(next_price)
                
                # Plot the cone
                ax1.fill_between(future_dates, path_10, path_90, color='gold', alpha=0.2, label='80% Probability Band (P10-P90)')
                ax1.fill_between(future_dates, path_25, path_75, color='orange', alpha=0.4, label='50% Probability Band (P25-P75)')
                ax1.plot(future_dates, sim_path, color='red', linestyle='-', linewidth=2, label='Simulated Realistic Path')
                
                # Label the future Diwali date
                ax1.text(target_date, p50 * 1.05, target_date.strftime('%b %d\n%Y\n(TARGET)'), color='red', weight='bold', fontsize=9, horizontalalignment='center')
                
                # Draw a horizontal line across the entire chart for the final target
                ax1.axhline(y=p50, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Diwali Target Price (₹{:.2f})'.format(p50))
                
                # Plot vertical lines for past Diwali dates
                from src.features.seasonality import DIWALI_DATES
                for d_date in DIWALI_DATES:
                    if d_date <= df.index[-1]:
                        ax1.axvline(x=d_date, color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
                        # Add the text label slightly above the max price so it doesn't clutter the graph
                        y_pos = df['Target_Price'].max() * 0.98
                        ax1.text(d_date, y_pos, d_date.strftime('%b %d\n%Y'), color='purple', alpha=0.8, fontsize=8, rotation=90, verticalalignment='top')
                # Add one invisible line just for the legend
                ax1.axvline(x=df.index[0], color='purple', linestyle=':', linewidth=1.5, alpha=0.5, label='Historical Diwali Dates')
                
                # Plot Historical Backtest Predictions
                try:
                    bt = pd.read_csv("data/raw/backtest_results.csv")
                    bt['Prediction_Date'] = pd.to_datetime(bt['Prediction_Date'])
                    # Diwali date is exactly 30 days after prediction date in our walk-forward logic
                    bt['Diwali_Date'] = bt['Prediction_Date'] + pd.Timedelta(days=30)
                    
                    # Plot them as scatter points
                    ax1.scatter(bt['Diwali_Date'], bt['Predicted_Diwali_Price'], color='blue', marker='o', s=80, zorder=5, label='Past ML Predictions')
                except Exception as e_bt:
                    print("Could not load backtest results:", e_bt)
                
                ax1.legend(loc='upper left')
    except Exception as e:
        print("Could not load prediction:", e)
        
    plt.title('Gold ETF Historical Price & Diwali 2026 Forecast')
    fig.tight_layout()
    
    # Add scroll-to-zoom functionality
    def zoom_fun(event):
        # Get the current x and y limits
        cur_xlim = ax1.get_xlim()
        cur_ylim = ax1.get_ylim()
        
        # Get event location
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
            
        # Determine scale factor (zoom in = < 1, zoom out = > 1)
        scale_factor = 0.8 if event.button == 'up' else 1.2
        
        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        # Keep the cursor point stationary
        ax1.set_xlim([xdata - new_width * (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]),
                      xdata + new_width * (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])])
                      
        ax1.set_ylim([ydata - new_height * (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]),
                      ydata + new_height * (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])])
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', zoom_fun)
    
    print("Opening pyplot window... (Close the window to exit)")
    plt.show()

if __name__ == "__main__":
    main()
