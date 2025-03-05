import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from flask import Flask, render_template, request
import plotly.graph_objs as go
import holidays
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as ETS

# Load data
data = pd.read_csv('data/passengers.csv')
df = pd.DataFrame(data, columns=['date', 'n'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date')

# Create complete date range
full_dates = pd.date_range(start='2023-11-23', end='2025-02-28', freq='D')
df_full = pd.DataFrame(index=full_dates).join(df).rename(columns={'n': 'actual'})

# ========================
# FEATURE ENGINEERING
# ========================

# Temporal features
df_full['day_of_week'] = df_full.index.dayofweek
df_full['month'] = df_full.index.month
df_full['is_weekend'] = df_full['day_of_week'].isin([5, 6]).astype(int)

# German holidays
de_holidays = holidays.Germany()
df_full['is_holiday'] = [d in de_holidays for d in df_full.index]

# Lag features (7-day moving average)
df_full['lag_7'] = df_full['actual'].shift(7).rolling(7).mean()

# ========================
# MODEL TRAINING
# ========================

# Prophet Model
prophet_df = df_full.reset_index()[['index', 'actual']].rename(
    columns={'index': 'ds', 'actual': 'y'}).dropna()
prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
prophet_model.fit(prophet_df)

# XGBoost Model
train_data = df_full.dropna().copy()
X = train_data[['day_of_week', 'month', 'is_weekend', 'is_holiday', 'lag_7']]
y = train_data['actual']

xgb_model = XGBRegressor(n_estimators=100, max_depth=3)
xgb_model.fit(X, y)

# SARIMA Model
sarima_df = df_full['actual'].dropna()
sarima_model = SARIMAX(sarima_df, order=(5, 1, 0), seasonal_order=(1, 1, 0, 7))
sarima_model_fit = sarima_model.fit(disp=False)

# ETS Model
ets_df = df_full['actual'].dropna()
ets_model = ETS(ets_df, error='add', trend='add', seasonal='add', seasonal_periods=7)
ets_model_fit = ets_model.fit()

# ========================
# FLASK WEB SERVICE
# ========================

app = Flask(__name__)

@app.route('/')
def home():
    min_date = datetime(2023, 11, 23)
    max_date = datetime(2025, 12, 31)
    return render_template('index.html', min_date=min_date, max_date=max_date)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d')
    
    # Get selected models
    prophet_selected = request.form.get('prophet') == 'on'
    xgb_selected = request.form.get('xgb') == 'on'
    sarima_selected = request.form.get('sarima') == 'on'
    ets_selected = request.form.get('ets') == 'on'

    # Create prediction DataFrame
    pred_dates = pd.date_range(start=start_date, end=end_date)
    pred_df = pd.DataFrame(index=pred_dates)
    
    # Generate features
    pred_df['day_of_week'] = pred_df.index.dayofweek
    pred_df['month'] = pred_df.index.month
    pred_df['is_weekend'] = pred_df['day_of_week'].isin([5, 6]).astype(int)
    pred_df['is_holiday'] = [d in de_holidays for d in pred_df.index]
    
    # Generate lag feature
    last_known = df_full['actual'].last_valid_index()
    pred_df['lag_7'] = df_full.loc[last_known - timedelta(days=7):last_known, 'actual'].mean()
    
    # Make predictions based on selected models
    predictions = []
    
    if prophet_selected:
        future = prophet_model.make_future_dataframe(periods=len(pred_df))
        prophet_pred = prophet_model.predict(future.tail(len(pred_df)))['yhat']
        predictions.append(prophet_pred.values)
    if xgb_selected:
        xgb_pred = xgb_model.predict(pred_df[['day_of_week', 'month', 'is_weekend', 'is_holiday', 'lag_7']])
        predictions.append(xgb_pred)
    if sarima_selected:
        sarima_pred = sarima_model_fit.get_forecast(steps=len(pred_df)).predicted_mean
        predictions.append(sarima_pred.values)
    if ets_selected:
        ets_pred = ets_model_fit.forecast(steps=len(pred_df))
        predictions.append(ets_pred.values)
    
    # Fusion prediction
    if predictions:
        pred_df['predicted'] = np.mean(predictions, axis=0)
    else:
        return render_template('index.html', 
                               min_date=datetime(2023, 11, 23), 
                               max_date=datetime(2025, 12, 31),
                               error="Please select at least one model.",
                               prophet_selected=prophet_selected,
                               xgb_selected=xgb_selected,
                               sarima_selected=sarima_selected,
                               ets_selected=ets_selected)
    
    # Merge with actuals
    result = pred_df.join(df_full['actual'])
    
    # Create visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result['actual'], 
                            name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=result.index, y=result['predicted'],
                            name='Predicted', line=dict(color='red', dash='dot')))

    # Highlight the space between y_true and y_pred
    fig.add_trace(go.Scatter(
        x=result.index, 
        y=result['actual'], 
        fill='tonexty', 
        fillcolor='rgba(255, 0, 0, 0.2)', 
        mode='none', 
        name='Error'
    ))

    fig.update_layout(
        title=f'Passenger Prediction: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
        xaxis_title='Date',
        yaxis_title='Passengers',
        template='plotly_white'
    )

    # Calculate errors
    mask = result['actual'].notna()
    mae = rmse = None
    if mask.any():
        y_true = result.loc[mask, 'actual']
        y_pred = result.loc[mask, 'predicted']
        mae = round(np.mean(np.abs(y_true - y_pred)), 2)
        rmse = round(np.sqrt(np.mean((y_true - y_pred)**2)), 2)

    return render_template('index.html', 
                           min_date=datetime(2023, 11, 23), 
                           max_date=datetime(2025, 12, 31),
                           plot=fig.to_html(full_html=False),
                           mae=mae,
                           rmse=rmse,
                           start_date=start_date.strftime('%Y-%m-%d'),
                           end_date=end_date.strftime('%Y-%m-%d'),
                           prophet_selected=prophet_selected,
                           xgb_selected=xgb_selected,
                           sarima_selected=sarima_selected,
                           ets_selected=ets_selected)

if __name__ == '__main__':
    app.run(debug=True)
