import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objs as go

# Title of the app
st.title('📈 StockPredict - Stock Price Prediction WebApp')

# Add logo to the sidebar
st.sidebar.image('Designer.jpeg', width=100)  # Adjust the width as needed

# Sidebar for user input
st.sidebar.header('Input Stock Parameters')

# Select box for stock symbol with a few options
stock_symbol = st.sidebar.selectbox('Select Stock Symbol:', ['NFLX', 'GOOGL', 'AMZN', 'MSFT'], index=0)

# Slider for date range selection
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))

# Fetch stock data from Yahoo Finance
st.write(f"Fetching stock data for **{stock_symbol}** from **{start_date}** to **{end_date}**...")
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Check if data is valid
if data.empty or len(data) < 2:
    st.error(f"Error: No or insufficient data found for {stock_symbol}. Please enter a valid stock symbol.")
else:
    # Show raw data with a nice header
    st.subheader(f'📊 {stock_symbol} Stock Data')
    st.write(data.tail())

    # Prepare data for Prophet model
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']]
    data.columns = ['ds', 'y']  # Prophet expects 'ds' and 'y' column names

    # Remove timezone information from 'ds'
    data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)

    # Train Prophet model
    model = Prophet()
    model.fit(data)

    # Make future predictions
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Show forecast results
    st.subheader(f'📈 Forecasting {stock_symbol} Stock Price')
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Convert 'ds' to datetime and 'yhat' to numeric
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    forecast['yhat'] = pd.to_numeric(forecast['yhat'], errors='coerce')

    # Handle invalid data
    forecast.dropna(subset=['ds', 'yhat'], inplace=True)

    # Plotting with Plotly
    fig = go.Figure()

    # Actual stock prices
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual Prices', line=dict(color='blue')))

    # Forecasted prices
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Prices', line=dict(color='orange')))

    # Add fill between the upper and lower bounds
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty',  # This will fill the area between upper and lower bounds
        fillcolor='rgba(0, 200, 200, 0.2)',  # Light green fill
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f'{stock_symbol} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        template='plotly_white'
    )

    # Render Plotly graph
    st.plotly_chart(fig)

    # Plot forecast components with detailed descriptions
    st.subheader(f'{stock_symbol} Stock Forecast Components')

    # Generate the full components plot
    components_fig = model.plot_components(forecast)

    # Add custom descriptions before displaying the components figure
    # This will ensure each component is described in detail

    # Trend component description
    st.write("### Trend Component")
    st.caption("**X-axis:** Date (Time Progression) | **Y-axis:** Price (USD)")
    st.write("This plot shows the overall trend of stock prices over time, capturing long-term upward or downward movements.")

    # Weekly seasonality component description
    st.write("### Weekly Seasonality Component")
    st.caption("**X-axis:** Day of the Week | **Y-axis:** Price Change (USD)")
    st.write("The weekly seasonality plot highlights the typical pattern in stock prices across different days of the week, which can help you identify the best days for trading.")

    # Yearly seasonality component description
    st.write("### Yearly Seasonality Component")
    st.caption("**X-axis:** Month of the Year | **Y-axis:** Price Change (USD)")
    st.write("The yearly seasonality plot displays recurring price changes throughout the year, showing how specific months tend to affect stock prices.")

    # Display the entire components figure
    st.pyplot(components_fig)


# Add footer
st.markdown('---')
st.write("Made with ❤️ by Ujjwal. Happy Trading!")
