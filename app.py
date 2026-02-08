import pandas as pd
from dash import Dash, dcc, html, Input, Output
from layout import create_page1_layout, create_page2_layout
from callbacks import register_callbacks

#data = pd.read_excel(r"C:\Users\SylvainPihet\OneDrive - S64 Ventures Ltd\Documents\Python\Modern_Portfolio_Theory\Efficient_Frontier\Funds_Valuations_Data.xlsx", sheet_name="Raw_Data")
#data = data.set_index("Date")
#for col in data.columns:
    #ts = data[col]
    #first_index = ts.first_valid_index()
    #if first_index is not None:
        #ts.loc[first_index:] = ts.loc[first_index:].interpolate(method="linear")
    #data[col] = ts
#data = data.drop(columns=["MLoan SICAV S.A.", "Carlyle AlpInvest Private Markets Sub-Fund"])
#import yfinance as yf
tickers = ["TSLA", "PLTR", "GOOG", "AMZN", "CRWV", "^NDX", "MSFT", "COIN", "AMD"]
#data = yf.download(tickers, period="10y", threads=False, progress=False)["Close"]

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),       # URL for page navigation
    dcc.Store(id='basket-store', data=[]),       # Global basket store
    html.Div(id='page-content')                  # Page content will be dynamically loaded
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'))

def display_page(pathname):
    if pathname == '/analytics':
        return create_page2_layout()
    else:
        return create_page1_layout()

register_callbacks(app=app, tickers=tickers)

# --- Run server ---
if __name__ == "__main__":
    app.run(debug=True, port=5002)
