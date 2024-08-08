from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_stocks = int(request.form['num_stocks'])
        tickers = [request.form[f'stock{i+1}'].strip().upper() + ".NS" for i in range(num_stocks)]
        return redirect(url_for('results', tickers=','.join(tickers)))
    return render_template('index.html')

@app.route('/results')
def results():
    tickers = request.args.get('tickers').split(',')
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")['Adj Close']
    daily_returns = data.pct_change().dropna()
    expected_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252

    def calculate_portfolio_performance(weights, expected_returns, cov_matrix):
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_std_dev

    def minimize_portfolio_volatility(weights, expected_returns, cov_matrix):
        return calculate_portfolio_performance(weights, expected_returns, cov_matrix)[1]

    num_assets = len(tickers)
    initial_weights = np.repeat(1 / num_assets, num_assets)
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    efficient_frontier = minimize(minimize_portfolio_volatility, initial_weights,
                                   args=(expected_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

    num_points = 100
    frontier_weights = []
    frontier_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_points)
    for r in frontier_returns:
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'eq', 'fun': lambda weights: calculate_portfolio_performance(weights, expected_returns, cov_matrix)[0] - r})
        result = minimize(minimize_portfolio_volatility, initial_weights, args=(expected_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
        frontier_weights.append(result['x'])

    frontier_weights = np.array(frontier_weights)
    portfolio_volatilities = [calculate_portfolio_performance(weights, expected_returns, cov_matrix)[1] for weights in frontier_weights]

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_volatilities, frontier_returns, label='Efficient Frontier', color='b', linestyle='--')
    plt.scatter(portfolio_volatilities, frontier_returns, color='r', marker='o', label='Efficient Portfolios')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    optimized_weights = efficient_frontier.x
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(optimized_weights, expected_returns, cov_matrix)
    weights = {tickers[i]: optimized_weights[i] * 100 for i in range(len(tickers))}
    return render_template('results.html', plot_url=plot_url, weights=weights,
                           portfolio_return=portfolio_return * 100, portfolio_volatility=portfolio_volatility * 100)

if __name__ == '__main__':
    app.run(debug=True)
