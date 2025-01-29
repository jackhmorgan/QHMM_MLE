def calculate_volatility_bins(volatilities):
    return [(volatilities[i]+volatilities[i+1])/2 for i in range(len(volatilities)-1)]