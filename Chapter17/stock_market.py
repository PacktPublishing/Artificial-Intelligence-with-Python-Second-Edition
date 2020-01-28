import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# Load historical stock quotes from matplotlib package 
start_date = datetime.date(1970, 9, 4) 
end_date = datetime.date(2016, 5, 17)

intc = yf.Ticker('INTC').history(start=start_date, end=end_date)

# Extract the closing quotes everyday
closing_quotes = np.array([quote[2] for quote in stock_quotes])

# Take the percentage difference of closing stock prices
diff_percentages = 100.0 * np.diff(intc.Close) / intc.Close[:-1]

# Stack the differences and volume values column-wise for training
training_data = np.column_stack([diff_percentages, intc.Volume[:-1]])

# Create and train Gaussian HMM 
hmm = GaussianHMM(n_components=7, covariance_type='diag', n_iter=1000)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    hmm.fit(training_data)

# Generate data using the HMM model
num_samples = 300 
samples, _ = hmm.sample(num_samples) 

# Plot the difference percentages 
plt.figure()
plt.title('Difference percentages')
plt.plot(np.arange(num_samples), samples[:, 0], c='black')

# Plot the volume of shares traded
plt.figure()
plt.title('Volume of shares')
plt.plot(np.arange(num_samples), samples[:, 1], c='black')
plt.ylim(ymin=0)

plt.show()
