# Generate synthetic credit card transaction data

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(
    start_date='2020-01-01',
    end_date='2023-12-31',
    avg_daily_transactions=1000,
    fraud_rate=0.02,
    seasonal_pattern=True,
    output_path='data/raw/credit_card_data.csv'
):
    """
    Generate synthetic credit card transaction data.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        avg_daily_transactions (int): Average number of transactions per day.
        fraud_rate (float): Percentage of fraudulent transactions.
        seasonal_pattern (bool): Whether to include seasonal patterns in the data.
        output_path (str): Path to save the generated data.

    Returns:
        pd.DataFrame: Generated transaction data.
    """
    # Convert start and end dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Create a date range
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

    # Generate daily transactions
    transactions = []
    for day in date_range:
        # Add seasonal patterns
        if seasonal_pattern:
            # More transactions on weekends
            if day.dayofweek >= 5:  # Saturday or Sunday
                daily_factor = 1.5
            else:
                daily_factor = 1.0

            # Monthly pattern (more at the beginning of the month)
            if day.day <= 10:
                monthly_factor = 1.2
            else:
                monthly_factor = 1.0

            # Yearly pattern (more during holiday season)
            if day.month == 12:
                yearly_factor = 1.3
            elif day.month in [6, 7]:  # Summer
                yearly_factor = 1.2
            else:
                yearly_factor = 1.0

            # Combine factors
            total_factor = daily_factor * monthly_factor * yearly_factor
        else:
            total_factor = 1.0

        # Add some random noise
        noise = np.random.normal(loc=1.0, scale=0.1)
        total_factor *= noise

        # Calculate number of transactions for the day
        num_transactions = int(avg_daily_transactions * total_factor)

        # Generate transactions for the day
        for _ in range(num_transactions):
            # Generate a random transaction amount (between $10 and $1000)
            amount = np.random.uniform(10, 1000)

            # Determine if the transaction is fraudulent
            is_fraud = np.random.random() < fraud_rate

            # Create a timestamp for the transaction within the day
            transaction_time = day + timedelta(
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60),
                seconds=np.random.randint(0, 60)
            )

            # Create a transaction record
            transaction = {
                'trans_date': transaction_time.strftime('%Y-%m-%d'),
                'trans_amount': round(amount, 2),
                'is_fraud': int(is_fraud)
            }

            transactions.append(transaction)

    # Create a DataFrame
    df = pd.DataFrame(transactions)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the data
    df.to_csv(output_path, index=False)

    return df

if __name__ == '__main__':
    # Generate synthetic data
    df = generate_synthetic_data()
    print(f"Generated {len(df)} transactions.")
    print(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean() * 100:.2f}%)")
    print(f"Data saved to data/raw/credit_card_data.csv") 