import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker for realistic fake data
fake = Faker('en_IN')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(num_samples):
    data = []
    
    for _ in range(num_samples):
        # Base user characteristics
        sim_age_months = random.randint(3, 60)  # 3 months to 5 years
        account_age_days = sim_age_months * 30
        
        # Mobile behavior features
        recharge_freq = np.random.normal(loc=4, scale=1.5)
        avg_recharge = max(100, min(1000, np.random.normal(500, 150)))
        call_stability = np.random.beta(2, 5)  # Most users have stable patterns
        
        # Utility payments
        late_payment_ratio = np.random.beta(2, 8)  # Skewed toward timely payments
        avg_delay_days = np.random.poisson(7) if late_payment_ratio > 0.2 else 0
        
        # Financial behavior
        upi_freq = max(1, np.random.poisson(10))
        credit_debit_ratio = np.random.normal(1.1, 0.3)
        
        # Location stability
        location_stability = max(6, np.random.poisson(18))  # Months at address
        commute_regularity = np.random.uniform(1, 5)  # km std deviation
        
        # App engagement
        app_opens = max(1, np.random.poisson(3))
        session_duration = np.random.normal(8, 3)
        
        # Generate target variable (default risk) based on features
        default_risk = (
            0.4 * late_payment_ratio +
            0.3 * (1 - call_stability) +
            0.2 * (1 / location_stability) +
            0.1 * (1 - min(1, credit_debit_ratio))
        )
        default_risk = min(max(0, default_risk + np.random.normal(0, 0.1)), 1)
        default_risk_label = 1 if default_risk > 0.65 else 0  # Threshold at 65%
        
        # Create record
        record = {
            # Core features
            'recharge_frequency': max(1, recharge_freq),
            'avg_recharge_amount': round(avg_recharge, 2),
            'sim_age_months': sim_age_months,
            'call_stability': round(call_stability, 4),
            
            # Utility features
            'late_payment_ratio': round(late_payment_ratio, 4),
            'avg_delay_days': avg_delay_days,
            
            # Financial features
            'upi_transaction_freq': upi_freq,
            'credit_debit_ratio': round(credit_debit_ratio, 2),
            
            # Location features
            'location_stability': location_stability,
            'commute_regularity': round(commute_regularity, 2),
            
            # App features
            'daily_app_opens': app_opens,
            'avg_session_minutes': round(session_duration, 1),
            
            # Target
            'default_risk': default_risk_label,
            'default_probability': round(default_risk, 4)
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

# Generate datasets
print("Generating training data (10,000 samples)...")
train_df = generate_synthetic_data(10000)
print("Generating test data (100 samples)...")
test_df = generate_synthetic_data(100)

# Add synthetic IDs
train_df.insert(0, 'customer_id', ['TRN_' + str(x).zfill(6) for x in range(10000)])
test_df.insert(0, 'customer_id', ['TST_' + str(x).zfill(3) for x in range(100)])

# Save to CSV
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Datasets generated successfully!")
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print("\nSample training data:")
print(train_df.head(3))
