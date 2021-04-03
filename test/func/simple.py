'''
1. import tycho
2. train a model
3. make some predictions
'''
import pandas as pd
import os
from sklearn.metrics import r2_score
import tycho

# Load the data
full_data = pd.read_csv(os.path.dirname(os.path.realpath(__file__))
 + '/data/kin8nm.csv')

# Shuffle the data
shuffled_data = full_data.sample(frac=1, random_state=11).reset_index(drop=True)

# Split into train and test
train_df = shuffled_data.iloc[:int(len(shuffled_data)*0.9)]
test_df = shuffled_data.iloc[int(len(shuffled_data)*0.1):]

# Train a model
model = tycho.fit(train_df, 'y')

# Make some predictions
predictions = model.infer(test_df)

# Evaluate them for good measures
r2 = r2_score(test_df['y'], predictions)
print(f'Got an r2 score of {r2}')
