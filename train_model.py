import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv('heart.csv')

# 2. Separate Target
X = df.drop('target', axis=1)
y = df['target']

# 3. Define Preprocessing
categorical_features = ['cp', 'thal', 'slope']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# 4. Create Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# 5. Train
pipeline.fit(X, y)

# 6. Save Model AND Data Statistics (for the UI visualizations)
# We save the dataframe itself (or stats) to use for comparison graphs later
model_data = {
    'pipeline': pipeline,
    'data_description': df.describe(),
    'data_sample': df.head(100) # Save a sample for plotting distributions
}

joblib.dump(model_data, 'heart_disease_system.pkl')
print("System built successfully! Saved to 'heart_disease_system.pkl'")