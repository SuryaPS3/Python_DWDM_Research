
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = r'C:\Users\Surya Pratap Singh\OneDrive\Documents\VIT AP\Sem5_VIT_Fall_2024-25\DWDM-CSE4005\CrimesOnWomenData.csv'  # Updated file path
data = pd.read_csv(file_path)

# Preprocess the data
# Create a new column for total crimes by summing all relevant crime columns
data['Total_Crimes'] = data[['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']].sum(axis=1)

# Drop unnecessary columns
data_clean = data.drop(['Unnamed: 0'], axis=1)

# Separate the features (X) and target (y) for classification
X = data_clean[['Year', 'Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']]
y_regression = data_clean['Total_Crimes']

# Define classification target: High vs. Low crime rate
threshold = data_clean['Total_Crimes'].median()  # Median threshold for classification
y_classification = (data_clean['Total_Crimes'] > threshold).astype(int)

# Split the data into training and testing sets for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.3, random_state=42)

# Initialize the models
log_reg = LogisticRegression(max_iter=1000, random_state=42)
random_forest = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)
knn = KNeighborsClassifier()
naive_bayes = GaussianNB()

# Dictionary to store models
models = {
    'Logistic Regression': log_reg,
    'Random Forest': random_forest,
    'SVM': svm,
    'KNN': knn,
    'Naive Bayes': naive_bayes
}

# Dictionary to store accuracy results
accuracy_results = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train_class, y_train_class)
    y_pred = model.predict(X_test_class)
    accuracy = accuracy_score(y_test_class, y_pred)
    accuracy_results[model_name] = accuracy

# Print the accuracy results for each model
print("Classification Model Accuracies:")
for model_name, accuracy in accuracy_results.items():
    print(f"Accuracy of {model_name}: {accuracy * 100:.2f}%")

# ------ Part 2: Predicting crimes for 2022 and 2023 using regression ------

# Use a regression model (e.g., Linear Regression) to predict total crimes
regression_model = LinearRegression()

# Split the data into training and testing sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.3, random_state=42)

# Train the regression model
regression_model.fit(X_train_reg, y_train_reg)
# Calculate the average of past values for each crime type to use as an estimate for 2022 and 2023
avg_values = X.mean()

# Prepare input data for the years 2022 and 2023 using the average values for each crime type

future_years = pd.DataFrame({
    'Year': [2022, 2023],  # Two entries for two years
    'Rape': [avg_values['Rape'], avg_values['Rape']],
    'K&A': [avg_values['K&A'], avg_values['K&A']],
    'DD': [avg_values['DD'], avg_values['DD']],
    'AoW': [avg_values['AoW'], avg_values['AoW']],
    'AoM': [avg_values['AoM'], avg_values['AoM']],
    'DV': [avg_values['DV'], avg_values['DV']],
    'WT': [avg_values['WT'], avg_values['WT']]
})
