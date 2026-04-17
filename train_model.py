import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

file_path = input("Enter CSV file path: ")
df = pd.read_csv(file_path)

features = [
    'Gender',
    'AttendanceRate',
    'PreviousGrade',
    'ExtracurricularActivities',
    'ParentalSupport',
    'Attendance (%)',
    'Online Classes Taken'
]

target = 'StudyHoursPerWeek'

df = df[features + [target]]

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

df = df.dropna()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"\n Model MAE: {mae:.2f}")

joblib.dump(model, "study_model.pkl")
joblib.dump(features, "feature_names.pkl")

print(" Clean model trained & saved!")
