import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, HTML

# Upload CSV
print("📂 Upload your dataset (house_predictor.csv)")
uploaded = files.upload()

df = pd.read_csv("house_predictor.csv")
df.rename(columns={"price_lakhs": "price"}, inplace=True)

df = df.dropna()

# Features
X = df[["area", "size_bhk", "location", "location_score", "age_years", "furnishing"]]
y = df["price"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("location_encoder", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        ("furnishing_encoder", OneHotEncoder(handle_unknown="ignore"), ["furnishing"])
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("transform", preprocessor),
    ("model", LinearRegression())
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

display(HTML("<h3>📊 Model Performance</h3>"))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# --- UI SECTION ---

display(HTML("<h2>🏠 Predict House Price</h2>"))

area = widgets.FloatText(description="Area (sqft):")
bhk = widgets.IntSlider(description="BHK:", min=1, max=10, value=2)
location = widgets.Text(description="Location:")
location_score = widgets.IntSlider(description="Location Score:", min=1, max=10, value=5)
age = widgets.IntSlider(description="Age (years):", min=0, max=50, value=2)
furnishing = widgets.Dropdown(
    options=["furnished", "semi-furnished", "unfurnished"],
    description="Furnishing:"
)

predict_button = widgets.Button(description="Predict Price", button_style='success')
output = widgets.Output()

def predict_price(b):
    with output:
        output.clear_output()
        data = pd.DataFrame({
            "area": [area.value],
            "size_bhk": [bhk.value],
            "location": [location.value],
            "location_score": [location_score.value],
            "age_years": [age.value],
            "furnishing": [furnishing.value]
        })

        price = pipeline.predict(data)[0]
        print(f"💰 Estimated Price: ₹{price:.2f} Lakhs")

predict_button.on_click(predict_price)

ui = widgets.VBox([area, bhk, location, location_score, age, furnishing, predict_button, output])
display(ui)