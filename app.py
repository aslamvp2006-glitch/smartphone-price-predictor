import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smartphone Price Predictor",
    page_icon="📱",
    layout="wide"
)

# -----------------------------
# Load Trained Model
# -----------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "smartphone_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_names.pkl"

st.title("📱 Smartphone Price Category Predictor")
st.write(
    "Predict a smartphone's price category using "
    "Machine Learning and a Random Forest Classifier."
)

st.divider()

if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error("Trained model files were not found.")
    st.write(
        "Please run `python train_model.py` first "
        "to generate the model files."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# -----------------------------
# Smartphone Input Form
# -----------------------------
st.subheader("Enter Smartphone Specifications")
st.caption("Adjust the specifications and click Predict.")

inputs = {}

# Expected ranges for the standard smartphone dataset
settings = {
    "battery_power": (500, 2000, 100),
    "blue": (0, 1, 1),
    "clock_speed": (0.5, 3.0, 0.5),
    "dual_sim": (0, 1, 1),
    "fc": (0, 19, 1),
    "four_g": (0, 1, 1),
    "int_memory": (2, 64, 2),
    "m_dep": (0.1, 1.0, 0.1),
    "mobile_wt": (80, 200, 5),
    "n_cores": (1, 8, 1),
    "pc": (0, 20, 1),
    "px_height": (0, 1960, 100),
    "px_width": (500, 2000, 100),
    "ram": (256, 4000, 256),
    "sc_h": (5, 19, 1),
    "sc_w": (0, 18, 1),
    "talk_time": (2, 20, 1),
    "three_g": (0, 1, 1),
    "touch_screen": (0, 1, 1),
    "wifi": (0, 1, 1)
}

labels = {
    "battery_power": "Battery Power (mAh)",
    "blue": "Bluetooth",
    "clock_speed": "Clock Speed (GHz)",
    "dual_sim": "Dual SIM",
    "fc": "Front Camera (MP)",
    "four_g": "4G Support",
    "int_memory": "Internal Memory (GB)",
    "m_dep": "Mobile Depth (cm)",
    "mobile_wt": "Mobile Weight (g)",
    "n_cores": "Number of Cores",
    "pc": "Primary Camera (MP)",
    "px_height": "Pixel Height",
    "px_width": "Pixel Width",
    "ram": "RAM (MB)",
    "sc_h": "Screen Height (cm)",
    "sc_w": "Screen Width (cm)",
    "talk_time": "Talk Time (hours)",
    "three_g": "3G Support",
    "touch_screen": "Touch Screen",
    "wifi": "Wi-Fi"
}

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    for index, feature in enumerate(feature_names):
        column = [col1, col2, col3][index % 3]
        label = labels.get(
            feature, feature.replace("_", " ").title()
        )

        with column:
            if feature in settings:
                minimum, maximum, step = settings[feature]

                if feature in [
                    "blue", "dual_sim", "four_g",
                    "three_g", "touch_screen", "wifi"
                ]:
                    inputs[feature] = st.selectbox(
                        label,
                        options=[0, 1],
                        format_func=lambda value: (
                            "Yes" if value == 1 else "No"
                        )
                    )
                elif feature == "clock_speed":
                    inputs[feature] = st.number_input(
                        label,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=1.5,
                        step=float(step)
                    )
                elif feature == "m_dep":
                    inputs[feature] = st.number_input(
                        label,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=0.5,
                        step=float(step)
                    )
                else:
                    default = min(
                        max(1000, minimum), maximum
                    )
                    inputs[feature] = st.number_input(
                        label,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default),
                        step=int(step)
                    )
            else:
                st.warning(
                    f"No input range configured for {feature}"
                )
                inputs[feature] = st.number_input(
                    label,
                    value=0.0
                )

    submitted = st.form_submit_button(
        "🔍 Predict Price Category",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    try:
        input_df = pd.DataFrame(
            [[inputs[feature] for feature in feature_names]],
            columns=feature_names
        )

        prediction = int(model.predict(input_df)[0])

        categories = {
            0: "Low Cost",
            1: "Medium-Low Cost",
            2: "Medium-High Cost",
            3: "High Cost"
        }

        st.divider()
        st.subheader("Prediction Result")

        result = categories.get(
            prediction, f"Category {prediction}"
        )

        st.success(f"Predicted Price Category: {result}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence = max(probabilities) * 100

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is the model's estimated probability "
                "for its selected category, not a guarantee "
                "that the prediction is correct."
            )

        st.subheader("Specifications Submitted")
        st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the feature names and order match "
            "those used when training the model."
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Mini Project | Machine Learning | "
    "Random Forest Classifier"
)
