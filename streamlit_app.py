import pandas as pd
import sqlite3
import joblib
from datetime import datetime
import base64
import cv2
import numpy as np
from PIL import Image
import io
import streamlit as st

# Function to load the image and encode it in Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Add custom CSS for background image
background_image_path = "assets/background.png"
background_base64 = get_base64_image(background_image_path)

st.markdown(
    f"""
    <style>
    body {{
        background-image: url("data:image/png;base64,{background_base64}"); 
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}
    .stApp {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model and encoders
model = joblib.load('model/gradient_boosting_model.pkl')
label_encoders_features = joblib.load('model/label_encoders.pkl')

# Database setup for plant tracking
conn = sqlite3.connect('plant_tracking.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS plants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date_planted DATE,
        days_to_maturity INTEGER,
        SEED REAL,
        SEEDKGHA REAL,
        DFF INTEGER,
        MATURE INTEGER,
        notes TEXT
    )
''')
conn.commit()

# Helper function to calculate growth stage
def get_growth_stage(days_remaining):
    if days_remaining > 35:
        return "Germination"
    elif 21 < days_remaining <= 35:
        return "Vegetative"
    elif 10 < days_remaining <= 21:
        return "Flowering"
    elif 0 < days_remaining <= 10:
        return "Podding"
    else:
        return "Maturity"

# Function to extract a frame from a video using OpenCV
def get_frame_from_video(video_path, frame_time):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(frame_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not extract frame. Check the video path and time.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

# Function to delete all entries from the database
def delete_all_entries():
    cursor.execute("DELETE FROM plants")
    conn.commit()

# Function to delete a specific plant by name
def delete_plant_by_name(plant_name):
    cursor.execute("DELETE FROM plants WHERE name = ?", (plant_name,))
    conn.commit()

# Streamlit UI with tabs
st.title("Cowpea Crop Monitoring & Growth Tracker")

tabs = st.tabs(["Prediction Tool", "Growth Tracker"])

# Prediction Tool Tab
with tabs[0]:
    st.header("Prediction Tool")
    st.write("Select values for REP, VARIETY, and GID to predict crop parameters.")
    
    # Dropdowns for prediction inputs
    rep_options = label_encoders_features['REP'].classes_
    variety_options = label_encoders_features['VARIETY'].classes_
    gid_options = label_encoders_features['GID'].classes_

    rep = st.selectbox("Select REP", options=rep_options)
    variety = st.selectbox("Select VARIETY", options=variety_options)
    gid = st.selectbox("Select GID", options=gid_options)

    # Add notes input
    notes = st.text_area("Add any notes about this plant (optional)")

    # Ask the user how many days have passed since planting
    days_passed = st.number_input(
        "How many days have passed since the plant was planted?",
        min_value=0,
        value=0,
        step=1
    )

    if st.button("Predict and Track"):
        encoded_rep = label_encoders_features['REP'].transform([rep])[0]
        encoded_variety = label_encoders_features['VARIETY'].transform([variety])[0]
        encoded_gid = label_encoders_features['GID'].transform([gid])[0]

        input_data = pd.DataFrame(
            [[encoded_rep, encoded_variety, encoded_gid]],
            columns=['REP', 'VARIETY', 'GID']
        )
        
        prediction = model.predict(input_data)
        st.subheader("Predicted Parameters:")
        target_columns = [
            "Number of expected seeds in POD", "SEEDKGHA",
            "Days to 50% maturity (DFF)", "Days to 95% maturity (MATURE)"
        ]
        
        SEED = prediction[0][0]
        SEEDKGHA = prediction[0][1]
        DFF = int(prediction[0][2])
        MATURE = int(prediction[0][3])

        for idx, value in enumerate([SEED, SEEDKGHA, DFF, MATURE]):
            st.write(f"{target_columns[idx]}: {value:.4f}")
        
        days_remaining = max(MATURE - days_passed, 0)
        date_planted = datetime.now().date()
        plant_name = f"Plant-{rep}-{variety}-{gid}"
        
        cursor.execute('''
            INSERT INTO plants (name, date_planted, days_to_maturity, SEED, SEEDKGHA, DFF, MATURE, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (plant_name, date_planted, days_remaining, SEED, SEEDKGHA, DFF, MATURE, notes))
        conn.commit()
        st.success(f"Plant '{plant_name}' has been added to the tracker automatically!")

# Growth Tracker Tab
with tabs[1]:
    st.header("Plant Growth Tracker")
    
    plants_df = pd.read_sql_query("SELECT * FROM plants", conn)

    if not plants_df.empty:
        st.write("Here are your tracked plants:")
        
        st.dataframe(plants_df.drop(columns=['id']), use_container_width=True)
        
        selected_plant = st.selectbox(
            "View details for:",
            plants_df['name'].tolist()
        )
        plant_details = plants_df[plants_df['name'] == selected_plant].iloc[0]
        days_elapsed = (datetime.now().date() - datetime.strptime(plant_details['date_planted'], "%Y-%m-%d").date()).days
        days_remaining = plant_details['MATURE'] - days_elapsed
        
        growth_stage = get_growth_stage(days_remaining)

        st.subheader(f"Details for {plant_details['name']}")
        st.write(f"- **Date Planted:** {plant_details['date_planted']}")
        st.write(f"- **Days Remaining to mature:** {days_remaining}")
        st.write(f"- **Current Growth Stage:** {growth_stage}")
        st.write(f"- **Notes:** {plant_details['notes']}")

        # Add delete buttons
        if st.button("Delete This Plant"):
            delete_plant_by_name(selected_plant)
            st.success(f"Plant '{selected_plant}' deleted successfully!")
            st.experimental_rerun()

        if st.button("Clear All Plants"):
            delete_all_entries()
            st.success("All plants have been cleared from the tracker!")
            st.experimental_rerun()
    else:
        st.warning("No plants have been added to the tracker yet.")

# Close the database connection
conn.close()
