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
import joblib

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
        color: white; /* Optional: Change text color for better readability */
    }}
    .stApp {{
        background-color: rgba(0, 0, 0, 0.6); /* Optional: Add overlay for better contrast */
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# # Load the trained model
# with open('model/gradient_boosting_model.pkl', 'rb') as f:
#     model = dill.load(f)

# # Load the saved encoders
# with open('model/label_encoders.pkl', 'rb') as f:
#     label_encoders_features = dill.load(f)

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
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    frame_number = int(frame_time * fps)  # Calculate the frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set the video position
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not extract frame. Check the video path and time.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return Image.fromarray(frame)

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
        value=0,  # Default to 0 if the user hasn't specified yet
        step=1
    )

    if st.button("Predict and Track"):
        # Encode the selected inputs
        encoded_rep = label_encoders_features['REP'].transform([rep])[0]
        encoded_variety = label_encoders_features['VARIETY'].transform([variety])[0]
        encoded_gid = label_encoders_features['GID'].transform([gid])[0]

        # Prepare input data for prediction
        input_data = pd.DataFrame(
            [[encoded_rep, encoded_variety, encoded_gid]],
            columns=['REP', 'VARIETY', 'GID']
        )
        
        # Make prediction
        prediction = model.predict(input_data)
        st.subheader("Predicted Parameters:")
        target_columns = [
            "SEEDKGHA", "Number of expected seeds in POD",
            "Days to 50% maturity (DFF)", "Days to 95% maturity (MATURE)"
        ]
        
        # Extract predicted values
        SEED = prediction[0][0]
        SEEDKGHA = prediction[0][1]
        DFF = int(prediction[0][2])  # Assuming the third value is DFF (Days to 50% maturity)
        MATURE = int(prediction[0][3])  # Assuming the fourth value is MATURE (Days to 95% maturity)

        # Display the predicted values
        for idx, value in enumerate([SEED, SEEDKGHA, DFF, MATURE]):
            st.write(f"{target_columns[idx]}: {value:.4f}")
        
        # Calculate days remaining after subtracting the days passed
        days_remaining = max(MATURE - days_passed, 0)
        date_planted = datetime.now().date()
        plant_name = f"Plant-{rep}-{variety}-{gid}"  # Generate a default plant name
        
        # Add the plant to the database with the predicted values
        cursor.execute('''
            INSERT INTO plants (name, date_planted, days_to_maturity, SEED, SEEDKGHA, DFF, MATURE, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (plant_name, date_planted, days_remaining, SEED, SEEDKGHA, DFF, MATURE, notes))
        conn.commit()
        st.success(f"Plant '{plant_name}' has been added to the tracker automatically!")

# Growth Tracker Tab
with tabs[1]:
    st.header("Plant Growth Tracker")
    
    # Fetch all plants
    plants_df = pd.read_sql_query("SELECT * FROM plants", conn)

    if not plants_df.empty:
        st.write("Here are your tracked plants:")
        
        # Display plants in a table
        st.dataframe(plants_df.drop(columns=['id']), use_container_width=True)
        
        # Select a plant for details
        selected_plant = st.selectbox(
            "View details for:",
            plants_df['name'].tolist()
        )
        plant_details = plants_df[plants_df['name'] == selected_plant].iloc[0]
        
        # Get the number of days the plant has been growing
        days_elapsed = (datetime.now().date() - datetime.strptime(plant_details['date_planted'], "%Y-%m-%d").date()).days

        days_planted = plant_details['MATURE'] - plant_details['days_to_maturity']

        # Calculate the remaining days based on user input
        days_remaining = max(plant_details['MATURE'] - days_planted, 0)
        
        # Determine the growth stage
        growth_stage = get_growth_stage(days_remaining)

        st.subheader(f"Details for {plant_details['name']}")
        st.write(f"- **Date Planted:** {plant_details['date_planted']}")
        st.write(f"- **Days Remaining to mature:** {plant_details['days_to_maturity']}")
        st.write(f"- **Current Growth Stage:** {growth_stage}")
        st.write(f"- **Predicted SEED per pod:** {int(plant_details['SEED'])}")
        st.write(f"- **Predicted yield (KGHA):** {round(plant_details['SEEDKGHA'], 3)}")
        st.write(f"- **Predicted DFF (50% Maturity):** {plant_details['DFF']}")
        st.write(f"- **Predicted MATURE (95% Maturity):** {plant_details['MATURE']}")
        st.write(f"- **Notes:** {plant_details['notes']}")

        # Calculate the time in seconds to extract the frame
        video_duration = 148  # Total video length in seconds (2 minutes 28 seconds)
        video_start_time = 148 - (int((plant_details['days_to_maturity'] / plant_details['MATURE']) * video_duration))

        # Ensure the video start time is within bounds
        video_start_time = max(0, video_start_time)

        # Extract a frame from the video
        video_path = "assets/timelapse_beans.mp4"  # Replace with the correct path
        try:
            frame_image = get_frame_from_video(video_path, video_start_time)
            st.image(frame_image, caption="Predicted Growth Stage", use_container_width=True)
        except Exception as e:
            st.error(f"Error extracting frame: {str(e)}")
    else:
        st.warning("No plants have been added to the tracker yet.")
