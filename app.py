import csv
import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# File to store fitness data
FILENAME = "fitness_tracker.csv"

# Configure page settings
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="💪",
    layout="wide"
)

# Load custom CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to initialize CSV file if it doesn't exist
def initialize_file():
    try:
        with open(FILENAME, 'x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Workout Type", "Duration (mins)", "Calories Burned", 
                           "Age", "Gender", "Heart Rate", "Body Temperature"])
    except FileExistsError:
        pass

# Function to log a workout with validation
def log_workout(workout_type, duration, calories, age, gender, heart_rate, body_temp):
    if not workout_type.strip():
        st.error("Please enter a workout type")
        return False
    
    date = datetime.date.today().strftime("%Y-%m-%d")
    with open(FILENAME, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date, workout_type, duration, calories, 
                        age, gender, heart_rate, body_temp])
    return True

# Function to load workout data
def load_data():
    try:
        df = pd.read_csv(FILENAME)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "Workout Type", "Duration (mins)", "Calories Burned"])

# Machine Learning Model to Predict Calories Burned
def train_model(data):
    if len(data) < 2:
        return None, None
    X = data[["Duration (mins)"]]
    y = data["Calories Burned"]
    model = LinearRegression()
    model.fit(X, y)
    return model, X

def analyze_fitness_data(data):
    if len(data) < 2:
        return None, None, None
    
    # Feature engineering
    data['Week'] = data['Date'].dt.isocalendar().week
    weekly_stats = data.groupby('Week').agg({
        'Calories Burned': 'sum',
        'Duration (mins)': 'sum',
        'Workout Type': 'count'
    }).reset_index()
    
    # Trend analysis
    X = weekly_stats[['Duration (mins)']]
    y = weekly_stats['Calories Burned']
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate efficiency score
    efficiency = data['Calories Burned'] / data['Duration (mins)']
    avg_efficiency = efficiency.mean()
    
    return model, X, avg_efficiency

def get_personalized_recommendations(data, efficiency):
    if data.empty:
        return []
    
    recommendations = []
    weekly_workouts = len(data[data['Date'] >= pd.Timestamp.now() - pd.Timedelta(days=7)])
    
    if weekly_workouts < 3:
        recommendations.append("Consider increasing workout frequency to at least 3 times per week")
    
    if efficiency < data['Calories Burned'].mean() / data['Duration (mins)'].mean():
        recommendations.append("Try incorporating high-intensity intervals to improve workout efficiency")
    
    return recommendations

def create_workout_stats(data):
    if not data.empty:
        total_workouts = len(data)
        total_calories = data["Calories Burned"].sum()
        avg_duration = data["Duration (mins)"].mean()
        most_common_workout = data["Workout Type"].mode().iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Workouts", f"{total_workouts:,}")
        with col2:
            st.metric("Total Calories Burned", f"{total_calories:,.0f}")
        with col3:
            st.metric("Avg. Duration (mins)", f"{avg_duration:.1f}")
        with col4:
            st.metric("Favorite Workout", most_common_workout)

def create_workout_charts(data):
    if not data.empty:
        # Calories burned over time
        fig_calories = px.line(data, x='Date', y='Calories Burned',
                             title='Calories Burned Over Time')
        st.plotly_chart(fig_calories, use_container_width=True)
        
        # Workout distribution
        fig_dist = px.pie(data, names='Workout Type',
                         title='Workout Type Distribution')
        st.plotly_chart(fig_dist, use_container_width=True)

def main():
    st.title("🏃‍♂️ Personal Fitness Tracker")
    
    menu = ["Dashboard", "Log Workout", "Workout History", "Predictions"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    data = load_data()
    
    if choice == "Dashboard":
        st.header("Fitness Dashboard")
        
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:
            st.markdown("""
            <div class='fitness-info'>
            <h3>Your Path to Better Health</h3>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("### 🏋️ Track your fitness journey")
            
        # Theme selector
        theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Light", "Pastel"])
        if theme:
            if theme == "Dark":
                st.markdown("""<style>
                    .stApp {
                        background: linear-gradient(135deg, #1a1f2c 0%, #2d3446 100%) !important;
                    }
                    </style>""", unsafe_allow_html=True)
            elif theme == "Light":
                st.markdown("""<style>
                    .stApp {
                        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%) !important;
                    }
                    .element-container, .stMarkdown, p, h1, h2, h3 {
                        color: #333333 !important;
                    }
                    </style>""", unsafe_allow_html=True)
            elif theme == "Pastel":
                st.markdown("""<style>
                    .stApp {
                        background: linear-gradient(135deg, #FFE5E5 0%, #FFD1D1 100%) !important;
                    }
                    .element-container, .stMarkdown, p, h1, h2, h3 {
                        color: #333333 !important;
                    }
                    </style>""", unsafe_allow_html=True)
                    
        st.markdown("""
        <div class='fitness-info'>
        <p>Regular exercise is key to maintaining good health and achieving your fitness goals. Track your workouts, monitor your progress, and stay motivated with our comprehensive fitness tracking system.</p>
        
        <p>Benefits of Regular Exercise:</p>
        • Improves cardiovascular health and endurance<br>
        • Builds strength and muscle tone<br>
        • Reduces stress and improves mental health<br>
        • Helps maintain healthy weight<br>
        • Increases energy levels and productivity
        </div>
        """, unsafe_allow_html=True)
        
        create_workout_stats(data)
        
        # ML-based analysis and insights
        if not data.empty:
            model, X, efficiency = analyze_fitness_data(data)
            if model and efficiency is not None:
                st.subheader("💡 Personalized Insights")
                recommendations = get_personalized_recommendations(data, efficiency)
                for rec in recommendations:
                    st.info(rec)
                
                # Efficiency trend
                st.metric("Workout Efficiency", 
                         f"{efficiency:.1f} calories/minute",
                         delta=f"{(efficiency - data['Calories Burned'].mean() / data['Duration (mins)'].mean()):.1f}")
        
        create_workout_charts(data)
    
    elif choice == "Log Workout":
        st.header("Log Your Workout")
        with st.form("workout_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                workout_suggestions = {
                    "Morning Cardio": "6:00 AM - 7:00 AM",
                    "Strength Training": "8:00 AM - 9:30 AM",
                    "Yoga/Stretching": "7:00 AM - 8:00 AM",
                    "HIIT Workout": "5:00 PM - 5:45 PM",
                    "Evening Run/Jog": "6:00 PM - 7:00 PM",
                    "Swimming": "Any time",
                    "Cycling": "Early morning/Evening",
                    "CrossFit": "10:00 AM - 11:30 AM",
                    "Pilates": "9:00 AM - 10:00 AM",
                    "Weight Training": "4:00 PM - 5:30 PM"
                }
                workout_type = st.selectbox("Workout Type", list(workout_suggestions.keys()))
                st.info(f"💡 Suggested time: {workout_suggestions[workout_type]}")
                age = st.slider("Age", min_value=15, max_value=80, value=30)
                heart_rate = st.slider("Heart Rate (bpm)", min_value=60, max_value=200, value=120)
                duration = st.slider("Duration (minutes)", min_value=1, max_value=180, value=30)
            
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                body_temp = st.slider("Body Temperature (°C)", min_value=35.0, max_value=40.0, value=37.0, step=0.1)
                calories = st.slider("Calories Burned", min_value=1, max_value=1000, value=100)
            
            submitted = st.form_submit_button("🏃‍♂️ Start Workout")
            
            if submitted:
                try:
                    if all([workout_type, duration > 0, calories > 0, age > 0, heart_rate > 0]):
                        if log_workout(workout_type, duration, calories, age, gender, heart_rate, body_temp):
                            st.success("🎉 Great job! Workout logged successfully!")
                            st.balloons()
                            
                            # Show workout summary
                            st.info(f"""
                            📊 Workout Summary:
                            - Type: {workout_type}
                            - Duration: {duration} minutes
                            - Calories: {calories} kcal
                            - Heart Rate: {heart_rate} bpm
                            """)
                    else:
                        st.error("Please fill in all required fields with valid values")
                except Exception as e:
                    st.error(f"Error logging workout: {str(e)}")
    
    elif choice == "Workout History":
        st.header("💪 Workout History")
        if not data.empty:
            tab1, tab2 = st.tabs(["📊 Overview", "📋 Detailed History"])
            
            with tab1:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Workouts", len(data))
                with col2:
                    st.metric("Most Popular", data["Workout Type"].mode().iloc[0])
                with col3:
                    st.metric("Total Calories", f"{data['Calories Burned'].sum():,.0f}")
                with col4:
                    st.metric("Avg Duration", f"{data['Duration (mins)'].mean():.0f} mins")

                # Bar chart for workout types
                workout_counts = data["Workout Type"].value_counts()
                st.bar_chart(workout_counts)
                
            with tab2:
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    selected_type = st.multiselect("Filter by Workout Type", data["Workout Type"].unique())
                with col2:
                    date_range = st.date_input("Date Range", [data["Date"].min(), data["Date"].max()])
                
                # Filter data
                filtered_data = data.copy()
                if selected_type:
                    filtered_data = filtered_data[filtered_data["Workout Type"].isin(selected_type)]
                
                # Display filtered data
                st.dataframe(filtered_data.sort_values("Date", ascending=False), use_container_width=True)

            # Weekly progress chart
            st.subheader("📈 Weekly Progress")
            weekly_data = data.groupby(pd.Grouper(key='Date', freq='W')).agg({
                'Calories Burned': 'sum',
                'Duration (mins)': 'sum'
            }).reset_index()
            
            fig = px.line(weekly_data, x='Date', y=['Calories Burned', 'Duration (mins)'],
                         title='Weekly Workout Progress')
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("🏃‍♂️ No workouts logged yet. Start by logging your first workout!")
    
    elif choice == "Predictions":
        st.header("Calorie Prediction")
        
        # User input parameters
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", min_value=10, max_value=100, value=30)
            bmi = st.slider("BMI", min_value=15, max_value=40, value=20)
            duration = st.slider("Duration (min)", min_value=0, max_value=35, value=15)
        with col2:
            heart_rate = st.slider("Heart Rate", min_value=60, max_value=200, value=80)
            gender = st.selectbox("Gender", ["Male", "Female"])
            body_temp = st.slider("Body Temperature (°C)", min_value=35.0, max_value=40.0, value=37.0)

        model, X = train_model(data)
        
        if model:
            # Basic prediction based on duration
            base_calories = model.predict([[duration]])[0]
            
            # Adjust prediction based on other parameters
            if gender == "Male":
                base_calories *= 1.2
            if heart_rate > 120:
                base_calories *= 1.15
            if bmi > 25:
                base_calories *= 0.9
            
            predicted_calories = base_calories
            
            # Create gauge chart for prediction
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = predicted_calories,
                title = {'text': "Predicted Calories"},
                gauge = {
                    'axis': {'range': [None, max(1000, predicted_calories * 1.2)]},
                    'bar': {'color': "#FF4B4B"},
                }
            ))
            st.plotly_chart(fig)
            
            st.info(f"💡 For a {duration} minute workout, you're likely to burn "
                   f"approximately {predicted_calories:.0f} calories!")
        else:
            st.warning("Not enough workout data to make predictions. "
                      "Log at least 2 workouts to enable predictions.")

if __name__ == "__main__":
    initialize_file()
    main()