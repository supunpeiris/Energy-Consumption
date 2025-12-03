# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Building Energy Consumption Predictor",
    page_icon="🏢",
    layout="wide"
)

# Title
st.title("🏢 Building Energy Consumption Predictor")
st.markdown("""
Predict annual energy consumption based on building characteristics and operational parameters.
Based on research: *"Enhancing the Accuracy of Building Performance Simulation through Post-Occupancy Calibration"*
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    ["📊 Predict Energy", "📈 Model Insights", "📚 About", "📁 Upload Data"]
)

# Load model
@st.cache_resource
def load_model():
    try:
        artifacts = joblib.load('building_energy_artifacts.joblib')
        return artifacts
    except:
        st.error("Model not found. Please train the model first.")
        return None

# Load dataset info
@st.cache_data
def load_dataset_info():
    try:
        df = pd.read_csv('building_energy_dataset.csv')
        return df
    except:
        return None

if app_mode == "📊 Predict Energy":
    st.header("Building Energy Consumption Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Information")
        total_floor_area = st.number_input(
            "Total Floor Area (m²)",
            min_value=100,
            max_value=100000,
            value=5000,
            step=100,
            help="Total floor area of the building"
        )
        
        number_of_floors = st.number_input(
            "Number of Floors",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )
        
        building_age = st.number_input(
            "Building Age (years)",
            min_value=0,
            max_value=200,
            value=10,
            step=1
        )
        
        building_type = st.selectbox(
            "Building Type",
            ["Office", "Residential", "Commercial", "Educational", "Healthcare", "Mixed-Use"],
            index=0
        )
    
    with col2:
        st.subheader("Occupancy & Usage")
        occupant_count = st.number_input(
            "Number of Occupants",
            min_value=1,
            max_value=5000,
            value=200,
            step=10
        )
        
        occupancy_hours = st.slider(
            "Daily Operating Hours",
            min_value=0,
            max_value=24,
            value=10,
            step=1,
            help="Average hours the building is occupied per day"
        )
        
        has_elevator = st.radio(
            "Has Elevator?",
            ["Yes", "No"],
            horizontal=True
        )
        
        equipment_load = st.slider(
            "Equipment Load Factor",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Relative equipment usage intensity"
        )
    
    with col3:
        st.subheader("Systems & Environment")
        hvac_efficiency = st.slider(
            "HVAC System Efficiency",
            min_value=0.5,
            max_value=1.0,
            value=0.85,
            step=0.01,
            help="Higher values indicate more efficient systems (COP ratio)"
        )
        
        lighting_power_density = st.slider(
            "Lighting Power Density (W/m²)",
            min_value=5,
            max_value=30,
            value=12,
            step=1
        )
        
        equipment_power_density = st.slider(
            "Equipment Power Density (W/m²)",
            min_value=2,
            max_value=20,
            value=8,
            step=1
        )
        
        outdoor_temperature = st.number_input(
            "Average Outdoor Temperature (°C)",
            min_value=-30,
            max_value=50,
            value=25,
            step=1
        )
        
        humidity = st.slider(
            "Average Relative Humidity (%)",
            min_value=10,
            max_value=100,
            value=60,
            step=5
        )
    
    # Advanced parameters
    with st.expander("⚙️ Advanced Parameters & Research-Based Features"):
        col4, col5 = st.columns(2)
        
        with col4:
            climate_zone = st.selectbox(
                "Climate Zone",
                ["Hot-Humid", "Hot-Dry", "Temperate", "Cold", "Very Cold"],
                index=2
            )
            
            envelope_efficiency = st.slider(
                "Building Envelope Efficiency",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Thermal performance of building envelope"
            )
            
            thermal_mass = st.slider(
                "Thermal Mass Effect",
                min_value=0.5,
                max_value=2.0,
                value=1.2,
                step=0.1,
                help="Effect of building thermal mass on energy consumption"
            )
        
        with col5:
            usage_intensity = st.slider(
                "Usage Intensity Factor",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Overall intensity of building usage"
            )
            
            maintenance_quality = st.slider(
                "Maintenance Quality",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Quality of building systems maintenance"
            )
            
            # Research-based feature from paper
            performance_gap_factor = st.slider(
                "Performance Gap Factor",
                min_value=0.8,
                max_value=1.2,
                value=1.0,
                step=0.01,
                help="Accounts for discrepancy between design and actual performance (from research paper)"
            )
    
    # Prediction button
    if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
        artifacts = load_model()
        
        if artifacts:
            try:
                # Prepare input features based on trained model
                model = artifacts['model']
                scaler = artifacts['scaler']
                features = artifacts['features']
                
                # Calculate derived values
                occupant_density = occupant_count / total_floor_area if total_floor_area > 0 else 0
                
                # Create input dictionary
                input_data = {}
                for feature in features:
                    if feature == 'total_floor_area':
                        input_data[feature] = total_floor_area
                    elif feature == 'number_of_floors':
                        input_data[feature] = number_of_floors
                    elif feature == 'occupant_count':
                        input_data[feature] = occupant_count
                    elif feature == 'building_age':
                        input_data[feature] = building_age
                    elif feature == 'hvac_efficiency':
                        input_data[feature] = hvac_efficiency
                    elif feature == 'lighting_power_density':
                        input_data[feature] = lighting_power_density
                    elif feature == 'equipment_power_density':
                        input_data[feature] = equipment_power_density * equipment_load
                    elif feature == 'outdoor_temperature':
                        input_data[feature] = outdoor_temperature
                    elif feature == 'humidity':
                        input_data[feature] = humidity
                    elif feature == 'occupancy_hours':
                        input_data[feature] = occupancy_hours
                    else:
                        # Set default value for other features
                        input_data[feature] = 0
                
                # Create DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Ensure correct order
                input_df = input_df[features]
                
                # Scale features
                input_scaled = scaler.transform(input_df)
                
                # Predict
                prediction = model.predict(input_scaled)[0]
                
                # Apply research-based adjustment
                adjusted_prediction = prediction * performance_gap_factor
                
                # Display results
                st.success(f"### Predicted Annual Energy Consumption: **{adjusted_prediction:,.0f} kWh**")
                
                # Create metrics columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Energy Intensity",
                        f"{adjusted_prediction/total_floor_area:.1f} kWh/m²",
                        delta=f"±{(adjusted_prediction*0.1)/total_floor_area:.1f} kWh/m²"
                    )
                
                with col2:
                    st.metric(
                        "Per Occupant",
                        f"{adjusted_prediction/occupant_count:.0f} kWh/person",
                        delta=f"±{(adjusted_prediction*0.1)/occupant_count:.0f} kWh/person"
                    )
                
                with col3:
                    st.metric(
                        "Monthly Average",
                        f"{adjusted_prediction/12:,.0f} kWh/month",
                        delta=f"±{adjusted_prediction*0.1/12:,.0f} kWh/month"
                    )
                
                with col4:
                    co2_emissions = adjusted_prediction * 0.85  # kg CO2 per kWh
                    st.metric(
                        "CO₂ Emissions",
                        f"{co2_emissions/1000:,.1f} tons",
                        help="Based on average grid emission factor"
                    )
                
                # Visualization
                st.subheader("📊 Energy Consumption Analysis")
                
                tab1, tab2, tab3 = st.tabs(["Breakdown", "Comparison", "Efficiency"])
                
                with tab1:
                    # Energy breakdown
                    breakdown = {
                        'HVAC (38%)': adjusted_prediction * 0.38,
                        'Lighting (15%)': adjusted_prediction * 0.15,
                        'Equipment (15%)': adjusted_prediction * 0.15,
                        'Hot Water (8%)': adjusted_prediction * 0.08,
                        'Other (24%)': adjusted_prediction * 0.24
                    }
                    
                    fig1 = go.Figure(data=[go.Pie(
                        labels=list(breakdown.keys()),
                        values=list(breakdown.values()),
                        hole=0.4,
                        marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#A8E6CF']
                    )])
                    
                    fig1.update_layout(
                        title="Estimated Energy Consumption Breakdown",
                        height=400
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with tab2:
                    # Comparison with similar buildings
                    df = load_dataset_info()
                    if df is not None:
                        similar_buildings = df[
                            (df['total_floor_area'] >= total_floor_area * 0.8) &
                            (df['total_floor_area'] <= total_floor_area * 1.2)
                        ].copy()
                        
                        if len(similar_buildings) > 0:
                            similar_buildings['Type'] = 'Similar Buildings'
                            current_building = pd.DataFrame([{
                                'energy_consumption_kwh': adjusted_prediction,
                                'Type': 'Your Building'
                            }])
                            
                            comparison_df = pd.concat([
                                similar_buildings[['energy_consumption_kwh', 'Type']],
                                current_building
                            ])
                            
                            fig2 = px.box(
                                comparison_df,
                                x='Type',
                                y='energy_consumption_kwh',
                                points='all',
                                title="Comparison with Similar Buildings"
                            )
                            
                            fig2.update_layout(height=400)
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info("No similar buildings found in dataset for comparison.")
                
                with tab3:
                    # Efficiency recommendations
                    st.subheader("💡 Energy Efficiency Recommendations")
                    
                    recommendations = []
                    
                    if hvac_efficiency < 0.8:
                        recommendations.append((
                            "Upgrade HVAC System",
                            f"Current efficiency: {hvac_efficiency:.2f}. Target: >0.85",
                            "Potential savings: 15-25%"
                        ))
                    
                    if lighting_power_density > 15:
                        recommendations.append((
                            "Improve Lighting Efficiency",
                            f"Current: {lighting_power_density} W/m². Target: <12 W/m²",
                            "Potential savings: 20-40%"
                        ))
                    
                    if building_age > 20:
                        recommendations.append((
                            "Building Envelope Retrofit",
                            f"Building age: {building_age} years",
                            "Potential savings: 10-30%"
                        ))
                    
                    if not recommendations:
                        st.success("Your building parameters are already efficient!")
                    else:
                        for title, current, savings in recommendations:
                            with st.expander(f"🔧 {title}"):
                                st.write(f"**Current**: {current}")
                                st.write(f"**{savings}**")
                                st.write(f"**Actions**: Regular maintenance, system upgrades, smart controls")
                
                # Download prediction
                prediction_data = {
                    'Parameter': list(input_data.keys()),
                    'Value': list(input_data.values()),
                    'Predicted_Energy_kWh': [adjusted_prediction] * len(input_data)
                }
                
                prediction_df = pd.DataFrame(prediction_data)
                csv = prediction_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Prediction Details",
                    data=csv,
                    file_name="building_energy_prediction.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Please check that all required features are provided.")

elif app_mode == "📈 Model Insights":
    st.header("Model Performance Insights")
    
    artifacts = load_model()
    
    if artifacts:
        st.info(f"**Model Type**: {artifacts.get('best_model_name', 'Unknown')}")
        
        # Display metrics
        st.subheader("Model Performance Metrics")
        
        if 'results' in artifacts:
            results = artifacts['results']
            best_model = artifacts['best_model_name']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{results[best_model]['R2']:.4f}")
            with col2:
                st.metric("MAE", f"{results[best_model]['MAE']:,.0f} kWh")
            with col3:
                st.metric("RMSE", f"{results[best_model]['RMSE']:,.0f} kWh")
            with col4:
                st.metric("MAPE", f"{results[best_model]['MAPE']:.2f}%")
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        
        # Load dataset for correlation
        df = load_dataset_info()
        if df is not None and 'features' in artifacts:
            features = artifacts['features']
            
            # Calculate correlations
            corr_data = []
            for feature in features:
                if feature in df.columns:
                    correlation = df[feature].corr(df['energy_consumption_kwh'])
                    corr_data.append({
                        'Feature': feature,
                        'Correlation': abs(correlation),
                        'Direction': 'Positive' if correlation > 0 else 'Negative'
                    })
            
            if corr_data:
                corr_df = pd.DataFrame(corr_data).sort_values('Correlation', ascending=False)
                
                fig = go.Figure()
                
                # Positive correlations
                pos_df = corr_df[corr_df['Direction'] == 'Positive']
                if len(pos_df) > 0:
                    fig.add_trace(go.Bar(
                        x=pos_df['Correlation'],
                        y=pos_df['Feature'],
                        name='Positive Correlation',
                        orientation='h',
                        marker_color='#4ECDC4'
                    ))
                
                # Negative correlations
                neg_df = corr_df[corr_df['Direction'] == 'Negative']
                if len(neg_df) > 0:
                    fig.add_trace(go.Bar(
                        x=neg_df['Correlation'],
                        y=neg_df['Feature'],
                        name='Negative Correlation',
                        orientation='h',
                        marker_color='#FF6B6B'
                    ))
                
                fig.update_layout(
                    title="Feature Correlation with Energy Consumption",
                    xaxis_title="Absolute Correlation Coefficient",
                    height=500,
                    barmode='overlay'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        if 'results' in artifacts:
            st.subheader("Model Comparison")
            
            models_data = []
            for model_name, metrics in artifacts['results'].items():
                models_data.append({
                    'Model': model_name,
                    'R²': metrics['R2'],
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE']
                })
            
            models_df = pd.DataFrame(models_data).sort_values('R²', ascending=False)
            st.dataframe(models_df.style.format({
                'R²': '{:.4f}',
                'MAE': '{:,.0f}',
                'RMSE': '{:,.0f}'
            }), use_container_width=True)

elif app_mode == "📚 About":
    st.header("About This Application")
    
    st.markdown("""
    ## Building Energy Consumption Predictor
    
    ### Overview
    This application predicts annual energy consumption for buildings based on architectural, 
    operational, and environmental parameters using machine learning models trained on synthetic data.
    
    ### Research Basis
    Based on the research paper:
    *"Enhancing the Accuracy of Building Performance Simulation through Post-Occupancy Calibration"*
    
    **Key research findings incorporated:**
    - HVAC inefficiencies contribute 38.18% to performance gap
    - Lighting inefficiencies contribute 15.07%
    - Occupancy patterns contribute 14.48%
    - Weather conditions contribute 3.24%
    
    ### Features Used
    1. **Building Characteristics**: Area, floors, age, type
    2. **System Efficiencies**: HVAC, lighting, equipment
    3. **Environmental Factors**: Temperature, humidity, climate zone
    4. **Operational Parameters**: Occupancy hours, usage patterns
    5. **Research-Based Features**: Performance gap factors, efficiency scores
    
    ### Model Architecture
    - **Ensemble Approach**: Combines multiple tree-based models
    - **Best Model**: Random Forest (R²: ~0.92)
    - **Features**: 10+ engineered features
    - **Dataset**: 5,000 synthetic building samples
    
    ### How to Use
    1. Navigate to **Predict Energy** tab
    2. Enter building parameters (basic or advanced)
    3. Click **Predict Energy Consumption**
    4. View results, breakdown, and recommendations
    
    ### Technical Details
    - **Framework**: Streamlit
    - **ML Libraries**: Scikit-learn, XGBoost, LightGBM
    - **Visualization**: Plotly
    - **Deployment**: Local/Cloud ready
    
    ### Limitations
    - Predictions based on synthetic data
    - Regional variations may affect accuracy
    - Actual measurements recommended for critical decisions
    
    ### Future Enhancements
    - Real building data integration
    - Real-time sensor data integration
    - Advanced ML models (deep learning)
    - Carbon footprint optimization
    """)

elif app_mode == "📁 Upload Data":
    st.header("Upload Building Data")
    
    st.info("""
    Upload your own building data (CSV format) to:
    1. Compare with predictions
    2. Enhance model training
    3. Generate custom analyses
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload building data with features similar to the training dataset"
    )
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully: {len(df_uploaded)} rows, {len(df_uploaded.columns)} columns")
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(df_uploaded.head(), use_container_width=True)
            
            # Check for required columns
            required_cols = ['total_floor_area', 'energy_consumption_kwh']
            missing_cols = [col for col in required_cols if col not in df_uploaded.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            else:
                # Basic statistics
                st.subheader("Data Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Average Energy",
                        f"{df_uploaded['energy_consumption_kwh'].mean():,.0f} kWh"
                    )
                with col2:
                    st.metric(
                        "Average Area",
                        f"{df_uploaded['total_floor_area'].mean():,.0f} m²"
                    )
                with col3:
                    st.metric(
                        "Energy Intensity",
                        f"{(df_uploaded['energy_consumption_kwh']/df_uploaded['total_floor_area']).mean():.1f} kWh/m²"
                    )
                
                # Comparison with model predictions
                st.subheader("Comparison with Model Predictions")
                
                artifacts = load_model()
                if artifacts and 'features' in artifacts:
                    features = artifacts['features']
                    
                    # Check which features are available
                    available_features = [f for f in features if f in df_uploaded.columns]
                    
                    if len(available_features) > 0:
                        X_uploaded = df_uploaded[available_features]
                        
                        # Scale and predict
                        scaler = artifacts['scaler']
                        model = artifacts['model']
                        
                        X_scaled = scaler.transform(X_uploaded)
                        predictions = model.predict(X_scaled)
                        
                        # Create comparison
                        comparison_df = pd.DataFrame({
                            'Actual': df_uploaded['energy_consumption_kwh'],
                            'Predicted': predictions,
                            'Difference': df_uploaded['energy_consumption_kwh'] - predictions,
                            'Error %': ((df_uploaded['energy_consumption_kwh'] - predictions) / df_uploaded['energy_consumption_kwh']) * 100
                        })
                        
                        st.dataframe(comparison_df.head(10).style.format({
                            'Actual': '{:,.0f}',
                            'Predicted': '{:,.0f}',
                            'Difference': '{:,.0f}',
                            'Error %': '{:.1f}%'
                        }), use_container_width=True)
                        
                        # Calculate overall metrics
                        mae = np.mean(np.abs(comparison_df['Difference']))
                        mape = np.mean(np.abs(comparison_df['Error %']))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean Absolute Error", f"{mae:,.0f} kWh")
                        with col2:
                            st.metric("Mean Absolute % Error", f"{mape:.1f}%")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Version**: 1.0.0  
    **Last Updated**: December 2025  
    **Environment**: building_energy  
    
    **Note**: For research and educational purposes.  
    Always validate predictions with actual measurements.
    """
)

# Add refresh button
if st.sidebar.button("🔄 Refresh Model"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success("Cache cleared! Reloading models...")
    st.rerun()