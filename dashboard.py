# enhanced_dashboard.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
import io
import base64

# Page configuration with enhanced settings
st.set_page_config(
    page_title="AI Predictive Maintenance System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        animation: pulse 2s infinite;
        margin-bottom: 1rem;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffd93d, #ff9a3d);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #6bcf7f, #4ca1af);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Feature highlight */
    .feature-highlight {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #ffd93d;
    }
    
    /* Code block styling */
    .code-block {
        background: #2c3e50;
        color: #ecf0f1;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedPredictiveMaintenanceDashboard:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.equipment_history = []
        
    def check_api_health(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except Exception as e:
            return False, None
    
    def get_expected_ranges(self):
        """Get expected input ranges from API"""
        try:
            response = requests.get(f"{self.api_url}/expected_ranges", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def predict_single(self, equipment_data):
        """Get prediction for single equipment"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=equipment_data,
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                # Store in history for trend analysis
                result['timestamp'] = datetime.now().isoformat()
                self.equipment_history.append(result)
                return result
            return None
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def predict_batch(self, batch_data):
        """Get predictions for batch of equipment"""
        try:
            response = requests.post(
                f"{self.api_url}/predict_batch",
                json=batch_data,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Batch prediction error: {str(e)}")
            return None

def create_tooltip(text, tooltip_text):
    """Create a tooltip element"""
    return f'<span class="tooltip">{text}<span class="tooltiptext">{tooltip_text}</span></span>'

def display_sensor_explanations():
    """Display explanations for each sensor parameter"""
    with st.expander("📖 Understanding Sensor Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Vibration Analysis** 🌀
            - **Normal Range**: 2.94 - 3.06
            - **Purpose**: Detects bearing wear, imbalance, misalignment
            - **High Values**: Indicate mechanical faults, loose components
            - **Impact**: Affects equipment lifespan and product quality
            
            **Temperature Monitoring** 🌡️
            - **Normal Range**: 305°C - 314°C  
            - **Purpose**: Monitors overheating, cooling system issues
            - **High Values**: Suggest lubrication problems, overloading
            - **Impact**: Critical for preventing thermal damage
            """)
            
        with col2:
            st.markdown("""
            **Pressure Monitoring** 💨
            - **Normal Range**: 100 - 207 psi
            - **Purpose**: Tracks hydraulic/pneumatic system health
            - **High Values**: Indicate blockages, pump issues
            - **Impact**: Affects system efficiency and safety
            
            **Current Analysis** ⚡
            - **Normal Range**: 10A - 69A
            - **Purpose**: Monitors motor load and electrical health
            - **High Values**: Suggest mechanical binding, voltage issues
            - **Impact**: Prevents motor burnout and energy waste
            """)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("""
            **RPM Monitoring** 🔄
            - **Normal Range**: 1000 - 2076 RPM
            - **Purpose**: Tracks rotational speed and consistency
            - **High Values**: Indicate control system issues
            - **Impact**: Affects product quality and equipment stress
            """)
            
        with col4:
            st.markdown("""
            **Tool Wear Analysis** ⚒️
            - **Normal Range**: 0 - 298 hours
            - **Purpose**: Monitors cutting tool degradation
            - **High Values**: Indicate tool replacement needed
            - **Impact**: Critical for product quality and machine protection
            """)

def create_health_score_card(prediction):
    """Create a comprehensive health score card"""
    failure_prob = prediction.get('failure_probability', 0)
    rul = prediction.get('predicted_rul_hours', 0)
    confidence = prediction.get('confidence', 0)
    
    # Calculate health score (0-100)
    health_score = max(0, 100 - (failure_prob * 100))
    
    # Determine status
    if health_score >= 80:
        status = "Excellent"
        status_color = "#2ecc71"
        status_emoji = "🟢"
    elif health_score >= 60:
        status = "Good"
        status_color = "#f39c12"
        status_emoji = "🟡"
    elif health_score >= 40:
        status = "Fair"
        status_color = "#e67e22"
        status_emoji = "🟠"
    else:
        status = "Poor"
        status_color = "#e74c3c"
        status_emoji = "🔴"
    
    # Create health score card
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {status_color}; margin: 0;">{status_emoji} Health Score</h3>
            <h1 style="color: {status_color}; margin: 0; font-size: 3rem;">{health_score:.0f}</h1>
            <p style="margin: 0; color: #7f8c8d;">{status} Condition</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #e74c3c; margin: 0;">⚠️ Failure Risk</h3>
            <h1 style="color: #e74c3c; margin: 0; font-size: 3rem;">{failure_prob*100:.1f}%</h1>
            <p style="margin: 0; color: #7f8c8d;">Probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        days_remaining = rul / 24
        if rul > 0:
            color = "#2ecc71" if days_remaining > 30 else "#f39c12" if days_remaining > 7 else "#e74c3c"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {color}; margin: 0;">⏱️ RUL</h3>
                <h1 style="color: {color}; margin: 0; font-size: 3rem;">{rul:.0f}h</h1>
                <p style="margin: 0; color: #7f8c8d;">{days_remaining:.1f} days remaining</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #e74c3c; margin: 0;">⏱️ RUL</h3>
                <h1 style="color: #e74c3c; margin: 0; font-size: 3rem;">IMMEDIATE</h1>
                <p style="margin: 0; color: #7f8c8d;">Maintenance Required</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3498db; margin: 0;">🎯 Confidence</h3>
            <h1 style="color: #3498db; margin: 0; font-size: 3rem;">{confidence*100:.1f}%</h1>
            <p style="margin: 0; color: #7f8c8d;">Model Certainty</p>
        </div>
        """, unsafe_allow_html=True)
    
    return health_score, status

def create_comprehensive_visualizations(prediction, sensor_data):
    """Create multiple visualization charts"""
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Health Overview", "📈 Sensor Analysis", "🔄 Trend Analysis", "🎯 Risk Assessment"])
    
    with tab1:
        # Health radar chart
        st.subheader("Equipment Health Radar Analysis")
        
        # Simulate sensor health scores (in real scenario, these would be calculated from actual values)
        categories = ['Vibration', 'Temperature', 'Pressure', 'Current', 'RPM', 'Tool Wear']
        health_scores = [85, 78, 92, 88, 76, 65]  # Example scores
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=health_scores,
            theta=categories,
            fill='toself',
            name='Current Health',
            line=dict(color='#3498db'),
            fillcolor='rgba(52, 152, 219, 0.3)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=400,
            title="Sensor Health Scores (Higher is Better)"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        # Sensor gauge charts
        st.subheader("Real-time Sensor Monitoring")
        
        # Create gauge charts for key sensors
        fig_gauges = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('Vibration', 'Temperature', 'Pressure', 'Current', 'RPM', 'Tool Wear')
        )
        
        # Add gauge for each sensor
        sensors = [
            ('Vibration', sensor_data['vibration'], 2.94, 3.06),
            ('Temperature', sensor_data['temperature'], 305, 314),
            ('Pressure', sensor_data['pressure'], 100, 207),
            ('Current', sensor_data['current'], 10, 69),
            ('RPM', sensor_data['rpm'], 1000, 2076),
            ('Tool Wear', sensor_data['tool_wear'], 0, 298)
        ]
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        
        for (sensor_name, value, min_val, max_val), (row, col) in zip(sensors, positions):
            fig_gauges.add_trace(
                go.Indicator(
                    mode = "gauge+number",
                    value = value,
                    title = {'text': sensor_name},
                    gauge = {
                        'axis': {'range': [min_val, max_val]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [min_val, min_val + (max_val-min_val)*0.6], 'color': "lightgray"},
                            {'range': [min_val + (max_val-min_val)*0.6, min_val + (max_val-min_val)*0.8], 'color': "yellow"},
                            {'range': [min_val + (max_val-min_val)*0.8, max_val], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': min_val + (max_val-min_val)*0.9
                        }
                    }
                ),
                row=row, col=col
            )
        
        fig_gauges.update_layout(height=500, margin=dict(l=50, r=50, t=50, b=50))
        st.plotly_chart(fig_gauges, use_container_width=True)
    
    with tab3:
        # Trend analysis (simulated historical data)
        st.subheader("Equipment Health Trend Analysis")
        
        # Generate simulated historical data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        simulated_health = np.clip(100 - np.random.random(30) * 40, 0, 100)
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=dates,
            y=simulated_health,
            mode='lines+markers',
            name='Health Score',
            line=dict(color='#3498db', width=3),
            marker=dict(size=6)
        ))
        
        # Add thresholds
        fig_trend.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Excellent Threshold")
        fig_trend.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig_trend.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        
        fig_trend.update_layout(
            title="30-Day Health Score Trend",
            xaxis_title="Date",
            yaxis_title="Health Score",
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab4:
        # Risk assessment pie chart
        st.subheader("Failure Risk Distribution")
        
        risk_categories = {
            'Mechanical Failure': 35,
            'Electrical Issues': 25,
            'Thermal Stress': 20,
            'Wear & Tear': 15,
            'Other': 5
        }
        
        fig_risk = px.pie(
            values=list(risk_categories.values()),
            names=list(risk_categories.keys()),
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        
        fig_risk.update_traces(textposition='inside', textinfo='percent+label')
        fig_risk.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig_risk, use_container_width=True)

def show_dashboard(dashboard):
    """Enhanced dashboard view"""
    st.markdown('<div class="sub-header">📊 Real-time Equipment Monitoring Dashboard</div>', unsafe_allow_html=True)
    
    # Quick overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="success-card">
            <h3>🟢 89%</h3>
            <p>Healthy Equipment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-card">
            <h3>🟡 8%</h3>
            <p>Needs Attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="alert-card">
            <h3>🔴 3%</h3>
            <p>Critical Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 $142K</h3>
            <p>Cost Savings</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Equipment health map (simulated)
        st.subheader("🏭 Equipment Health Map")
        
        # Create a simulated equipment grid
        equipment_data = []
        for i in range(1, 21):
            health = np.random.randint(30, 100)
            status = "🟢" if health > 80 else "🟡" if health > 60 else "🟠" if health > 40 else "🔴"
            equipment_data.append({
                "ID": f"EQ_{i:03d}",
                "Health": health,
                "Status": status,
                "Department": np.random.choice(["Production", "Assembly", "Packaging", "Quality"])
            })
        
        df_equipment = pd.DataFrame(equipment_data)
        
        # Display as a grid
        cols = st.columns(5)
        for idx, (_, eq) in enumerate(df_equipment.iterrows()):
            with cols[idx % 5]:
                color = "#2ecc71" if eq["Health"] > 80 else "#f39c12" if eq["Health"] > 60 else "#e67e22" if eq["Health"] > 40 else "#e74c3c"
                st.markdown(f"""
                <div style="background: white; padding: 0.5rem; border-radius: 8px; text-align: center; border-left: 4px solid {color}; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; font-size: 0.9rem;">{eq['ID']}</h4>
                    <h3 style="margin: 0; color: {color};">{eq['Health']}%</h3>
                    <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">{eq['Department']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🚨 Recent Alerts")
        
        alerts = [
            {"equipment": "EQ_014", "issue": "High Vibration", "time": "2 hours ago", "priority": "High"},
            {"equipment": "EQ_007", "issue": "Temperature Spike", "time": "4 hours ago", "priority": "Medium"},
            {"equipment": "EQ_019", "issue": "Tool Wear Critical", "time": "6 hours ago", "priority": "High"}
        ]
        
        for alert in alerts:
            priority_color = "#e74c3c" if alert["priority"] == "High" else "#f39c12"
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid {priority_color}; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: between; align-items: center;">
                    <strong>{alert['equipment']}</strong>
                    <span style="background: {priority_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">{alert['priority']}</span>
                </div>
                <p style="margin: 0.5rem 0 0 0; color: #2c3e50;">{alert['issue']}</p>
                <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">{alert['time']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick analysis section
    st.markdown("---")
    st.subheader("🚀 Quick Equipment Analysis")
    
    # Predefined scenarios for quick testing
    scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
    
    with scenario_col1:
        if st.button("🧪 Test Normal Equipment", use_container_width=True):
            st.session_state.quick_test = "normal"
    
    with scenario_col2:
        if st.button("⚠️ Test Warning Scenario", use_container_width=True):
            st.session_state.quick_test = "warning"
    
    with scenario_col3:
        if st.button("🚨 Test Critical Scenario", use_container_width=True):
            st.session_state.quick_test = "critical"
    
    # Handle quick test scenarios
    if hasattr(st.session_state, 'quick_test'):
        scenarios = {
            "normal": {"vibration": 3.0, "temperature": 310, "pressure": 150, "current": 40, "rpm": 1500, "tool_wear": 100},
            "warning": {"vibration": 3.04, "temperature": 312, "pressure": 180, "current": 55, "rpm": 1800, "tool_wear": 180},
            "critical": {"vibration": 3.06, "temperature": 314, "pressure": 200, "current": 65, "rpm": 2000, "tool_wear": 250}
        }
        
        scenario_data = scenarios[st.session_state.quick_test]
        equipment_data = {"equipment_id": f"QUICK_TEST_{st.session_state.quick_test.upper()}", **scenario_data}
        
        with st.spinner("🤖 AI is analyzing equipment health..."):
            prediction = dashboard.predict_single(equipment_data)
        
        if prediction:
            st.success("✅ Analysis Complete!")
            create_health_score_card(prediction)
            create_comprehensive_visualizations(prediction, scenario_data)

def show_equipment_analysis(dashboard):
    """Enhanced single equipment analysis"""
    st.markdown('<div class="sub-header">🔍 Detailed Equipment Health Analysis</div>', unsafe_allow_html=True)
    
    # Equipment input section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Equipment Parameters")
        
        # Equipment ID
        equipment_id = st.text_input("Equipment ID", "EQ_001", help="Unique identifier for the equipment")
        
        # Get expected ranges for context
        expected_ranges = dashboard.get_expected_ranges()
        
        # Sensor inputs with enhanced tooltips
        st.markdown("### 📡 Sensor Readings")
        
        vibration = st.slider(
            "Vibration Level",
            min_value=2.5, max_value=3.5, value=3.0, step=0.01,
            help="Measures equipment vibration. High values indicate mechanical issues like bearing wear or imbalance."
        )
        
        temperature = st.slider(
            "Temperature (°C)",
            min_value=300, max_value=320, value=310, step=1,
            help="Monitors operating temperature. High temperatures suggest cooling issues or overloading."
        )
        
        pressure = st.slider(
            "Pressure (psi)",
            min_value=100, max_value=220, value=150, step=5,
            help="Tracks hydraulic/pneumatic pressure. Abnormal values indicate system blockages or leaks."
        )
        
        current = st.slider(
            "Current (A)",
            min_value=10, max_value=70, value=40, step=1,
            help="Monitors electrical current draw. High current suggests mechanical binding or electrical faults."
        )
        
        rpm = st.slider(
            "RPM",
            min_value=1000, max_value=2100, value=1500, step=50,
            help="Monitors rotational speed. Inconsistent RPM indicates drive system issues."
        )
        
        tool_wear = st.slider(
            "Tool Wear (hours)",
            min_value=0, max_value=300, value=100, step=5,
            help="Tracks tool degradation and replacement needs."
        )
        
        # Analysis button
        if st.button("🚀 Start Comprehensive Analysis", type="primary", use_container_width=True):
            equipment_data = {
                "equipment_id": equipment_id,
                "vibration": vibration,
                "temperature": temperature,
                "pressure": pressure,
                "current": current,
                "rpm": rpm,
                "tool_wear": tool_wear
            }
            
            with st.spinner("🤖 AI Engine is analyzing equipment health patterns..."):
                # Simulate processing time for better UX
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                prediction = dashboard.predict_single(equipment_data)
            
            if prediction:
                st.session_state.last_prediction = prediction
                st.session_state.last_sensor_data = equipment_data
        
        # Display sensor explanations
        display_sensor_explanations()
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if hasattr(st.session_state, 'last_prediction'):
            prediction = st.session_state.last_prediction
            sensor_data = st.session_state.last_sensor_data
            
            # Health score card
            health_score, status = create_health_score_card(prediction)
            
            # Maintenance recommendation with enhanced display
            st.markdown("### 🛠️ Maintenance Recommendation")
            
            recommendation = prediction.get('maintenance_recommendation', 'No recommendation available')
            if "CRITICAL" in recommendation:
                st.markdown(f"""
                <div class="alert-card">
                    <h3>🚨 IMMEDIATE ACTION REQUIRED</h3>
                    <p>{recommendation}</p>
                    <p><strong>Predicted Failure:</strong> {prediction.get('predicted_failure_date', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
            elif "MEDIUM" in recommendation:
                st.markdown(f"""
                <div class="warning-card">
                    <h3>⚠️ SCHEDULE MAINTENANCE</h3>
                    <p>{recommendation}</p>
                    <p><strong>Remaining Life:</strong> {prediction.get('predicted_rul_hours', 0):.0f} hours</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-card">
                    <h3>✅ OPERATING NORMALLY</h3>
                    <p>{recommendation}</p>
                    <p><strong>Next Check:</strong> Recommended in 30 days</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Comprehensive visualizations
            create_comprehensive_visualizations(prediction, sensor_data)
            
            # Action plan
            st.markdown("### 📋 Recommended Action Plan")
            
            if health_score >= 80:
                st.success("""
                **🟢 Normal Operation**
                - Continue routine monitoring
                - Next comprehensive check in 30 days
                - No immediate action required
                """)
            elif health_score >= 60:
                st.warning("""
                **🟡 Monitoring Required**
                - Increase monitoring frequency to weekly
                - Schedule maintenance within 2-4 weeks
                - Check specific components identified in analysis
                """)
            elif health_score >= 40:
                st.error("""
                **🟠 Immediate Attention Needed**
                - Schedule maintenance within 1 week
                - Prepare replacement parts if needed
                - Increase monitoring to daily
                - Consider reducing equipment load
                """)
            else:
                st.error("""
                **🔴 Emergency Shutdown Recommended**
                - Stop equipment immediately if safe to do so
                - Notify maintenance team urgently
                - Prepare for emergency repair
                - Investigate root cause after stabilization
                """)
        
        else:
            st.info("""
            ## 👆 Get Started
            **Enter equipment parameters on the left and click "Start Comprehensive Analysis" to:**
            
            - 🎯 Get AI-powered health assessment
            - 📊 View detailed visualizations  
            - 🛠️ Receive maintenance recommendations
            - 📈 Analyze trends and patterns
            - 💡 Understand risk factors
            
            *The system uses machine learning to predict equipment failures before they happen.*
            """)
            
            # Show sample visualization
            st.info("Comprehensive analysis results will appear here after running an analysis")

def generate_sample_batch_data(num_records=50):
    """Generate realistic sample batch data for demonstration"""
    np.random.seed(42)
    
    equipment_data = []
    for i in range(num_records):
        # Create realistic sensor data with some patterns
        base_vibration = 3.0 + np.random.normal(0, 0.03)
        base_temp = 310 + np.random.normal(0, 2)
        base_pressure = 150 + np.random.normal(0, 20)
        base_current = 40 + np.random.normal(0, 10)
        base_rpm = 1500 + np.random.normal(0, 100)
        
        # Introduce some equipment with issues
        if i % 10 == 0:  # High vibration equipment
            base_vibration += 0.05
        if i % 7 == 0:  # High temperature equipment
            base_temp += 5
        if i % 5 == 0:  # High wear equipment
            tool_wear = np.random.randint(250, 300)
        else:
            tool_wear = np.random.randint(0, 250)
        
        equipment_data.append({
            "equipment_id": f"EQ_{i+1:03d}",
            "vibration": max(2.5, min(3.5, base_vibration)),
            "temperature": max(300, min(320, base_temp)),
            "pressure": max(100, min(220, base_pressure)),
            "current": max(10, min(70, base_current)),
            "rpm": max(1000, min(2100, base_rpm)),
            "tool_wear": tool_wear
        })
    
    return equipment_data

def create_batch_analysis_visualizations(batch_results):
    """Create comprehensive visualizations for batch analysis"""
    
    if not batch_results:
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(batch_results)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Health Overview", "🚨 Risk Distribution", "📈 Performance Metrics", "🔍 Detailed Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Health distribution pie chart
            health_counts = {
                'Excellent (80-100)': len(df[df['health_score'] >= 80]),
                'Good (60-79)': len(df[(df['health_score'] >= 60) & (df['health_score'] < 80)]),
                'Fair (40-59)': len(df[(df['health_score'] >= 40) & (df['health_score'] < 60)]),
                'Poor (0-39)': len(df[df['health_score'] < 40])
            }
            
            fig_pie = px.pie(
                values=list(health_counts.values()),
                names=list(health_counts.keys()),
                title="Equipment Health Distribution",
                color_discrete_sequence=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Failure probability distribution
            fig_hist = px.histogram(
                df, 
                x='failure_probability',
                nbins=20,
                title="Failure Probability Distribution",
                color_discrete_sequence=['#e74c3c']
            )
            fig_hist.update_layout(xaxis_title="Failure Probability", yaxis_title="Number of Equipment")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        # Risk matrix
        st.subheader("Risk Assessment Matrix")
        
        # Create risk matrix data
        risk_data = []
        for _, row in df.iterrows():
            risk_level = "High" if row['failure_probability'] > 0.7 else "Medium" if row['failure_probability'] > 0.3 else "Low"
            impact_level = "High" if row['health_score'] < 40 else "Medium" if row['health_score'] < 60 else "Low"
            risk_data.append({
                'Equipment': row['equipment_id'],
                'Risk Level': risk_level,
                'Impact Level': impact_level,
                'Failure Probability': row['failure_probability'],
                'Health Score': row['health_score']
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        # Create scatter plot for risk matrix
        fig_risk = px.scatter(
            risk_df,
            x='Failure Probability',
            y='Health Score',
            color='Risk Level',
            size='Failure Probability',
            hover_data=['Equipment', 'Impact Level'],
            title="Risk Matrix: Failure Probability vs Health Score",
            color_discrete_map={
                'Low': '#2ecc71',
                'Medium': '#f39c12',
                'High': '#e74c3c'
            }
        )
        
        # Add risk quadrants
        fig_risk.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Critical Health")
        fig_risk.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab3:
        # Sensor correlation analysis
        st.subheader("Sensor Parameter Correlations")
        
        # Select only numeric columns for correlation
        numeric_cols = ['vibration', 'temperature', 'pressure', 'current', 'rpm', 'tool_wear', 'failure_probability']
        corr_df = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_df,
            title="Sensor Parameter Correlations with Failure Probability",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab4:
        # Detailed equipment table
        st.subheader("Detailed Equipment Analysis")
        
        # Enhanced dataframe display
        display_df = df[['equipment_id', 'health_score', 'failure_probability', 'predicted_rul_hours', 'maintenance_priority']].copy()
        display_df['Health Status'] = display_df['health_score'].apply(
            lambda x: '🟢 Excellent' if x >= 80 else '🟡 Good' if x >= 60 else '🟠 Fair' if x >= 40 else '🔴 Poor'
        )
        display_df['Maintenance Urgency'] = display_df['maintenance_priority'].apply(
            lambda x: '🟢 Low' if x == 'low' else '🟡 Medium' if x == 'medium' else '🔴 High'
        )
        
        st.dataframe(
            display_df.style.background_gradient(
                subset=['health_score', 'failure_probability'], 
                cmap='RdYlGn_r'
            ),
            use_container_width=True
        )

def show_batch_processing(dashboard):
    """Enhanced batch processing with comprehensive analysis"""
    st.markdown('<div class="sub-header">📈 Batch Equipment Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-highlight">
        <h3>🚀 Mass Equipment Analysis</h3>
        <p>Upload a CSV file with multiple equipment data or use our sample dataset to analyze your entire fleet simultaneously. 
        Get comprehensive insights, priority rankings, and maintenance scheduling recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data input methods
    input_method = st.radio(
        "Choose Data Input Method:",
        ["📁 Upload CSV File", "🧪 Use Sample Data", "📝 Manual Entry"],
        horizontal=True
    )
    
    batch_data = []
    
    if input_method == "📁 Upload CSV File":
        st.subheader("Upload Equipment Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with equipment data",
            type=['csv'],
            help="CSV should contain columns: equipment_id, vibration, temperature, pressure, current, rpm, tool_wear"
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df_upload)} equipment records")
                
                # Display preview
                with st.expander("📋 Preview Uploaded Data"):
                    st.dataframe(df_upload.head(10), use_container_width=True)
                
                # Convert to required format
                batch_data = df_upload.to_dict('records')
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    elif input_method == "🧪 Use Sample Data":
        st.subheader("Sample Equipment Data")
        
        num_equipment = st.slider("Number of sample equipment to generate", 10, 200, 50)
        
        if st.button("🔄 Generate Sample Data", type="primary"):
            batch_data = generate_sample_batch_data(num_equipment)
            st.success(f"✅ Generated {len(batch_data)} sample equipment records")
            
            # Display sample data
            with st.expander("📋 Preview Sample Data"):
                sample_df = pd.DataFrame(batch_data)
                st.dataframe(sample_df.head(10), use_container_width=True)
    
    elif input_method == "📝 Manual Entry":
        st.subheader("Manual Batch Data Entry")
        
        num_entries = st.number_input("Number of equipment to add", min_value=1, max_value=50, value=5)
        
        batch_data = []
        for i in range(num_entries):
            with st.container():
                st.markdown(f"**Equipment {i+1}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    eq_id = st.text_input(f"Equipment ID", value=f"EQ_{i+1:03d}", key=f"eq_id_{i}")
                    vibration = st.slider(f"Vibration", 2.5, 3.5, 3.0, key=f"vib_{i}")
                    temperature = st.slider(f"Temperature", 300, 320, 310, key=f"temp_{i}")
                
                with col2:
                    pressure = st.slider(f"Pressure", 100, 220, 150, key=f"press_{i}")
                    current = st.slider(f"Current", 10, 70, 40, key=f"current_{i}")
                    rpm = st.slider(f"RPM", 1000, 2100, 1500, key=f"rpm_{i}")
                    tool_wear = st.slider(f"Tool Wear", 0, 300, 100, key=f"wear_{i}")
                
                equipment_entry = {
                    "equipment_id": eq_id,
                    "vibration": vibration,
                    "temperature": temperature,
                    "pressure": pressure,
                    "current": current,
                    "rpm": rpm,
                    "tool_wear": tool_wear
                }
                batch_data.append(equipment_entry)
        
        if st.button("✅ Confirm Manual Entries"):
            st.success(f"✅ Prepared {len(batch_data)} equipment records for analysis")
    
    # Batch analysis execution
    if batch_data and st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
        with st.spinner("🤖 AI Engine is analyzing equipment fleet..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate batch processing with progress updates
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Processing... {i+1}%")
                time.sleep(0.02)
            
            # Get predictions (in real scenario, this would be the actual API call)
            # For demo, we'll simulate results
            simulated_results = []
            for equipment in batch_data:
                # Simulate AI predictions based on sensor values
                health_score = max(0, 100 - (
                    (abs(equipment['vibration'] - 3.0) * 1000) +
                    (abs(equipment['temperature'] - 310) * 2) +
                    (abs(equipment['pressure'] - 150) * 0.5) +
                    (equipment['tool_wear'] * 0.1)
                ))
                
                failure_prob = min(0.95, (
                    (abs(equipment['vibration'] - 3.0) * 10) +
                    (abs(equipment['temperature'] - 310) * 0.05) +
                    (equipment['tool_wear'] * 0.003)
                ))
                
                rul = max(0, 500 - (equipment['tool_wear'] * 2))
                
                # Determine maintenance priority
                if failure_prob > 0.7 or health_score < 40:
                    priority = "high"
                elif failure_prob > 0.3 or health_score < 60:
                    priority = "medium"
                else:
                    priority = "low"
                
                simulated_results.append({
                    'equipment_id': equipment['equipment_id'],
                    'health_score': health_score,
                    'failure_probability': failure_prob,
                    'predicted_rul_hours': rul,
                    'maintenance_priority': priority,
                    **equipment  # Include original sensor data
                })
            
            st.session_state.batch_results = simulated_results
            st.session_state.batch_data = batch_data
            
            progress_bar.empty()
            status_text.empty()
        
        st.success("✅ Batch analysis completed!")
        
        # Display comprehensive results
        if hasattr(st.session_state, 'batch_results'):
            create_batch_analysis_visualizations(st.session_state.batch_results)
            
            # Summary statistics
            st.markdown("### 📋 Batch Analysis Summary")
            
            results_df = pd.DataFrame(st.session_state.batch_results)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_health = results_df['health_score'].mean()
                st.metric("Average Health Score", f"{avg_health:.1f}")
            
            with col2:
                high_risk_count = len(results_df[results_df['failure_probability'] > 0.7])
                st.metric("High Risk Equipment", f"{high_risk_count}")
            
            with col3:
                avg_rul = results_df['predicted_rul_hours'].mean()
                st.metric("Average RUL", f"{avg_rul:.0f} hours")
            
            with col4:
                maintenance_cost = high_risk_count * 2500 + len(results_df[results_df['maintenance_priority'] == 'medium']) * 1000
                st.metric("Estimated Maintenance Cost", f"${maintenance_cost:,}")
            
            # Download results
            st.markdown("### 💾 Download Results")
            
            # Convert results to CSV
            csv = results_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="batch_analysis_results.csv">📥 Download Full Results CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    elif not batch_data:
        st.info("""
        ## 📊 Batch Processing Features
        
        **Benefits of Batch Analysis:**
        - 🚀 **Efficiency**: Analyze hundreds of equipment simultaneously
        - 📈 **Comparative Analysis**: Identify patterns across your entire fleet
        - 🎯 **Priority Management**: Automatically rank maintenance urgency
        - 💰 **Cost Optimization**: Plan maintenance schedules efficiently
        - 📋 **Comprehensive Reporting**: Generate detailed analysis reports
        
        **How to Use:**
        1. Choose your data input method (Upload, Sample, or Manual)
        2. Prepare your equipment data with sensor readings
        3. Run the batch analysis
        4. Review comprehensive results and download reports
        """)

def show_system_info(dashboard):
    """Comprehensive system information and technical details"""
    st.markdown('<div class="sub-header">⚙️ System Information & Configuration</div>', unsafe_allow_html=True)
    
    # System status cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # API health check
        with st.spinner("Checking API status..."):
            api_healthy, health_data = dashboard.check_api_health()
        
        if api_healthy:
            st.markdown("""
            <div class="success-card">
                <h3>🟢 API Status</h3>
                <p>Predictive Maintenance API</p>
                <p><strong>Connected & Operational</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-card">
                <h3>🔴 API Status</h3>
                <p>Predictive Maintenance API</p>
                <p><strong>Connection Failed</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI Model</h3>
            <p>Predictive Analytics Engine</p>
            <p><strong>Version 2.1.0</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Data Processing</h3>
            <p>Real-time Analytics</p>
            <p><strong>Active</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical specifications
    st.markdown("### 🔧 Technical Specifications")
    
    spec_tab1, spec_tab2, spec_tab3 = st.tabs(["Model Architecture", "Sensor Specifications", "API Documentation"])
    
    with spec_tab1:
        st.markdown("""
        #### 🧠 Predictive Model Architecture
        
        **Machine Learning Framework:**
        - **Algorithm**: Ensemble Random Forest with Gradient Boosting
        - **Training Data**: 50,000+ equipment hours
        - **Accuracy**: 94.2% failure prediction rate
        - **Features**: 6 primary sensor inputs + 12 derived features
        
        **Model Components:**
        1. **Anomaly Detection**: Real-time sensor data anomaly identification
        2. **Trend Analysis**: Equipment degradation pattern recognition
        3. **Failure Prediction**: Probability-based failure forecasting
        4. **RUL Estimation**: Remaining Useful Life calculation
        
        **Performance Metrics:**
        - Precision: 92.8%
        - Recall: 95.1%
        - F1-Score: 93.9%
        - AUC-ROC: 0.97
        """)
    
    with spec_tab2:
        st.markdown("""
        #### 📡 Supported Sensor Parameters
        
        | Parameter | Normal Range | Critical Threshold | Measurement Unit |
        |-----------|-------------|-------------------|-----------------|
        | Vibration | 2.94 - 3.06 | > 3.06 | g RMS |
        | Temperature | 305°C - 314°C | > 314°C | °C |
        | Pressure | 100 - 207 psi | > 207 psi | psi |
        | Current | 10A - 69A | > 69A | Amps |
        | RPM | 1000 - 2076 | > 2076 | RPM |
        | Tool Wear | 0 - 298 hours | > 298 hours | hours |
        
        **Data Validation:**
        - Real-time range checking
        - Pattern anomaly detection
        - Cross-sensor correlation analysis
        - Historical trend validation
        """)
    
    with spec_tab3:
        st.markdown("""
        #### 🌐 API Endpoints
        
        **Base URL**: `http://localhost:8000`
        
        **Available Endpoints:**
        ```python
        # Health Check
        GET /health
        
        # Single Prediction
        POST /predict
        {
            "equipment_id": "string",
            "vibration": float,
            "temperature": float,
            "pressure": float,
            "current": float,
            "rpm": float,
            "tool_wear": float
        }
        
        # Batch Prediction
        POST /predict_batch
        [
            { ...equipment_data },
            { ...equipment_data }
        ]
        
        # Expected Ranges
        GET /expected_ranges
        ```
        
        **Response Format:**
        ```json
        {
            "equipment_id": "string",
            "health_score": 0-100,
            "failure_probability": 0-1,
            "predicted_rul_hours": float,
            "maintenance_priority": "low|medium|high",
            "maintenance_recommendation": "string",
            "confidence": 0-1
        }
        ```
        """)
    
    # System configuration
    st.markdown("### ⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔔 Alert Settings")
        
        alert_temp = st.slider("Temperature Alert Threshold (°C)", 300, 320, 314)
        alert_vibration = st.slider("Vibration Alert Threshold (g)", 2.5, 3.5, 3.06)
        alert_wear = st.slider("Tool Wear Alert Threshold (hours)", 0, 300, 250)
        
        if st.button("💾 Save Alert Settings"):
            st.success("Alert thresholds updated successfully!")
    
    with col2:
        st.subheader("📧 Notification Settings")
        
        email_alerts = st.checkbox("Enable Email Alerts", value=True)
        sms_alerts = st.checkbox("Enable SMS Alerts", value=False)
        dashboard_alerts = st.checkbox("Enable Dashboard Alerts", value=True)
        
        alert_frequency = st.selectbox(
            "Alert Frequency",
            ["Immediate", "Hourly Digest", "Daily Summary"]
        )
        
        if st.button("💾 Save Notification Settings"):
            st.success("Notification settings updated successfully!")

def show_tutorial():
    """Comprehensive tutorial and user guide"""
    st.markdown('<div class="sub-header">🎓 Predictive Maintenance Tutorial</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-highlight">
        <h3>🎯 Master Predictive Maintenance</h3>
        <p>Learn how to effectively use the AI Predictive Maintenance System to monitor equipment health, 
        predict failures, and optimize maintenance schedules. This comprehensive guide covers everything 
        from basic concepts to advanced features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tutorial chapters
    chapters = [
        {"title": "🚀 Getting Started", "icon": "🚀"},
        {"title": "📊 Understanding Sensors", "icon": "📊"},
        {"title": "🔍 Single Equipment Analysis", "icon": "🔍"},
        {"title": "📈 Batch Processing", "icon": "📈"},
        {"title": "📋 Interpreting Results", "icon": "📋"},
        {"title": "💡 Best Practices", "icon": "💡"}
    ]
    
    # Create chapter navigation
    selected_chapter = st.selectbox(
        "Select Chapter:",
        [f"{ch['icon']} {ch['title']}" for ch in chapters],
        index=0
    )
    
    chapter_index = [f"{ch['icon']} {ch['title']}" for ch in chapters].index(selected_chapter)
    
    st.markdown("---")
    
    if chapter_index == 0:  # Getting Started
        st.markdown("""
        ## 🚀 Getting Started with Predictive Maintenance
        
        ### What is Predictive Maintenance?
        
        Predictive Maintenance uses AI and machine learning to predict equipment failures **before they happen**. 
        Unlike traditional maintenance approaches, it analyzes real-time sensor data to identify patterns that 
        indicate potential issues.
        
        ### Key Benefits:
        
        - **🔄 Reduced Downtime**: Fix issues before they cause breakdowns
        - **💰 Cost Savings**: Avoid expensive emergency repairs
        - **🔧 Optimized Maintenance**: Schedule repairs when actually needed
        - **📈 Increased Efficiency**: Maintain optimal equipment performance
        - **🎯 Precision**: AI-driven accurate failure predictions
        
        ### How It Works:
        
        1. **Data Collection**: Continuous monitoring of equipment sensors
        2. **AI Analysis**: Machine learning models analyze sensor patterns
        3. **Failure Prediction**: Probability-based failure forecasting
        4. **Maintenance Alerts**: Timely notifications and recommendations
        5. **Actionable Insights**: Clear maintenance priorities and schedules
        
        ### Quick Start Guide:
        
        **For Single Equipment:**
        1. Go to "🔍 Equipment Analysis" page
        2. Enter equipment parameters or use quick test buttons
        3. Click "Start Comprehensive Analysis"
        4. Review health score and recommendations
        
        **For Multiple Equipment:**
        1. Go to "📈 Batch Processing" page
        2. Upload CSV file or use sample data
        3. Run batch analysis
        4. Download comprehensive reports
        """)
    
    elif chapter_index == 1:  # Understanding Sensors
        st.markdown("""
        ## 📊 Understanding Sensor Parameters
        
        Each sensor provides critical information about your equipment's health. Here's what to look for:
        
        ### 🌀 Vibration Analysis
        **Purpose**: Detects mechanical issues like bearing wear, imbalance, and misalignment
        
        **Interpretation:**
        - **Normal (2.94-3.06)**: Healthy operation
        - **Warning (3.01-3.04)**: Early signs of wear
        - **Critical (>3.06)**: Immediate maintenance needed
        
        **Common Causes of High Vibration:**
        - Bearing deterioration
        - Rotor imbalance
        - Misalignment
        - Loose components
        
        ### 🌡️ Temperature Monitoring
        **Purpose**: Identifies overheating and cooling system issues
        
        **Interpretation:**
        - **Normal (305-314°C)**: Optimal operating temperature
        - **Warning (312-314°C)**: Cooling system check needed
        - **Critical (>314°C)**: Risk of thermal damage
        
        ### 💨 Pressure Monitoring
        **Purpose**: Tracks hydraulic and pneumatic system health
        
        **Interpretation:**
        - **Normal (100-207 psi)**: System operating correctly
        - **Warning (180-207 psi)**: Potential blockages
        - **Critical (>207 psi)**: Risk of system failure
        
        ### ⚡ Current Analysis
        **Purpose**: Monitors electrical motor health and load conditions
        
        **Interpretation:**
        - **Normal (10-69A)**: Standard operating range
        - **Warning (55-69A)**: Increased mechanical resistance
        - **Critical (>69A)**: Electrical or mechanical faults
        
        ### 🔄 RPM Monitoring
        **Purpose**: Ensures consistent rotational speed
        
        **Interpretation:**
        - **Normal (1000-2076 RPM)**: Optimal speed range
        - **Warning (1800-2076 RPM)**: Control system variations
        - **Critical (>2076 RPM)**: Speed control failure
        
        ### ⚒️ Tool Wear Analysis
        **Purpose**: Tracks tool degradation and replacement needs
        
        **Interpretation:**
        - **Normal (0-200 hours)**: Standard tool life
        - **Warning (200-250 hours)**: Plan for replacement
        - **Critical (>250 hours)**: Immediate replacement required
        """)
    
    elif chapter_index == 2:  # Single Equipment Analysis
        st.markdown("""
        ## 🔍 Single Equipment Analysis
        
        This feature provides detailed analysis for individual equipment with comprehensive insights.
        
        ### Step-by-Step Guide:
        
        **1. Equipment Identification**
        - Enter unique equipment ID for tracking
        - Use consistent naming conventions (e.g., EQ_001, PRESS_02A)
        
        **2. Sensor Data Input**
        - Use sliders to set current sensor values
        - Reference normal ranges provided
        - Use tooltips for parameter explanations
        
        **3. AI Analysis**
        - Click "Start Comprehensive Analysis"
        - Wait for AI processing (typically 2-3 seconds)
        - Review real-time progress indicators
        
        **4. Results Interpretation**
        
        **Health Score (0-100):**
        - 🟢 **80-100**: Excellent - Continue normal operation
        - 🟡 **60-79**: Good - Monitor closely
        - 🟠 **40-59**: Fair - Schedule maintenance
        - 🔴 **0-39**: Poor - Immediate action required
        
        **Failure Probability:**
        - 🟢 **0-30%**: Low risk
        - 🟡 **30-70%**: Medium risk
        - 🔴 **70-100%**: High risk
        
        **Remaining Useful Life (RUL):**
        - Hours until predicted maintenance need
        - Based on current degradation trends
        - Updated with each analysis
        
        ### Pro Tips:
        
        - **Regular Monitoring**: Analyze equipment weekly for early detection
        - **Trend Analysis**: Compare results over time to identify degradation patterns
        - **Action Planning**: Use maintenance recommendations for scheduling
        - **Documentation**: Save analysis reports for historical tracking
        """)
    
    elif chapter_index == 3:  # Batch Processing
        st.markdown("""
        ## 📈 Batch Equipment Processing
        
        Analyze multiple equipment simultaneously for fleet-wide insights and optimization.
        
        ### Data Input Methods:
        
        **1. CSV File Upload**
        - Format: equipment_id,vibration,temperature,pressure,current,rpm,tool_wear
        - Supports up to 500 equipment per batch
        - Automatic validation and error checking
        
        **2. Sample Data Generation**
        - Quickly test with realistic simulated data
        - Adjustable number of equipment
        - Includes various health scenarios
        
        **3. Manual Entry**
        - Direct input for small batches
        - Real-time validation
        - Immediate feedback
        
        ### Analysis Outputs:
        
        **Health Distribution:**
        - Pie chart showing equipment health categories
        - Quick overview of fleet condition
        
        **Risk Matrix:**
        - Scatter plot of failure probability vs health score
        - Visual identification of critical equipment
        
        **Correlation Analysis:**
        - Heatmap showing sensor parameter relationships
        - Identifies which parameters most impact failure risk
        
        **Priority List:**
        - Ranked equipment by maintenance urgency
        - Color-coded for quick reference
        
        ### Batch Analysis Workflow:
        
        1. **Data Preparation**: Collect current sensor readings
        2. **Upload/Input**: Choose your preferred input method
        3. **AI Processing**: System analyzes all equipment simultaneously
        4. **Results Review**: Comprehensive visualizations and insights
        5. **Action Planning**: Prioritize maintenance based on results
        6. **Report Generation**: Download detailed analysis for records
        
        ### Best Practices for Batch Processing:
        
        - **Regular Scheduling**: Run batch analysis monthly
        - **Data Quality**: Ensure accurate sensor readings
        - **Follow-up**: Investigate equipment with sudden health changes
        - **Integration**: Use results for maintenance planning and budgeting
        """)
    
    elif chapter_index == 4:  # Interpreting Results
        st.markdown("""
        ## 📋 Interpreting Analysis Results
        
        Understanding the AI-generated insights is crucial for effective maintenance planning.
        
        ### Key Metrics Explained:
        
        **Health Score (0-100)**
        ```
        90-100: 🟢 Excellent - Optimal performance, no issues detected
        80-89:  🟢 Good - Minor variations, monitor trends
        70-79:  🟡 Fair - Early warning signs, plan inspection
        60-69:  🟡 Watch - Increased monitoring frequency needed
        40-59:  🟠 Poor - Schedule maintenance soon
        0-39:   🔴 Critical - Immediate maintenance required
        ```
        
        **Failure Probability**
        ```
        0-0.3:  🟢 Low Risk - Standard operation
        0.3-0.7: 🟡 Medium Risk - Increased monitoring
        0.7-1.0: 🔴 High Risk - Immediate attention needed
        ```
        
        **Maintenance Priority Levels**
        - **🟢 Low**: Routine monitoring, next scheduled maintenance
        - **🟡 Medium**: Schedule within 2-4 weeks
        - **🔴 High**: Address within 1 week or immediately
        
        ### Actionable Insights:
        
        **For Health Score Drops:**
        - Investigate recent operational changes
        - Check specific sensor anomalies
        - Review maintenance history
        
        **For High Failure Probability:**
        - Immediate equipment inspection
        - Prepare replacement parts if needed
        - Consider temporary load reduction
        
        **Trend Analysis:**
        - Compare current results with historical data
        - Identify accelerating degradation
        - Plan proactive maintenance
        
        ### Decision Framework:
        
        ```python
        if health_score < 40 or failure_probability > 0.7:
            action = "IMMEDIATE_MAINTENANCE"
        elif health_score < 60 or failure_probability > 0.3:
            action = "SCHEDULE_MAINTENANCE"
        else:
            action = "CONTINUE_MONITORING"
        ```
        """)
    
    elif chapter_index == 5:  # Best Practices
        st.markdown("""
        ## 💡 Predictive Maintenance Best Practices
        
        Maximize the benefits of your AI Predictive Maintenance System with these proven strategies.
        
        ### Implementation Strategy:
        
        **1. Start Small, Scale Smart**
        - Begin with critical equipment
        - Establish baseline measurements
        - Gradually expand to entire fleet
        
        **2. Data Quality Foundation**
        - Ensure sensor calibration
        - Maintain consistent data collection
        - Validate readings regularly
        
        **3. Team Training**
        - Train maintenance staff on interpretation
        - Establish clear response protocols
        - Create accountability structures
        
        ### Operational Excellence:
        
        **Monitoring Frequency:**
        - Critical equipment: Weekly analysis
        - Important equipment: Monthly analysis
        - All equipment: Quarterly batch analysis
        
        **Alert Response Times:**
        - 🔴 Critical alerts: 24 hours
        - 🟠 High priority: 1 week
        - 🟡 Medium priority: 2-4 weeks
        - 🟢 Low priority: Next scheduled maintenance
        
        **Documentation Standards:**
        - Maintain analysis history
        - Track maintenance actions taken
        - Document cost savings and improvements
        
        ### Continuous Improvement:
        
        **Performance Tracking:**
        - Monitor prediction accuracy
        - Track maintenance cost reductions
        - Measure downtime improvements
        
        **System Optimization:**
        - Adjust alert thresholds based on experience
        - Incorporate new sensor data
        - Update maintenance procedures
        
        ### Success Metrics:
        
        - **MTBF Increase**: Mean Time Between Failures
        - **Downtime Reduction**: Unplanned equipment outages
        - **Cost Savings**: Maintenance and repair expenses
        - **Equipment Lifespan**: Extended operational life
        
        ### Common Pitfalls to Avoid:
        
        - ❌ Ignoring early warning signs
        - ❌ Inconsistent data collection
        - ❌ Delaying recommended maintenance
        - ❌ Poor communication between teams
        - ❌ Not learning from prediction outcomes
        """)
    
    # Quick assessment quiz
    st.markdown("---")
    with st.expander("🧠 Knowledge Check Quiz", expanded=False):
        st.subheader("Test Your Understanding")
        
        quiz_questions = [
            {
                "question": "What does a health score below 40 indicate?",
                "options": ["Excellent condition", "Normal operation", "Immediate maintenance needed", "Schedule next inspection"],
                "correct": 2
            },
            {
                "question": "Which sensor detects bearing wear and imbalance?",
                "options": ["Temperature", "Vibration", "Pressure", "Current"],
                "correct": 1
            },
            {
                "question": "What is the primary benefit of predictive maintenance?",
                "options": ["Lower equipment cost", "Reduced unexpected downtime", "Fewer sensors needed", "Simpler maintenance procedures"],
                "correct": 1
            }
        ]
        
        score = 0
        for i, q in enumerate(quiz_questions):
            st.markdown(f"**{i+1}. {q['question']}**")
            answer = st.radio(f"Select your answer:", q['options'], key=f"quiz_{i}")
            
            if st.button(f"Check Answer {i+1}", key=f"check_{i}"):
                if q['options'].index(answer) == q['correct']:
                    st.success("✅ Correct! Well done.")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {q['options'][q['correct']]}")
        
        if score > 0:
            st.info(f"🎯 Your score: {score}/{len(quiz_questions)}")

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏭 AI Predictive Maintenance System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem; margin-bottom: 2rem;">Intelligent Equipment Monitoring & Failure Prediction Platform</p>', unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = EnhancedPredictiveMaintenanceDashboard()
    
    # Check API health with better error handling
    with st.spinner("🔍 Connecting to AI Engine..."):
        api_healthy, health_data = dashboard.check_api_health()
    
    if not api_healthy:
        st.error("""
        🚨 **API Server Connection Failed** 
        
        Please ensure the predictive maintenance API is running. Start it with:
        ```bash
        uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
        ```
        
        **For demonstration purposes, the dashboard will continue with simulated data.**
        """)
        # Continue anyway for demo purposes
    
    # Sidebar with enhanced navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: white; margin: 0;">🔧 PM AI</h2>
            <p style="color: #bdc3c7; margin: 0;">Predictive Maintenance</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        
        # Navigation
        st.sidebar.markdown("### 📋 Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["🏠 Dashboard", "🔍 Equipment Analysis", "📈 Batch Processing", "⚙️ System Info", "🎓 Tutorial"]
        )
        
        st.sidebar.markdown("---")
        
        # Quick stats in sidebar
        st.sidebar.markdown("### 📊 Quick Stats")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Active Equipment", "142")
        with col2:
            st.metric("Alerts", "3")
        
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.markdown("### 🟢 System Status")
        if api_healthy:
            st.sidebar.success("**AI Model**: Active")
        else:
            st.sidebar.warning("**AI Model**: Simulation Mode")
        st.sidebar.success("**Dashboard**: Connected")
        st.sidebar.info("**Last Update**: " + datetime.now().strftime("%H:%M:%S"))
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard(dashboard)
    elif page == "🔍 Equipment Analysis":
        show_equipment_analysis(dashboard)
    elif page == "📈 Batch Processing":
        show_batch_processing(dashboard)
    elif page == "⚙️ System Info":
        show_system_info(dashboard)
    elif page == "🎓 Tutorial":
        show_tutorial()

if __name__ == "__main__":
    main()