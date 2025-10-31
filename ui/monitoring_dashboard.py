"""
Monitoring Dashboard - Grafana-style system monitoring

Real-time metrics for:
- Inference latency
- Throughput
- Confidence distributions
- Error rates
- Queue depths
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="RaeLM Monitoring",
    page_icon="📈",
    layout="wide"
)


def main():
    """Main monitoring dashboard."""
    st.title("🔍 RaeLM System Monitoring")
    
    # Auto-refresh
    refresh_rate = st.sidebar.selectbox("Refresh Rate", ["5s", "10s", "30s", "1m"], index=1)
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 5 minutes", "Last 15 minutes", "Last hour", "Last 24 hours"],
        index=1
    )
    
    # System health
    st.header("System Health")
    health_cols = st.columns(5)
    
    with health_cols[0]:
        st.metric("API Status", "🟢 Healthy", "")
    
    with health_cols[1]:
        st.metric("Celery Workers", "4/4", "")
    
    with health_cols[2]:
        st.metric("Database", "🟢 Online", "")
    
    with health_cols[3]:
        st.metric("Redis", "🟢 Online", "")
    
    with health_cols[4]:
        st.metric("Model Server", "🟢 Ready", "")
    
    st.markdown("---")
    
    # Performance metrics
    st.header("Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Inference Latency")
        
        # Generate mock time series data
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=100,
            freq='1T'
        )
        
        latency_data = pd.DataFrame({
            'timestamp': timestamps,
            'p50': np.random.normal(1.2, 0.2, 100),
            'p95': np.random.normal(2.5, 0.4, 100),
            'p99': np.random.normal(3.8, 0.6, 100)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=latency_data['timestamp'],
            y=latency_data['p50'],
            name='p50',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=latency_data['timestamp'],
            y=latency_data['p95'],
            name='p95',
            line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=latency_data['timestamp'],
            y=latency_data['p99'],
            name='p99',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Latency (seconds)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Throughput")
        
        throughput_data = pd.DataFrame({
            'timestamp': timestamps,
            'requests_per_min': np.random.poisson(25, 100)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=throughput_data['timestamp'],
            y=throughput_data['requests_per_min'],
            fill='tozeroy',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Requests/min",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ML metrics
    st.header("ML Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Confidence Trends")
        
        confidence_data = pd.DataFrame({
            'timestamp': timestamps[-20:],
            'avg_confidence': np.random.beta(9, 1, 20)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=confidence_data['timestamp'],
            y=confidence_data['avg_confidence'],
            mode='lines+markers',
            line=dict(color='purple')
        ))
        
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                     annotation_text="Threshold")
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Avg Confidence",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Error Rate")
        
        error_data = pd.DataFrame({
            'timestamp': timestamps[-20:],
            'error_rate': np.random.beta(1, 20, 20)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=error_data['timestamp'],
            y=error_data['error_rate'] * 100,
            mode='lines+markers',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Error Rate (%)",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Review Queue Depth")
        
        queue_data = pd.DataFrame({
            'timestamp': timestamps[-20:],
            'queue_depth': np.random.poisson(12, 20)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=queue_data['timestamp'],
            y=queue_data['queue_depth'],
            mode='lines+markers',
            line=dict(color='orange')
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Queue Depth",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Resource utilization
    st.header("Resource Utilization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", "45%", "-5%")
    
    with col2:
        st.metric("Memory Usage", "12.3 GB", "+0.8 GB")
    
    with col3:
        st.metric("GPU Usage", "78%", "+12%")
    
    with col4:
        st.metric("Disk Usage", "124 GB", "+2 GB")


if __name__ == "__main__":
    main()

