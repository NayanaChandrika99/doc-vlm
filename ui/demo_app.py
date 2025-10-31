"""
Streamlit Demo App - Interactive document extraction demo

Features:
- Upload document
- View processing status
- See extracted data with confidence scores
- Submit corrections
- View analytics
"""
import streamlit as st
import requests
import json
from PIL import Image
import pandas as pd
from pathlib import Path
import time

# Configure page
st.set_page_config(
    page_title="RaeLM Document Understanding",
    page_icon="📄",
    layout="wide"
)

# API configuration
API_BASE_URL = "http://localhost:8000"


def main():
    """Main app entry point."""
    st.title("🏥 RaeLM Medical Form Extraction")
    st.markdown("Extract structured data from medical forms with confidence scores")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Upload & Extract", "Review Results", "Analytics", "Settings"]
    )
    
    if page == "Upload & Extract":
        upload_page()
    elif page == "Review Results":
        review_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Settings":
        settings_page()


def upload_page():
    """Document upload and extraction page."""
    st.header("Upload Document")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a medical form (PDF, PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        # Display preview
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Document Preview")
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            else:
                st.info("PDF preview (would use pdf2image)")
        
        with col2:
            st.subheader("Extraction Settings")
            
            # Settings
            priority = st.slider("Processing Priority", 1, 10, 5)
            model_variant = st.selectbox(
                "Model Variant",
                ["Auto (Recommended)", "olmocr-v1", "olmocr-v2"]
            )
            
            # Process button
            if st.button("🚀 Extract Data", type="primary"):
                with st.spinner("Uploading document..."):
                    try:
                        # Upload file to /extract endpoint
                        uploaded_file.seek(0)  # Reset file pointer
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        data = {"priority": priority}
                        
                        response = requests.post(
                            f"{API_BASE_URL}/extract",
                            files=files,
                            data=data
                        )
                        response.raise_for_status()
                        
                        upload_result = response.json()
                        job_id = upload_result["job_id"]
                        
                        st.success(f"✅ Document uploaded! Job ID: {job_id}")
                        
                        # Poll for status
                        with st.spinner("Processing document..."):
                            progress_bar = st.progress(0)
                            max_polls = 60  # 60 seconds timeout
                            
                            for poll_count in range(max_polls):
                                status_response = requests.get(
                                    f"{API_BASE_URL}/status/{job_id}"
                                )
                                status_response.raise_for_status()
                                status = status_response.json()
                                
                                if status["status"] == "completed":
                                    progress_bar.progress(1.0)
                                    break
                                elif status["status"] == "failed":
                                    st.error("❌ Processing failed")
                                    break
                                else:
                                    # Update progress
                                    progress = status.get("progress", poll_count / max_polls)
                                    progress_bar.progress(progress)
                                    time.sleep(1)
                            
                            # Fetch results
                            if status["status"] == "completed":
                                results_response = requests.get(
                                    f"{API_BASE_URL}/results/{job_id}"
                                )
                                results_response.raise_for_status()
                                results_data = results_response.json()
                                
                                st.success("✅ Extraction complete!")
                                
                                # Display results
                                st.subheader("Extracted Data")
                                
                                results = results_data["results"]
                                confidence = results_data["confidence"]
                                
                                # Display with confidence indicators
                                st.metric("Overall Confidence", f"{confidence:.0%}")
                                
                                st.write("**Patient Information**")
                                st.write(f"- Patient ID: {results.get('patient_id', 'N/A')}")
                                st.write(f"- Date: {results.get('date', 'N/A')}")
                                
                                if "symptoms" in results:
                                    st.write("**Symptoms**")
                                    for symptom in results['symptoms']:
                                        conf = symptom['confidence']
                                        emoji = "✅" if symptom['checked'] else "❌"
                                        color = "green" if conf > 0.9 else "orange" if conf > 0.7 else "red"
                                        
                                        st.markdown(
                                            f"{emoji} **{symptom['label']}** "
                                            f"<span style='color:{color}'>({conf:.0%} confidence)</span>",
                                            unsafe_allow_html=True
                                        )
                                
                                # Show processing time
                                st.info(f"⏱️ Processing time: {results_data['processing_time_ms']:.0f}ms")
                    
                    except requests.RequestException as e:
                        st.error(f"❌ Error communicating with API: {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")


def review_page():
    """Review and correction page for HITL."""
    st.header("Review Extractions")
    
    st.info("👤 Human-in-the-Loop Review Interface")
    
    # Fetch review queue from API
    try:
        response = requests.get(f"{API_BASE_URL}/review-queue?limit=20")
        response.raise_for_status()
        review_items = response.json()
    except requests.RequestException as e:
        st.error(f"❌ Failed to fetch review queue: {e}")
        review_items = []
    
    # Group by reason
    low_conf_items = [item for item in review_items if item["reason"] == "low_confidence"]
    validation_items = [item for item in review_items if item["reason"] == "validation_error"]
    other_items = [item for item in review_items if item["reason"] not in ["low_confidence", "validation_error"]]
    
    # Tabs for different queues
    tab1, tab2, tab3 = st.tabs([
        f"Low Confidence ({len(low_conf_items)})", 
        f"Validation Errors ({len(validation_items)})", 
        f"Other ({len(other_items)})"
    ])
    
    with tab1:
        st.subheader("Low Confidence Predictions")
        st.markdown("These predictions have confidence < 80% and need review")
        
        if not low_conf_items:
            st.info("✅ No low confidence items at this time")
        else:
            for i, item in enumerate(low_conf_items):
                with st.expander(f"📋 {item['job_id']} - {item['region_type']} ({item['confidence']:.0%})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Region Type:** {item['region_type']}")
                        st.write(f"**Confidence:** {item['confidence']:.0%}")
                        st.write(f"**Priority:** {item['priority']:.2f}")
                        st.write(f"**Predicted Value:**")
                        st.json(item['predicted_value'])
                        
                        corrected = st.text_area(
                            "Correction (JSON format, leave blank if correct)",
                            key=f"correct_{item['job_id']}_{item['region_id']}"
                        )
                        
                        comment = st.text_input(
                            "Optional comment",
                            key=f"comment_{item['job_id']}_{item['region_id']}"
                        )
                        
                        if st.button("✓ Submit Correction", key=f"submit_{item['job_id']}_{item['region_id']}"):
                            try:
                                # Parse corrected value
                                corrected_value = json.loads(corrected) if corrected else item['predicted_value']
                                
                                # Submit to /review/correct
                                correction_data = {
                                    "job_id": item['job_id'],
                                    "region_id": item['region_id'],
                                    "corrected_value": corrected_value,
                                    "original_value": item['predicted_value'],
                                    "comment": comment
                                }
                                
                                response = requests.post(
                                    f"{API_BASE_URL}/review/correct",
                                    json=correction_data
                                )
                                response.raise_for_status()
                                
                                st.success("✅ Correction recorded!")
                            except json.JSONDecodeError:
                                st.error("❌ Invalid JSON format")
                            except requests.RequestException as e:
                                st.error(f"❌ Failed to submit: {e}")
                    
                    with col2:
                        if item.get('image_url'):
                            st.image(item['image_url'], caption="Region")
                        else:
                            st.info("No image available")
    
    with tab2:
        st.subheader("Validation Errors")
        
        if not validation_items:
            st.info("✅ No validation errors at this time")
        else:
            for item in validation_items:
                with st.expander(f"⚠️ {item['job_id']} - {item['region_type']}"):
                    st.write(f"**Reason:** {item['reason']}")
                    st.write(f"**Predicted Value:**")
                    st.json(item['predicted_value'])
    
    with tab3:
        st.subheader("Other Review Items")
        
        if not other_items:
            st.info("✅ No other items at this time")
        else:
            for item in other_items:
                with st.expander(f"📋 {item['job_id']} - {item['reason']}"):
                    st.json(item['predicted_value'])


def analytics_page():
    """Analytics and monitoring dashboard."""
    st.header("📊 Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Processed", "1,247", "+12%")
    
    with col2:
        st.metric("Avg Confidence", "0.89", "+0.02")
    
    with col3:
        st.metric("Human Review Rate", "15%", "-3%")
    
    with col4:
        st.metric("Processing Time", "2.3s", "-0.4s")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confidence Distribution")
        
        # Mock data
        confidence_data = pd.DataFrame({
            'Confidence Range': ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'],
            'Count': [15, 42, 98, 312, 780]
        })
        
        st.bar_chart(confidence_data.set_index('Confidence Range'))
    
    with col2:
        st.subheader("Region Type Performance")
        
        region_data = pd.DataFrame({
            'Region Type': ['Checkbox', 'Table', 'Text', 'Signature'],
            'Accuracy': [0.92, 0.85, 0.88, 0.76]
        })
        
        st.bar_chart(region_data.set_index('Region Type'))
    
    st.markdown("---")
    
    # Detailed table
    st.subheader("Recent Processing History")
    
    history_data = pd.DataFrame({
        'Document ID': ['doc_001', 'doc_002', 'doc_003', 'doc_004', 'doc_005'],
        'Timestamp': ['2025-10-30 10:23', '2025-10-30 10:21', '2025-10-30 10:18', 
                     '2025-10-30 10:15', '2025-10-30 10:12'],
        'Status': ['✅ Completed', '✅ Completed', '⚠️ Review', '✅ Completed', '✅ Completed'],
        'Confidence': ['0.92', '0.88', '0.65', '0.94', '0.89'],
        'Processing Time': ['2.1s', '2.3s', '2.8s', '2.0s', '2.4s']
    })
    
    st.dataframe(history_data, use_container_width=True)


def settings_page():
    """Configuration and settings page."""
    st.header("⚙️ Settings")
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Primary Model", ["olmocr-v1", "olmocr-v2"])
        st.slider("Confidence Threshold", 0.0, 1.0, 0.8)
        st.number_input("Self-Consistency Samples (k)", 1, 5, 3)
    
    with col2:
        st.selectbox("Routing Strategy", ["Weighted", "Epsilon-Greedy", "Sticky"])
        st.checkbox("Enable A/B Testing", value=True)
        st.checkbox("Shadow Mode", value=False)
    
    st.markdown("---")
    
    st.subheader("Active Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Uncertainty Weight", 0.0, 1.0, 0.5)
        st.slider("Diversity Weight", 0.0, 1.0, 0.3)
    
    with col2:
        st.slider("Value Weight", 0.0, 1.0, 0.2)
        st.number_input("Annotation Batch Size", 1, 50, 10)
    
    st.markdown("---")
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")


if __name__ == "__main__":
    main()

