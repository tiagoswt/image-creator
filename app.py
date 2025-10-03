"""
Automated Product Image Processing Tool - Streamlit Web Interface
MVP for batch processing e-commerce product images
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

import streamlit as st
import os
import time
import zipfile
from pathlib import Path
from PIL import Image
import io
from image_processor import ImageProcessor

# Page configuration
st.set_page_config(
    page_title="Product Image Processor",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
TEMP_DIR = "temp"

# Ensure directories exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)


def clear_directory(directory: str):
    """Clear all files in a directory"""
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error deleting {file_path}: {e}")


def create_download_zip(processed_files: list) -> bytes:
    """Create a ZIP file containing all processed images"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in processed_files:
            if os.path.exists(file_path):
                zip_file.write(file_path, os.path.basename(file_path))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def main():
    # Header
    st.title("🖼️ Automated Product Image Processor")
    st.markdown("""
    **Transform your product images in seconds!**

    This tool automatically:
    - ✨ Removes backgrounds using AI
    - 🌟 Eliminates shadows
    - ✂️ Smart crops to product boundaries
    - 📐 Resizes to 1000x1000px with white background
    - 🎨 Enhances image quality
    """)

    st.divider()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        target_size = st.number_input(
            "Target Size (px)",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="Output image dimensions (square)"
        )

        st.divider()

        st.subheader("🎨 Background Removal")

        bg_model = st.selectbox(
            "AI Model",
            options=[
                'u2net',
                'isnet-general-use',
                'u2netp',
                'u2net_human_seg',
                'silueta',
                'rmbg-1.4'
            ],
            index=5,
            help="u2net: Balanced & Recommended | isnet-general-use: Best quality | u2netp: Fastest | rmbg-1.4: BRIA AI model (CPU-friendly)"
        )

        # RMBG-1.4 info
        hf_api_token = None
        if bg_model == 'rmbg-1.4':
            st.info("💡 RMBG-1.4 uses transformers pipeline (CPU-friendly, first run downloads model)")

        alpha_matting = st.checkbox(
            "Alpha Matting",
            value=True,
            help="Refine edges for smoother transitions (slower but better quality)"
        )

        post_process = st.checkbox(
            "Post-process Mask",
            value=True,
            help="Remove small artifacts from the mask"
        )

        st.divider()

        st.subheader("🔧 Processing Options")

        enable_shadow_removal = st.checkbox(
            "Enable Shadow Removal",
            value=False,
            help="Detect and remove shadows from products"
        )

        enable_smart_crop = st.checkbox(
            "Enable Smart Cropping",
            value=True,
            help="Use advanced contour detection for precise cropping"
        )

        save_intermediate = st.checkbox(
            "Save Intermediate Steps",
            value=False,
            help="Save images after each processing step for inspection"
        )

        st.divider()

        st.subheader("✂️ Cropping Settings")

        padding_percent = st.slider(
            "Padding %",
            min_value=0.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
            help="Amount of space around the product (higher = more space)"
        )

        crop_tightness = st.selectbox(
            "Crop Tightness",
            options=['tight', 'normal', 'loose'],
            index=0,
            help="How closely to crop to product boundaries"
        )

        st.divider()

        st.header("📊 Processing Stats")
        if 'stats' in st.session_state:
            stats = st.session_state.stats
            st.metric("Images Processed", stats.get('total', 0))
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
            st.metric("Avg Processing Time", f"{stats.get('avg_time', 0):.2f}s")

        st.divider()

        if st.button("🗑️ Clear All Files", type="secondary"):
            clear_directory(UPLOAD_DIR)
            clear_directory(PROCESSED_DIR)
            st.session_state.clear()
            st.success("All files cleared!")
            st.rerun()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Images")

        uploaded_files = st.file_uploader(
            "Choose product images",
            type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
            accept_multiple_files=True,
            help="Upload one or multiple images to process"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} image(s) uploaded")

            # Show thumbnails
            with st.expander("📸 Preview Uploaded Images"):
                cols = st.columns(3)
                for idx, uploaded_file in enumerate(uploaded_files):
                    with cols[idx % 3]:
                        img = Image.open(uploaded_file)
                        st.image(img, caption=uploaded_file.name, width='stretch')

    with col2:
        st.header("⚡ Process Images")

        if uploaded_files:
            if st.button("🚀 Start Processing", type="primary", width='stretch'):
                # Cache processor in session state to avoid reinitialization
                processor_key = f"{bg_model}_{alpha_matting}_{post_process}_{hf_api_token}"

                if 'processor' not in st.session_state or st.session_state.get('processor_key') != processor_key:
                    with st.spinner("🔧 Loading AI model... (first run may take a minute)"):
                        processor = ImageProcessor(
                            target_size=target_size,
                            padding_percent=padding_percent,
                            crop_tightness=crop_tightness,
                            enable_shadow_removal=enable_shadow_removal,
                            enable_smart_crop=enable_smart_crop,
                            save_intermediate=save_intermediate,
                            bg_model=bg_model,
                            alpha_matting=alpha_matting,
                            post_process=post_process,
                            hf_api_token=hf_api_token
                        )
                        st.session_state.processor = processor
                        st.session_state.processor_key = processor_key
                else:
                    processor = st.session_state.processor
                    # Update dynamic settings
                    processor.target_size = target_size
                    processor.padding_percent = padding_percent / 100.0
                    processor.crop_tightness = crop_tightness
                    processor.enable_shadow_removal = enable_shadow_removal
                    processor.enable_smart_crop = enable_smart_crop
                    processor.save_intermediate = save_intermediate

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()

                # Save uploaded files
                uploaded_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    uploaded_paths.append(file_path)

                # Process images
                results = []
                processed_files = []
                start_time = time.time()

                for idx, file_path in enumerate(uploaded_paths):
                    filename = os.path.basename(file_path)

                    # Use status container to show detailed progress
                    with status_text.container():
                        st.write(f"**Processing {idx + 1}/{len(uploaded_paths)}:** {filename}")

                        with st.spinner("🔄 Removing background..."):
                            output_filename = f"{os.path.splitext(filename)[0]}_processed.png"
                            output_path = os.path.join(PROCESSED_DIR, output_filename)

                            # Process image
                            result = processor.process_image(file_path, output_path)
                            result['filename'] = filename
                            results.append(result)

                            if result['status'] == 'success':
                                processed_files.append(output_path)

                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_paths))

                total_time = time.time() - start_time

                # Display results
                status_text.empty()
                progress_bar.empty()

                success_count = sum(1 for r in results if r['status'] == 'success')
                error_count = len(results) - success_count

                with results_container:
                    st.success(f"✅ Processing Complete!")

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Successful", success_count)
                    col_b.metric("Failed", error_count)
                    col_c.metric("Total Time", f"{total_time:.2f}s")

                    # Store stats
                    st.session_state.stats = {
                        'total': len(results),
                        'success_rate': (success_count / len(results)) * 100 if results else 0,
                        'avg_time': total_time / len(results) if results else 0
                    }

                    # Show detailed results
                    with st.expander("📋 Detailed Results"):
                        for result in results:
                            if result['status'] == 'success':
                                st.success(f"✅ {result['filename']}")
                            else:
                                st.error(f"❌ {result['filename']}: {result.get('error', 'Unknown error')}")

                # Store processed files in session state
                st.session_state.processed_files = processed_files
                st.session_state.results = results

        else:
            st.info("👆 Upload images to begin processing")

    # Download section
    if 'processed_files' in st.session_state and st.session_state.processed_files:
        st.divider()
        st.header("📥 Download Processed Images")

        col_dl1, col_dl2 = st.columns([2, 1])

        with col_dl1:
            # Download all as ZIP
            if len(st.session_state.processed_files) > 1:
                zip_data = create_download_zip(st.session_state.processed_files)
                st.download_button(
                    label="📦 Download All as ZIP",
                    data=zip_data,
                    file_name="processed_images.zip",
                    mime="application/zip",
                    width='stretch'
                )

            # Download individual files
            if st.checkbox("📁 Show Individual Downloads"):
                for file_path in st.session_state.processed_files:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label=f"⬇️ {os.path.basename(file_path)}",
                                data=f.read(),
                                file_name=os.path.basename(file_path),
                                mime="image/png",
                                key=file_path
                            )

        with col_dl2:
            st.metric("Files Ready", len(st.session_state.processed_files))

        # Preview processed images
        st.subheader("🎨 Before & After Comparison")

        if st.session_state.results:
            for result in st.session_state.results:
                if result['status'] == 'success':
                    with st.expander(f"🔍 {result['filename']}"):
                        # Show intermediate steps if available
                        if 'intermediate_paths' in result and result['intermediate_paths']:
                            st.markdown("### 📊 Processing Steps")

                            # Define step labels
                            step_labels = {
                                'background_removed': '1️⃣ Background Removed',
                                'shadow_removed': '2️⃣ Shadow Removed',
                                'cropped': '3️⃣ Smart Cropped',
                                'resized': '4️⃣ Resized & Centered'
                            }

                            # Display intermediate steps in grid
                            intermediate = result['intermediate_paths']
                            num_steps = len(intermediate)
                            cols_per_row = 2

                            for i, (step_key, step_path) in enumerate(intermediate.items()):
                                if i % cols_per_row == 0:
                                    cols = st.columns(cols_per_row)

                                with cols[i % cols_per_row]:
                                    st.markdown(f"**{step_labels.get(step_key, step_key)}**")
                                    if os.path.exists(step_path):
                                        st.image(step_path, use_container_width=True)

                        st.markdown("### 🔄 Before & After")
                        col_before, col_after = st.columns(2)

                        with col_before:
                            st.markdown("**Before**")
                            original_path = os.path.join(UPLOAD_DIR, result['filename'])
                            if os.path.exists(original_path):
                                st.image(original_path, width='stretch')

                        with col_after:
                            st.markdown("**After**")
                            if os.path.exists(result['output_path']):
                                st.image(result['output_path'], width='stretch')
                                with open(result['output_path'], 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Download Image",
                                        data=f.read(),
                                        file_name=os.path.basename(result['output_path']),
                                        mime="image/png",
                                        key=f"download_{result['filename']}",
                                        width='stretch'
                                    )

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Automated Product Image Processing Tool - MVP v1.0</p>
        <p>Built for beauty, skincare, haircare, and healthcare e-commerce</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

