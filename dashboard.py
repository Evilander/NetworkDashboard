# --- START OF FILE dashboard.py ---

import streamlit as st
import time
import pandas as pd
import random
from datetime import datetime, timedelta
import logging
import base64
import os
import gc

# Local application imports
from packet_processor import PacketProcessor
from llm_analyzer import LLMAnalyzer
from dashboard_views import (
    display_dashboard_view,
    display_network_graph_view,
    display_top_talkers_view,
    display_trend_analysis_view,
    display_security_insights_view,
    reset_cache
    # Note: visualizations module functions are typically called *within*
    # the dashboard_views functions, so direct import here might not be needed.
)
from config import (
    logger, DB_PATH, AUTO_REFRESH_DEFAULT,
    REFRESH_INTERVAL_DEFAULT, DEFAULT_TIMEFRAME,
    MEMORY_WARNING_THRESHOLD, MEMORY_CRITICAL_THRESHOLD,
    MEMORY_CHECK_INTERVAL
)

# Import the memory manager with graceful fallback if unavailable
try:
    from memory_manager import MemoryManager
except ImportError:
    logger.warning("memory_manager.py not found. Memory monitoring disabled.")
    # Create a dummy MemoryManager class to avoid errors if the module is not available
    class MemoryManager:
        def __init__(self, *args, **kwargs): pass
        def start_monitoring(self): pass
        def stop_monitoring(self): pass
        def manual_cleanup(self): return 0
        def get_memory_usage(self): return 0
        warning_threshold = 70 # Default values for display logic
        critical_threshold = 85

# Set up page configuration
st.set_page_config(
    page_title="Network Analysis Dashboard", # Generalized title
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (remains generic)
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    .stAlert {border-radius: 5px;}
    h1, h2, h3 {padding-top: 1rem; padding-bottom: 0.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 5px 5px 0px 0px;
    }
    .highlight {
        background-color: #f0f7fb;
        padding: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 10px;
    }
    .danger-zone {
        background-color: #fff8f8;
        padding: 15px;
        border-left: 5px solid #e74c3c;
        margin-top: 20px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .stButton button {
        width: 100%;
    }
    .memory-usage-bar {
        margin-top: 10px;
        margin-bottom: 10px;
        width: 100%;
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
    }
    .memory-usage-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize or update session state variables if not already present."""
    if 'initialized' not in st.session_state:
        logger.info("Initializing session state...")
        st.session_state.processor = None
        st.session_state.analyzer = None
        st.session_state.memory_manager = None
        st.session_state.start_time = time.time()
        st.session_state.last_refresh = time.time()
        st.session_state.auto_refresh = AUTO_REFRESH_DEFAULT
        st.session_state.refresh_interval = REFRESH_INTERVAL_DEFAULT
        st.session_state.selected_timeframe = DEFAULT_TIMEFRAME
        st.session_state.selected_protocols = []
        st.session_state.selected_view = "Dashboard"
        st.session_state.show_reset_confirm = False
        # Initialize a random cache key for dashboard_views caching functions
        st.session_state.cache_key = f"cache-{random.randint(1, 1000000)}-{time.time()}"

        # --- Initialize Core Components ---
        try:
            st.session_state.processor = PacketProcessor(db_path=DB_PATH)
            logger.info("Packet Processor initialized.")
            # Attempt to start packet capture - Requires elevated privileges
            st.session_state.processor.start_capture()
            logger.info("Packet capture thread started.")
        except ImportError as e:
             logger.error(f"Failed to import required modules for PacketProcessor: {e}")
             st.error(f"Initialization Error: Missing required libraries ({e}). Please check requirements.txt.")
             st.stop() # Halt execution if essential parts fail
        except PermissionError as e:
            logger.error(f"Permission denied starting packet capture: {e}. Need root/admin privileges.")
            st.error("Permission Error: Failed to start packet capture. Run Streamlit with sudo or as Administrator.")
            # Continue initialization but capture won't work
        except Exception as e:
            logger.error(f"Failed to initialize or start Packet Processor: {e}", exc_info=True)
            st.error(f"Initialization Error: Could not start packet capture. Error: {e}")
            # Decide if we should stop or continue with limited functionality
            # For now, let's continue, but metrics will be empty.

        try:
            st.session_state.analyzer = LLMAnalyzer(from_config=True)
            logger.info("LLM Analyzer initialized.")
            if not st.session_state.analyzer.api_key:
                 logger.warning("LLM API Key not found in config/env. LLM features will be limited.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Analyzer: {e}", exc_info=True)
            st.warning("Could not initialize LLM Analyzer. LLM features disabled.")

        try:
            st.session_state.memory_manager = MemoryManager(
                warning_threshold=MEMORY_WARNING_THRESHOLD,
                critical_threshold=MEMORY_CRITICAL_THRESHOLD,
                check_interval=MEMORY_CHECK_INTERVAL
            )
            st.session_state.memory_manager.start_monitoring()
            logger.info("Memory Manager initialized and monitoring started.")
        except Exception as e:
            logger.error(f"Failed to initialize Memory Manager: {e}", exc_info=True)
            st.warning("Could not initialize Memory Manager. Memory monitoring disabled.")

        st.session_state.initialized = True # Mark initialization as complete
        logger.info("Session state initialization complete.")


def display_memory_usage():
    """Display memory usage information in the sidebar."""
    if 'memory_manager' in st.session_state and st.session_state.memory_manager:
        try:
            # Get memory usage
            memory_usage = st.session_state.memory_manager.get_memory_usage()
            warning_threshold = st.session_state.memory_manager.warning_threshold
            critical_threshold = st.session_state.memory_manager.critical_threshold

            # Determine color based on usage
            if memory_usage > critical_threshold:
                color = "#e74c3c"  # Red
            elif memory_usage > warning_threshold:
                color = "#f39c12"  # Orange
            else:
                color = "#2ecc71"  # Green

            # Display memory usage bar
            st.markdown(f"### RAM Usage: {memory_usage:.1f}%")
            st.markdown(
                f"""
                <div class="memory-usage-bar">
                    <div class="memory-usage-fill" style="width: {min(memory_usage, 100)}%; background-color: {color};"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Add cleanup button if memory usage is moderately high
            if memory_usage > 50:
                if st.button("🧹 Clean Memory Now", key="clean_memory_button", use_container_width=True):
                    with st.spinner("Cleaning up memory..."):
                        # Run garbage collection
                        gc.collect()
                        # Clear Streamlit's data cache
                        reset_cache()
                        # Run memory manager cleanup
                        new_usage = st.session_state.memory_manager.manual_cleanup()
                        st.success(f"Memory cleaned! Usage now: {new_usage:.1f}%")
                        time.sleep(1)
                        st.rerun()

        except Exception as e:
            st.error(f"Error displaying memory usage: {str(e)}")
            logger.error(f"Error in display_memory_usage: {e}", exc_info=True)
    else:
        st.info("Memory Manager not available.")


def main() -> None:
    """Main function to run the Streamlit dashboard application."""

    initialize_session_state() # Ensure state is initialized

    # --- Dashboard Title and Optional Logo ---
    st.markdown("<h1 style='text-align: center;'> 🌐 Network Analysis Dashboard 🌐</h1>", unsafe_allow_html=True)

    # Optional: Add a generic logo (e.g., place 'logo.png' in an 'assets' folder)
    logo_path = 'assets/logo.png'
    if os.path.exists(logo_path):
        try:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                    <img src="data:image/png;base64,{base64.b64encode(open(logo_path, 'rb').read()).decode()}" alt="Logo" style="width: 100px; height: auto;">
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            logger.error(f"Error displaying logo image: {e}")
    # else:
    #     logger.info(f"Logo file not found at {logo_path}, skipping logo display.")

    # Check if essential components initialized correctly
    if not st.session_state.processor:
         st.error("Packet Processor failed to initialize. Dashboard functionality will be limited. Please check logs and permissions.")
         # Optionally st.stop() here if the dashboard is unusable without the processor
         # return

    # --- Tab Navigation ---
    view_tabs = [
        "Dashboard",
        "Network Graph",
        "Top Talkers",
        "Trend Analysis",
        "Security Insights",
    ]

    # Ensure tab index persists across reruns and sync with selected_view
    if "main_tabs" not in st.session_state:
        st.session_state.main_tabs = (
            view_tabs.index(st.session_state.selected_view)
            if st.session_state.selected_view in view_tabs
            else 0
        )

    tabs = st.tabs(view_tabs, key="main_tabs")
    st.session_state.selected_view = view_tabs[st.session_state.main_tabs]

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings & Filters")

        # Display memory usage
        display_memory_usage()

        # Timeframe selection
        st.subheader("Timeframe")
        timeframe_options = ["hour", "day", "week", "month"]
        current_timeframe_index = timeframe_options.index(st.session_state.selected_timeframe) if st.session_state.selected_timeframe in timeframe_options else 1 # Default to 'day' if invalid
        timeframe = st.radio(
            "Select timeframe for analysis:",
            options=timeframe_options,
            index=current_timeframe_index,
            key="timeframe_radio"
        )
        if timeframe != st.session_state.selected_timeframe:
            st.session_state.selected_timeframe = timeframe
            reset_cache() # Clear cache when timeframe changes
            st.rerun() # Rerun to apply change immediately

        # Protocol filter
        st.subheader("Protocol Filter")
        available_protocols = []
        if st.session_state.processor:
             # Fetch a small sample for protocol list without hitting cache issues
             df_proto = st.session_state.processor.db.get_recent_packets(limit=500)
             if not df_proto.empty and 'protocol' in df_proto.columns:
                 try:
                     available_protocols = sorted(df_proto['protocol'].unique().tolist())
                 except Exception as e:
                     logger.error(f"Error getting unique protocols from DB sample: {e}")
                     available_protocols = ["TCP", "UDP", "ICMP", "DNS", "HTTP", "HTTPS"] # Provide common fallbacks

        if not available_protocols:
            st.info("No packet data yet to populate protocol filter.")

        selected_protocols = st.multiselect(
            "Filter by protocols (optional):",
            options=available_protocols,
            default=st.session_state.selected_protocols,
            key="protocol_multiselect"
        )

        if selected_protocols != st.session_state.selected_protocols:
            st.session_state.selected_protocols = selected_protocols
            reset_cache() # Clear cache when filter changes
            # Apply filter (assuming set_filter/clear_filters exist on processor)
            if st.session_state.processor:
                if selected_protocols:
                    filter_set = set(selected_protocols)
                    st.session_state.processor.set_filter('protocol', filter_set)
                else:
                    st.session_state.processor.clear_filters()
            st.rerun() # Rerun to apply filter

        # Refresh settings
        st.subheader("Refresh Settings")
        auto_refresh = st.checkbox("Auto refresh", value=st.session_state.auto_refresh, key="auto_refresh_checkbox")
        st.session_state.auto_refresh = auto_refresh

        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", min_value=2, max_value=60, value=st.session_state.refresh_interval, key="refresh_interval_slider")
            st.session_state.refresh_interval = refresh_interval

        # Manual refresh button
        if st.button('🔄 Refresh Data Now', key="manual_refresh_button", use_container_width=True):
            reset_cache() # Clear cache on manual refresh
            st.session_state.last_refresh = time.time()
            st.rerun()

        # View selection handled by tabs in the main area

        # Database info
        st.subheader("Database Info")
        if st.session_state.processor:
            db_size = st.session_state.processor.db.get_database_size()
            total_packets = st.session_state.processor.get_total_packet_count()
            st.metric("Total Packets Stored", f"{total_packets:,}")
            st.metric("Database Size", f"{db_size:.2f} MB")
        else:
            st.metric("Total Packets Stored", "N/A")
            st.metric("Database Size", "N/A")


        # --- Data Management Section ---
        st.markdown('<div class="danger-zone"><h3>Data Management</h3>', unsafe_allow_html=True)

        # Optimize Database
        if st.button("Optimize Database", key="optimize_db_button", use_container_width=True, help="Reclaims unused space and optimizes DB structure."):
             if st.session_state.processor:
                 with st.spinner("Optimizing database... This might take a moment."):
                     try:
                        st.session_state.processor.db.optimize_database()
                        # Clean memory after potential large operation
                        if 'memory_manager' in st.session_state and st.session_state.memory_manager:
                            st.session_state.memory_manager.manual_cleanup()
                        st.success("Database optimized successfully!")
                        time.sleep(1) # Pause for user to see success
                     except Exception as e:
                         logger.error(f"Error optimizing database: {e}", exc_info=True)
                         st.error(f"Database optimization failed: {e}")
             else:
                 st.warning("Packet processor not available.")

        # Cleanup Old Data
        cleanup_col1, cleanup_col2 = st.columns([3, 1])
        with cleanup_col1:
            # Ensure value is reasonable, default to config or 30
            default_days = getattr(st.session_state.processor.db, 'AUTO_CLEANUP_DAYS', 30) if st.session_state.processor else 30
            days_to_keep = st.number_input("Days of data to keep:", min_value=1, max_value=365, value=default_days, key="cleanup_days_input")
        with cleanup_col2:
            st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True) # Align button vertically
            if st.button("Cleanup", key="cleanup_button", use_container_width=True, help=f"Remove data older than {days_to_keep} days."):
                if st.session_state.processor:
                    with st.spinner(f"Cleaning up data older than {days_to_keep} days..."):
                        try:
                            st.session_state.processor.db.cleanup_old_data(days_to_keep=days_to_keep)
                            reset_cache() # Data changed, clear cache
                            if 'memory_manager' in st.session_state and st.session_state.memory_manager:
                                st.session_state.memory_manager.manual_cleanup()
                            st.success(f"Cleaned up data older than {days_to_keep} days.")
                            st.session_state.last_refresh = time.time()
                            time.sleep(1)
                            st.rerun() # Refresh view after cleanup
                        except Exception as e:
                            logger.error(f"Error cleaning up old data: {e}", exc_info=True)
                            st.error(f"Data cleanup failed: {e}")
                else:
                    st.warning("Packet processor not available.")

        st.markdown("<hr style='margin-top:1rem; margin-bottom:1rem;'>", unsafe_allow_html=True)

        # Reset Dashboard (Clear All Data)
        if st.button("⚠️ Reset Dashboard", key="reset_dashboard_button", use_container_width=True, help="Deletes ALL captured data and resets the dashboard."):
            st.session_state.show_reset_confirm = True

        if st.session_state.get('show_reset_confirm', False):
            st.warning("🚨 **Confirm Reset** 🚨\n\nThis action will permanently delete **ALL** captured network data from the database. This cannot be undone.")
            confirm_col1, confirm_col2 = st.columns(2)
            with confirm_col1:
                if st.button("✓ Yes, Delete Everything", key="confirm_reset_button", use_container_width=True):
                    if st.session_state.processor:
                        with st.spinner("Resetting dashboard and clearing all data..."):
                            try:
                                success = st.session_state.processor.db.clear_all_data()
                                if success:
                                    st.session_state.show_reset_confirm = False
                                    st.session_state.start_time = time.time() # Reset session start time
                                    reset_cache() # Clear any potentially stale cached data
                                    if 'memory_manager' in st.session_state and st.session_state.memory_manager:
                                        st.session_state.memory_manager.manual_cleanup() # Clean memory
                                    st.success("All data cleared. Dashboard reset!")
                                    st.session_state.last_refresh = time.time()
                                    time.sleep(1.5)
                                    st.rerun() # Force a full refresh
                                else:
                                    st.error("Error clearing data. Please check logs.")
                            except Exception as e:
                                logger.error(f"Error during dashboard reset: {e}", exc_info=True)
                                st.error(f"Dashboard reset failed: {e}")
                    else:
                         st.warning("Packet processor not available.")
                         st.session_state.show_reset_confirm = False # Hide confirmation if cannot proceed

            with confirm_col2:
                if st.button("✕ Cancel", key="cancel_reset_button", use_container_width=True):
                    st.session_state.show_reset_confirm = False
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True) # End danger-zone div


    # --- Main Content Area ---
    selected_index = st.session_state.main_tabs

    with tabs[0]:
        if selected_index == 0:
            display_dashboard_view()

    with tabs[1]:
        if selected_index == 1:
            display_network_graph_view()

    with tabs[2]:
        if selected_index == 2:
            display_top_talkers_view()

    with tabs[3]:
        if selected_index == 3:
            display_trend_analysis_view()

    with tabs[4]:
        if selected_index == 4:
            display_security_insights_view()

    # --- Auto Refresh Logic ---
    if st.session_state.auto_refresh:
        current_time = time.time()
        # Check if last_refresh exists (might not if init failed)
        if 'last_refresh' in st.session_state and (current_time - st.session_state.last_refresh > st.session_state.refresh_interval):
            logger.debug(f"Auto-refresh triggered after {st.session_state.refresh_interval}s.")
            st.session_state.last_refresh = current_time
            reset_cache() # Reset cache on auto refresh

            # Periodically run garbage collection during auto-refresh
            if 'start_time' in st.session_state:
                 uptime = current_time - st.session_state.start_time
                 # Run GC approx every 15 mins after 1 hour, aligned with refresh interval
                 if uptime > 3600 and int(uptime) % 900 < st.session_state.refresh_interval:
                     logger.info("Running periodic garbage collection...")
                     collected = gc.collect()
                     logger.info(f"Garbage collector collected {collected} objects.")

            time.sleep(0.1) # Brief pause before triggering rerun
            st.rerun()

if __name__ == "__main__":
    # No privilege check here; it's handled at runtime by scapy/PacketProcessor
    # Running this script directly `python dashboard.py` is not the intended way.
    # Use `streamlit run dashboard.py` (potentially with sudo/admin privileges).
    logger.info("Starting Network Analysis Dashboard application.")
    main()
    logger.info("Network Analysis Dashboard application finished.")

# --- END OF FILE dashboard.py ---