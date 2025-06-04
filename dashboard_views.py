import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add a cache breaker for consistent dashboard refresh
def get_cache_key():
    """Return a unique cache key that changes on data reset"""
    return st.session_state.get('cache_key', 'initial')

@st.cache_data(ttl=15, show_spinner=False)
def cached_get_recent_packets(limit: int, cache_key: str = None):
    """Get recent packets with cache control"""
    return st.session_state.processor.db.get_recent_packets(limit=limit)

@st.cache_data(ttl=15, show_spinner=False)
def cached_get_packets_by_timeframe(hours: float, protocol: Optional[str], cache_key: str = None):
    """Get packets by timeframe with cache control"""
    return st.session_state.processor.db.get_packets_by_timeframe(hours=hours, protocol=protocol)

@st.cache_data(ttl=30, show_spinner=False)
def cached_get_top_talkers(timeframe: str, limit: int, cache_key: str = None):
    """Get top talkers with cache control"""
    return st.session_state.processor.db.get_top_talkers(timeframe=timeframe, limit=limit)

@st.cache_data(ttl=30, show_spinner=False)
def cached_get_protocol_distribution(timeframe: str, cache_key: str = None):
    """Get protocol distribution with cache control"""
    return st.session_state.processor.db.get_protocol_distribution(timeframe=timeframe)

@st.cache_data(ttl=30, show_spinner=False)
def cached_get_packet_timeline(interval: str, timeframe: str, cache_key: str = None):
    """Get packet timeline with cache control"""
    return st.session_state.processor.db.get_packet_timeline(interval=interval, timeframe=timeframe)

def reset_cache():
    """Clear all cached data"""
    # Generate new random cache key
    st.session_state['cache_key'] = f"cache-{random.randint(1, 1000000)}-{time.time()}"
    # Clear all cached functions
    cached_get_recent_packets.clear()
    cached_get_packets_by_timeframe.clear()
    cached_get_top_talkers.clear()
    cached_get_protocol_distribution.clear() 
    cached_get_packet_timeline.clear()
    logger.info("Cache cleared with new key: " + st.session_state['cache_key'])

def display_dashboard_view() -> None:
    """Display the main dashboard view"""
    # Create a header for the current section
    st.header("Dashboard Overview")
    
    cache_key = get_cache_key()
    
    # Get current data
    df = cached_get_recent_packets(limit=1000, cache_key=cache_key)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_packets = st.session_state.processor.get_total_packets()
        st.metric("Total Packets", f"{total_packets:,}", help="Total number of packets captured and stored in the database.")
    
    with col2:
        duration = time.time() - st.session_state.start_time
        days, remainder = divmod(duration, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            duration_str = f"{int(days)}d {int(hours)}h {int(minutes)}m"
        elif hours > 0:
            duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        else:
            duration_str = f"{int(minutes)}m {int(seconds)}s"
        
        st.metric("Capture Duration", duration_str, help="Total duration since packet capturing started.")
    
    with col3:
        if not df.empty and 'size' in df.columns:
            avg_size = df['size'].mean()
            st.metric("Avg Packet Size", f"{avg_size:.1f} bytes", help="Average size of captured packets in bytes.")
        else:
            st.metric("Avg Packet Size", "N/A")
    
    with col4:
        if not df.empty and 'protocol' in df.columns:
            protocols = df['protocol'].nunique()
            st.metric("Unique Protocols", protocols, help="Number of unique network protocols detected.")
        else:
            st.metric("Unique Protocols", "N/A")
    
    # Check if we have any data
    if df.empty:
        st.warning("No packet data available yet. The dashboard will automatically update when packets are captured.")
        st.info("If you're not seeing any data after a while, check your network interfaces and ensure the packet capture is working correctly.")
        return
    
    # Display visualizations
    from visualizations import create_visualization
    create_visualization(
        st.session_state.processor,
        timeframe=st.session_state.selected_timeframe,
        protocol_filter=st.session_state.selected_protocols[0] if st.session_state.selected_protocols else None
    )
    
    # Display recent packets
    st.subheader("Recent Packets")
    if not df.empty:
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
        st.dataframe(
            df.tail(10)[['timestamp', 'source', 'destination', 'protocol', 'size']].sort_values('timestamp', ascending=False),
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time",
                    format="HH:mm:ss.SSS",
                    help="Packet timestamp"
                ),
                "source": "Source IP",
                "destination": "Destination IP",
                "protocol": "Protocol",
                "size": st.column_config.NumberColumn(
                    "Size (bytes)",
                    format="%d",
                    help="Packet size in bytes"
                )
            }
        )
    else:
        st.info("No packet data available. Waiting for network traffic...")

def display_network_graph_view() -> None:
    """Display the network connection graph view"""
    st.header("Network Connection Graph")
    st.markdown("This graph shows connections between top source and destination IPs in your network.")
    
    # Get cache key for consistent caching
    cache_key = get_cache_key()
    
    # Create the network graph
    from visualizations import create_network_graph
    create_network_graph(st.session_state.processor)

def display_top_talkers_view() -> None:
    """Display the top talkers analysis view"""
    st.header("Top Talkers Analysis")
    
    # Get cache key for consistent caching
    cache_key = get_cache_key()
    
    # Timeframe tabs for top talkers
    timeframe_tabs = st.tabs(["Day", "Week", "Month", "Year"])
    
    with timeframe_tabs[0]:
        top_talkers = cached_get_top_talkers(timeframe='day', limit=10, cache_key=cache_key)
        display_top_talkers_data("Daily", top_talkers)
    
    with timeframe_tabs[1]:
        top_talkers = cached_get_top_talkers(timeframe='week', limit=10, cache_key=cache_key)
        display_top_talkers_data("Weekly", top_talkers)
    
    with timeframe_tabs[2]:
        top_talkers = cached_get_top_talkers(timeframe='month', limit=10, cache_key=cache_key)
        display_top_talkers_data("Monthly", top_talkers)
    
    with timeframe_tabs[3]:
        top_talkers = cached_get_top_talkers(timeframe='year', limit=10, cache_key=cache_key)
        display_top_talkers_data("Yearly", top_talkers)

def display_top_talkers_data(period: str, top_talkers: dict) -> None:
    """Display top talkers data for a specific timeframe"""
    if top_talkers['sources'].empty and top_talkers['destinations'].empty:
        st.info(f"No {period.lower()} data available yet.")
        return
    
    st.subheader(f"{period} Top Talkers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Source IPs")
        if not top_talkers['sources'].empty:
            # Format data for display
            display_df = top_talkers['sources'].copy()
            display_df['total_kb'] = display_df['total_bytes'] / 1024
            
            # Create bar chart
            import plotly.express as px
            fig = px.bar(
                display_df,
                x='source',
                y='packet_count',
                color='total_kb',
                labels={
                    'source': 'Source IP',
                    'packet_count': 'Packet Count',
                    'total_kb': 'Total KB'
                },
                title=f"{period} Top Source IPs"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display data table
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "source": "Source IP",
                    "packet_count": st.column_config.NumberColumn("Packets"),
                    "total_kb": st.column_config.NumberColumn(
                        "KB Transferred",
                        format="%.2f KB"
                    )
                }
            )
        else:
            st.info("No source data available for this timeframe.")
    
    with col2:
        st.markdown("#### Top Destination IPs")
        if not top_talkers['destinations'].empty:
            # Format data for display
            display_df = top_talkers['destinations'].copy()
            display_df['total_kb'] = display_df['total_bytes'] / 1024
            
            # Create bar chart
            import plotly.express as px
            fig = px.bar(
                display_df,
                x='destination',
                y='packet_count',
                color='total_kb',
                labels={
                    'destination': 'Destination IP',
                    'packet_count': 'Packet Count',
                    'total_kb': 'Total KB'
                },
                title=f"{period} Top Destination IPs"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display data table
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "destination": "Destination IP",
                    "packet_count": st.column_config.NumberColumn("Packets"),
                    "total_kb": st.column_config.NumberColumn(
                        "KB Transferred",
                        format="%.2f KB"
                    )
                }
            )
        else:
            st.info("No destination data available for this timeframe.")

def display_trend_analysis_view() -> None:
    """Display trend analysis view"""
    st.header("Network Traffic Trend Analysis")
    
    # Get cache key for consistent caching
    cache_key = get_cache_key()
    
    # Get data for trend analysis
    df = cached_get_recent_packets(limit=5000, cache_key=cache_key)
    
    if df.empty:
        st.info("Insufficient data for trend analysis. Waiting for more network traffic...")
        return
    
    # Prepare data for analysis
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Time period selection
    time_periods = ["hour", "day", "week", "month"]
    selected_period = st.selectbox(
        "Select time period for trend analysis:",
        options=time_periods,
        index=time_periods.index(st.session_state.selected_timeframe)
    )
    
    # Get trend analysis from LLM analyzer
    trend_analysis = st.session_state.analyzer.analyze_trends(df, timeframe=selected_period)
    
    if trend_analysis["status"] == "error":
        st.error(trend_analysis["message"])
        return
    
    # Display trend statistics
    st.subheader("Trend Statistics")
    
    trend_stats = trend_analysis["trend_stats"]
    
    # Display trend direction with appropriate color
    direction = trend_stats.get("trend_direction", "unknown")
    percentage = trend_stats.get("trend_percentage", 0)
    
    if direction == "increasing":
        direction_color = "red" if percentage > 20 else "orange"
        direction_icon = "📈"
    elif direction == "decreasing":
        direction_color = "green" if percentage > 20 else "orange"
        direction_icon = "📉"
    else:
        direction_color = "blue"
        direction_icon = "📊"
    
    st.markdown(f"### Traffic Trend: <span style='color:{direction_color}'>{direction_icon} {direction.title()}</span>", unsafe_allow_html=True)
    
    # Display statistics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Packets",
            f"{trend_stats.get('avg_packets_per_interval', 0):.1f} pkts/{trend_stats.get('interval', '1min')}"
        )
    
    with col2:
        st.metric(
            "Maximum Traffic",
            f"{trend_stats.get('max_packets_per_interval', 0)} pkts/{trend_stats.get('interval', '1min')}"
        )
    
    with col3:
        if direction != "stable":
            st.metric(
                "Change Rate",
                f"{percentage:.1f}%",
                delta=f"{percentage:.1f}%" if direction == "increasing" else f"-{percentage:.1f}%"
            )
        else:
            st.metric("Change Rate", "Stable")
    
    # Display trend insights
    st.subheader("Trend Insights")
    
    insights = trend_analysis.get("insights", [])
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No trend insights available. Need more data for meaningful analysis.")
    
    # Create timeline visualization
    st.subheader("Traffic Timeline")
    
    timeline_df = cached_get_packet_timeline(
        interval='1min' if selected_period == 'hour' else '5min' if selected_period == 'day' else 'hour',
        timeframe=selected_period,
        cache_key=cache_key
    )
    
    if not timeline_df.empty:
        import plotly.express as px
        
        fig = px.line(
            timeline_df,
            x='timestamp',
            y='count',
            title=f"Traffic Volume over Time ({selected_period.capitalize()})",
            labels={
                'timestamp': 'Time',
                'count': 'Packet Count'
            }
        )
        
        # Add a trend line
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Packet Count",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient timeline data. Waiting for more traffic...")

def display_security_insights_view() -> None:
    """Display security insights and LLM analysis view"""
    st.header("Network Security Insights")
    
    # Get cache key for consistent caching
    cache_key = get_cache_key()
    
    # Get data for analysis
    df = cached_get_recent_packets(limit=1000, cache_key=cache_key)
    
    if df.empty:
        st.info("Insufficient data for security analysis. Waiting for more network traffic...")
        return
    
    # Run LLM analysis
    analysis_result = st.session_state.analyzer.analyze_traffic_patterns(df)
    
    if analysis_result["status"] == "insufficient_data":
        st.warning("Not enough data for comprehensive analysis. Basic insights only.")
    
    # Display security insights
    st.subheader("Network Traffic Insights")
    
    insights = analysis_result.get("insights", [])
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No significant insights detected. Need more data for meaningful analysis.")
    
    # Display anomalies if any
    anomalies = analysis_result.get("anomalies", [])
    if anomalies:
        st.subheader("Detected Anomalies")
        
        for anomaly in anomalies:
            anomaly_type = anomaly.get("type", "unknown")
            description = anomaly.get("description", "No description available")
            
            if anomaly_type == "large_packets":
                st.warning(f"⚠️ {description} (threshold: {anomaly.get('threshold', 0):.2f} bytes)")
            elif anomaly_type == "traffic_spike":
                st.warning(f"⚠️ {description} ({anomaly.get('count', 0)} packets, expected ~{anomaly.get('expected', 0):.1f})")
            elif anomaly_type == "protocol_dominance":
                st.warning(f"⚠️ {description} ({anomaly.get('percentage', 0)}% of traffic)")
            else:
                st.info(f"ℹ️ {description}")
    
    # Security recommendations
    st.subheader("Security Recommendations")
    
    recommendations = st.session_state.analyzer.generate_security_recommendations(df)
    
    for i, recommendation in enumerate(recommendations):
        st.markdown(f"{i+1}. {recommendation}")
    
    # Optional: Display network statistics
    with st.expander("Network Statistics Details"):
        stats = analysis_result.get("statistics", {})
        
        if stats:
            # Protocol distribution
            if "protocol_distribution" in stats:
                st.subheader("Protocol Distribution")
                
                # Convert to DataFrame for better display
                protocol_df = cached_get_protocol_distribution(timeframe='day', cache_key=cache_key)
                
                st.dataframe(protocol_df, use_container_width=True)
            
            # Top sources and destinations
            col1, col2 = st.columns(2)
            
            with col1:
                if "top_sources" in stats:
                    st.subheader("Top Source IPs")
                    source_df = pd.DataFrame({
                        'IP': list(stats["top_sources"].keys()),
                        'Count': list(stats["top_sources"].values())
                    }).sort_values('Count', ascending=False)
                    
                    st.dataframe(source_df, use_container_width=True)
            
            with col2:
                if "top_destinations" in stats:
                    st.subheader("Top Destination IPs")
                    dest_df = pd.DataFrame({
                        'IP': list(stats["top_destinations"].keys()),
                        'Count': list(stats["top_destinations"].values())
                    }).sort_values('Count', ascending=False)
                    
                    st.dataframe(dest_df, use_container_width=True)
        else:
            st.info("No detailed statistics available.")

def display_ai_solution_center_view() -> None:
    """Display AI Solution Center with Freshdesk search."""
    st.header("AI Solution Center")

    query = st.text_input("Search Freshdesk articles:")
    if query:
        with st.spinner("Searching Freshdesk..."):
            from solution_center import search_articles
            results = search_articles(query)
            if isinstance(results, dict) and results.get("error"):
                st.error(results["error"])
            elif results:
                for article in results:
                    title = article.get("title", "Untitled")
                    url = article.get("url", "#")
                    st.markdown(f"- [{title}]({url})")
            else:
                st.info("No articles found.")
