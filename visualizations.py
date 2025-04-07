import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

# Safer check for cached functions
def has_caching_functions():
    """Check if caching functions are available in a safe way"""
    try:
        # Try to import the functions without executing them
        from dashboard_views import (
            cached_get_packets_by_timeframe,
            cached_get_protocol_distribution,
            cached_get_packet_timeline,
            cached_get_top_talkers,
            get_cache_key
        )
        return True
    except (ImportError, AttributeError):
        return False

def create_visualization(processor, timeframe: str = 'day', protocol_filter: Optional[str] = None) -> None:
    """Create all dashboard visualizations"""
    
    # Get data from database
    try:
        use_cache = has_caching_functions()
        
        if use_cache:
            # If dashboard_views is imported with caching functions
            from dashboard_views import cached_get_packets_by_timeframe, get_cache_key
            cache_key = get_cache_key()
            
            if protocol_filter:
                df = cached_get_packets_by_timeframe(hours=24, protocol=protocol_filter, cache_key=cache_key)
            else:
                df = cached_get_packets_by_timeframe(hours=24, protocol=None, cache_key=cache_key)
        else:
            # Fallback to direct access
            if protocol_filter:
                df = processor.get_dataframe(limit=1000, protocol=protocol_filter)
            else:
                df = processor.get_dataframe(limit=1000)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        df = pd.DataFrame()
    
    if len(df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Protocol distribution
            try:
                use_cache = has_caching_functions()
                
                if use_cache:
                    # Use cached version if available
                    from dashboard_views import cached_get_protocol_distribution, get_cache_key
                    cache_key = get_cache_key()
                    protocol_df = cached_get_protocol_distribution(timeframe=timeframe, cache_key=cache_key)
                else:
                    protocol_df = processor.get_protocol_distribution(timeframe=timeframe)
                
                if len(protocol_df) > 0:
                    fig_protocol = px.pie(
                        values=protocol_df['count'].values,
                        names=protocol_df['protocol'].values,
                        title=f'Protocol Distribution ({timeframe.capitalize()})',
                        hole=0.4
                    )
                    st.plotly_chart(fig_protocol, use_container_width=True)
                else:
                    st.info("No protocol data available for the selected timeframe.")
            except Exception as e:
                st.error(f"Error creating protocol distribution chart: {str(e)}")
        
        with col2:
            # Packet timeline
            try:
                use_cache = has_caching_functions()
                
                if use_cache:
                    # Use cached version if available
                    from dashboard_views import cached_get_packet_timeline, get_cache_key
                    cache_key = get_cache_key()
                    
                    timeline_df = cached_get_packet_timeline(
                        interval='1min' if timeframe == 'hour' else '5min' if timeframe == 'day' else 'hour',
                        timeframe=timeframe,
                        cache_key=cache_key
                    )
                else:
                    timeline_df = processor.get_packet_timeline(
                        interval='1min' if timeframe == 'hour' else '5min' if timeframe == 'day' else 'hour',
                        timeframe=timeframe
                    )
                
                if len(timeline_df) > 0:
                    fig_timeline = px.line(
                        timeline_df,
                        x='timestamp',
                        y='count',
                        title=f'Packets Over Time ({timeframe.capitalize()})'
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                else:
                    st.info("No timeline data available for the selected timeframe.")
            except Exception as e:
                st.error(f"Error creating timeline chart: {str(e)}")
        
        # Top talkers section
        st.subheader(f"Top Talkers ({timeframe.capitalize()})")
        
        try:
            use_cache = has_caching_functions()
            
            if use_cache:
                # Use cached version if available
                from dashboard_views import cached_get_top_talkers, get_cache_key
                cache_key = get_cache_key()
                top_talkers = cached_get_top_talkers(timeframe=timeframe, limit=10, cache_key=cache_key)
            else:
                top_talkers = processor.get_top_talkers(timeframe=timeframe)
            
            if top_talkers['sources'].empty and top_talkers['destinations'].empty:
                st.info("No top talkers data available for the selected timeframe.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Top Source IPs")
                    if not top_talkers['sources'].empty:
                        # Create bar chart for top sources
                        fig_sources = px.bar(
                            top_talkers['sources'],
                            x='source',
                            y='packet_count',
                            title='Top Source IPs',
                            labels={'source': 'IP Address', 'packet_count': 'Packet Count'}
                        )
                        st.plotly_chart(fig_sources, use_container_width=True)
                    else:
                        st.info("No source data available.")
                
                with col2:
                    st.markdown("#### Top Destination IPs")
                    if not top_talkers['destinations'].empty:
                        # Create bar chart for top destinations
                        fig_destinations = px.bar(
                            top_talkers['destinations'],
                            x='destination',
                            y='packet_count',
                            title='Top Destination IPs',
                            labels={'destination': 'IP Address', 'packet_count': 'Packet Count'}
                        )
                        st.plotly_chart(fig_destinations, use_container_width=True)
                    else:
                        st.info("No destination data available.")
                
                # Show top talkers data table
                with st.expander("View Top Talkers Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if not top_talkers['sources'].empty:
                            st.dataframe(
                                top_talkers['sources'],
                                use_container_width=True,
                                column_config={
                                    "source": "Source IP",
                                    "packet_count": "Packets",
                                    "total_bytes": st.column_config.NumberColumn(
                                        "Total Bytes",
                                        format="%.2f KB",
                                        help="Total data transferred in kilobytes",
                                    ),
                                }
                            )
                    
                    with col2:
                        if not top_talkers['destinations'].empty:
                            st.dataframe(
                                top_talkers['destinations'],
                                use_container_width=True,
                                column_config={
                                    "destination": "Destination IP",
                                    "packet_count": "Packets",
                                    "total_bytes": st.column_config.NumberColumn(
                                        "Total Bytes",
                                        format="%.2f KB",
                                        help="Total data transferred in kilobytes",
                                    ),
                                }
                            )
        except Exception as e:
            st.error(f"Error creating top talkers visualization: {str(e)}")
    else:
        st.warning("No data available for visualization. The dashboard will update automatically when network traffic is captured.")

def create_network_graph(processor, limit: int = 20) -> None:
    """Create a network graph visualization of top connections"""
    st.subheader("Network Connections Graph")
    
    try:
        use_cache = has_caching_functions()
        
        # Get top talkers data
        if use_cache:
            # Use cached version if available
            from dashboard_views import cached_get_top_talkers, get_cache_key
            cache_key = get_cache_key()
            top_talkers = cached_get_top_talkers(timeframe='day', limit=limit, cache_key=cache_key)
        else:
            top_talkers = processor.get_top_talkers(timeframe='day')
        
        if top_talkers['sources'].empty and top_talkers['destinations'].empty:
            st.info("Not enough data to create a network graph.")
            return
        
        # Get top sources and destinations
        top_sources = set(top_talkers['sources']['source'].head(limit).tolist())
        top_dests = set(top_talkers['destinations']['destination'].head(limit).tolist())
        
        # Get recent packets involving top talkers
        if use_cache:
            # Use cached version if available
            from dashboard_views import cached_get_recent_packets, get_cache_key
            cache_key = get_cache_key()
            df = cached_get_recent_packets(limit=1000, cache_key=cache_key)
        else:
            df = processor.get_dataframe(limit=1000)
        
        if df.empty:
            st.info("No packet data available for network graph.")
            return
        
        # Filter to only include top talkers
        df_filtered = df[
            (df['source'].isin(top_sources) | df['destination'].isin(top_dests))
        ]
        
        if df_filtered.empty:
            st.info("Not enough connection data for network graph.")
            return
        
        # Create network edges by counting source-destination pairs
        edges = df_filtered.groupby(['source', 'destination']).size().reset_index(name='weight')
        
        # Get top edges
        edges = edges.sort_values('weight', ascending=False).head(limit * 2)
        
        # Create nodes set from edges
        nodes = set(edges['source'].tolist() + edges['destination'].tolist())
        
        # Create network graph
        import networkx as nx
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node)
        
        # Add edges
        for _, row in edges.iterrows():
            G.add_edge(row['source'], row['destination'], weight=row['weight'])
        
        # Use networkx spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Create plotly figure
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        # Scale edge width based on weight
        max_weight = max(edge_weights) if edge_weights else 1
        scaled_weights = [1 + (w / max_weight) * 5 for w in edge_weights]
        
        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=scaled_weights, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_labels = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node)
        
        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_labels,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        # Color nodes by number of connections
        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
        
        nodes_trace.marker.color = node_adjacencies
        
        # Create figure
        fig = go.Figure(data=[edges_trace, nodes_trace])
        
        # Update layout
        fig.update_layout(
            title='Network Traffic Connection Graph',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating network graph: {str(e)}")
        st.info("Try refreshing the data or waiting for more network traffic to be captured.")
