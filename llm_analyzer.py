import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import json
import requests
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Analyze network traffic data using statistical methods and optionally LLM insights"""
    
    def __init__(self, api_key: Optional[str] = None, api_endpoint: Optional[str] = None, from_config: bool = False):
        """
        Initialize LLM analyzer
        
        Args:
            api_key: API key for the LLM service (optional)
            api_endpoint: Endpoint URL for the LLM service (optional)
            from_config: If True, load API key and endpoint from config
        """
        if from_config:
            try:
                from config import LLM_API_KEY, LLM_API_ENDPOINT, LLM_MODEL
                self.api_key = LLM_API_KEY
                self.api_endpoint = LLM_API_ENDPOINT
                self.model = LLM_MODEL
                logger.info("LLM configuration loaded from config")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not load LLM configuration from config: {str(e)}")
                self.api_key = api_key
                self.api_endpoint = api_endpoint
                self.model = "gpt-4o"
        else:
            self.api_key = api_key
            self.api_endpoint = api_endpoint
            self.model = "gpt-4o"
        
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 3600  # Cache time-to-live in seconds
    
    def analyze_traffic_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze traffic patterns in the network data"""
        if len(data) < 10:
            return {"status": "insufficient_data", "message": "Not enough data for analysis"}
        
        # Calculate basic statistics
        stats = self._calculate_statistics(data)
        
        # Look for anomalies
        anomalies = self._detect_anomalies(data, stats)
        
        # Prepare analysis result
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "anomalies": anomalies,
            "insights": self._generate_insights(data, stats, anomalies)
        }
        
        return result
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics from the data"""
        stats = {}
        
        # Protocol distribution
        if 'protocol' in data.columns:
            stats["protocol_distribution"] = data['protocol'].value_counts().to_dict()
        
        # Traffic volume by time
        if 'timestamp' in data.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
            
            # Group by hour
            hourly_traffic = data.groupby(pd.Grouper(key='timestamp', freq='1h')).size()
            stats["hourly_traffic"] = {str(dt): count for dt, count in zip(hourly_traffic.index, hourly_traffic.values)}
            
            # Calculate peak times
            if not hourly_traffic.empty:
                peak_hour_idx = hourly_traffic.idxmax()
                stats["peak_traffic_hour"] = peak_hour_idx.strftime('%Y-%m-%d %H:%M:%S')
                stats["peak_traffic_count"] = int(hourly_traffic.max())
        
        # Average packet size
        if 'size' in data.columns:
            stats["avg_packet_size"] = float(data['size'].mean())
            stats["max_packet_size"] = int(data['size'].max())
            stats["min_packet_size"] = int(data['size'].min())
        
        # Top source and destination IPs
        if 'source' in data.columns:
            top_sources = data['source'].value_counts().head(5).to_dict()
            stats["top_sources"] = top_sources
        
        if 'destination' in data.columns:
            top_destinations = data['destination'].value_counts().head(5).to_dict()
            stats["top_destinations"] = top_destinations
        
        return stats
    
    def _detect_anomalies(self, data: pd.DataFrame, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in network traffic"""
        anomalies = []
        
        try:
            # Check for unusual packet sizes
            if 'size' in data.columns and 'avg_packet_size' in stats:
                avg_size = stats['avg_packet_size']
                std_size = data['size'].std()
                
                # Look for unusually large packets (3 standard deviations above mean)
                large_packets = data[data['size'] > avg_size + 3 * std_size]
                if len(large_packets) > 0:
                    anomalies.append({
                        "type": "large_packets",
                        "description": f"Found {len(large_packets)} unusually large packets",
                        "threshold": float(avg_size + 3 * std_size),
                        "count": int(len(large_packets))
                    })
            
            # Check for unusual traffic spikes
            if 'hourly_traffic' in stats:
                hourly_counts = list(stats['hourly_traffic'].values())
                if len(hourly_counts) > 1:
                    avg_hourly = np.mean(hourly_counts)
                    std_hourly = np.std(hourly_counts)
                    
                    # Find hours with traffic 3 standard deviations above mean
                    for hour, count in stats['hourly_traffic'].items():
                        if count > avg_hourly + 3 * std_hourly:
                            anomalies.append({
                                "type": "traffic_spike",
                                "description": f"Unusual traffic spike at {hour}",
                                "count": count,
                                "expected": float(avg_hourly),
                                "hour": hour
                            })
            
            # Check for unusual protocol distribution
            if 'protocol_distribution' in stats:
                protocols = stats['protocol_distribution']
                total_packets = sum(protocols.values())
                
                # If more than 80% of traffic is a single protocol, flag it
                for protocol, count in protocols.items():
                    if count / total_packets > 0.8 and total_packets > 100:
                        anomalies.append({
                            "type": "protocol_dominance",
                            "description": f"Unusual dominance of {protocol} protocol",
                            "protocol": protocol,
                            "percentage": round(count / total_packets * 100, 2)
                        })
        
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}", exc_info=True)
            anomalies.append({
                "type": "error",
                "description": f"Error during anomaly detection: {str(e)}"
            })
        
        return anomalies
    
    def _generate_insights(self, data: pd.DataFrame, stats: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate human-readable insights about the network traffic"""
        insights = []
        
        # Check if we should use cached insights
        cache_key = f"insights_{datetime.now().strftime('%Y%m%d%h')}"
        if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        try:
            # Basic traffic insights
            if 'protocol_distribution' in stats:
                protocols = stats['protocol_distribution']
                top_protocol = max(protocols.items(), key=lambda x: x[1])
                insights.append(f"The most common protocol is {top_protocol[0]} with {top_protocol[1]} packets.")
            
            if 'avg_packet_size' in stats:
                insights.append(f"Average packet size is {stats['avg_packet_size']:.2f} bytes.")
            
            # Insights about anomalies
            for anomaly in anomalies:
                if anomaly['type'] == 'large_packets':
                    insights.append(f"Detected {anomaly['count']} unusually large packets that exceed {anomaly['threshold']:.2f} bytes.")
                
                elif anomaly['type'] == 'traffic_spike':
                    insights.append(f"Unusual traffic spike detected at {anomaly['hour']} with {anomaly['count']} packets (expected around {anomaly['expected']:.2f}).")
                
                elif anomaly['type'] == 'protocol_dominance':
                    insights.append(f"The {anomaly['protocol']} protocol accounts for {anomaly['percentage']}% of all traffic, which is unusually high.")
            
            # Traffic pattern insights
            if 'hourly_traffic' in stats and len(stats['hourly_traffic']) > 0:
                insights.append(f"Network traffic peaked at {stats.get('peak_traffic_hour', 'unknown time')} with {stats.get('peak_traffic_count', 0)} packets.")
            
            # Source/destination insights
            if 'top_sources' in stats and stats['top_sources']:
                top_source = list(stats['top_sources'].items())[0]
                insights.append(f"The most active source IP is {top_source[0]} with {top_source[1]} packets.")
            
            if 'top_destinations' in stats and stats['top_destinations']:
                top_dest = list(stats['top_destinations'].items())[0]
                insights.append(f"The most common destination IP is {top_dest[0]} with {top_dest[1]} packets.")
            
            # If we have an LLM API configured, try to get more advanced insights
            if self.api_key and self.api_endpoint:
                llm_insights = self._get_llm_insights(data, stats, anomalies)
                if llm_insights:
                    insights.extend(llm_insights)
            
            # Cache insights
            self.cache[cache_key] = insights
            self.cache_expiry[cache_key] = (datetime.now() + timedelta(hours=1)).timestamp()
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
            return [f"Error generating insights: {str(e)}"]
    
    def _get_llm_insights(self, data: pd.DataFrame, stats: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> List[str]:
        """Use LLM to generate advanced insights"""
        # This is a placeholder for integration with an LLM API
        try:
            if not self.api_key or not self.api_endpoint:
                return []
                
            # Prepare data for LLM
            context = {
                "data_sample": data.head(50).to_dict(orient='records'),
                "statistics": stats,
                "anomalies": anomalies,
                "timestamp": datetime.now().isoformat()
            }
            
            # Call LLM API
            response = requests.post(
                self.api_endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a network security analyst. Analyze the following network traffic data and provide insights about patterns, potential security issues, and recommendations."},
                        {"role": "user", "content": json.dumps(context)}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Split into insights
                if content:
                    insights = [line.strip() for line in content.split("\n") if line.strip()]
                    return insights[:5]  # Limit to top 5 insights
            
            return []
        
        except Exception as e:
            logger.error(f"Error getting LLM insights: {str(e)}", exc_info=True)
            return []
    
    def generate_security_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Generate security recommendations based on traffic analysis"""
        recommendations = []
        
        # Check for basic security issues
        stats = self._calculate_statistics(data)
        anomalies = self._detect_anomalies(data, stats)
        
        # Generate recommendations based on observed patterns
        if data.empty:
            return ["Insufficient data for security analysis"]
        
        # Check protocol distribution
        if 'protocol' in data.columns:
            protocols = data['protocol'].value_counts()
            
            # If no HTTPS/TLS traffic is detected
            if 'TCP' in protocols and not any(p for p in protocols.index if 'TLS' in p or 'SSL' in p):
                recommendations.append("Consider encrypting network traffic with TLS/SSL for sensitive applications.")
            
            # If there's a lot of UDP traffic
            if 'UDP' in protocols and protocols.get('UDP', 0) > 100:
                recommendations.append("High volume of UDP traffic detected. Consider implementing UDP flood protection.")
        
        # Check for potential port scanning
        if 'destination' in data.columns:
            # Group by source and count unique destinations
            source_dest_counts = data.groupby('source')['destination'].nunique()
            # Sources connecting to many destinations could be scanners
            potential_scanners = source_dest_counts[source_dest_counts > 10].index.tolist()
            
            if potential_scanners:
                recommendations.append(f"Potential port scanning detected from {len(potential_scanners)} IP addresses. Consider implementing network access controls.")
        
        # Basic firewall recommendation
        recommendations.append("Implement a properly configured firewall to control incoming and outgoing network traffic.")
        
        # If we detected any large packet anomalies
        if any(a for a in anomalies if a['type'] == 'large_packets'):
            recommendations.append("Unusually large packets detected. Consider implementing packet size limitations to prevent buffer overflow attacks.")
        
        # If we detected any traffic spikes
        if any(a for a in anomalies if a['type'] == 'traffic_spike'):
            recommendations.append("Traffic spikes detected. Consider implementing rate limiting or DDoS protection.")
        
        # General recommendations
        recommendations.append("Regularly update and patch all network devices and software.")
        recommendations.append("Implement network segmentation to isolate critical systems.")
        recommendations.append("Set up regular network traffic monitoring and alerting.")
        
        return recommendations
    
    def analyze_trends(self, data: pd.DataFrame, timeframe: str = 'day') -> Dict[str, Any]:
        """Analyze trends in network traffic over time"""
        if data.empty:
            return {"status": "error", "message": "No data available for trend analysis"}
        
        # Ensure timestamp is datetime
        if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        
        # Determine grouping frequency based on timeframe
        if timeframe == 'hour':
            freq = '1min'
        elif timeframe == 'day':
            freq = '1H'
        elif timeframe == 'week':
            freq = '1D'
        elif timeframe == 'month':
            freq = '1D'
        else:
            freq = '1H'  # Default
        
        try:
            # Group data by time
            traffic_over_time = data.groupby(pd.Grouper(key='timestamp', freq=freq)).size()
            
            # Calculate trend statistics
            trend_stats = {
                "timeframe": timeframe,
                "interval": freq,
                "total_packets": int(len(data)),
                "avg_packets_per_interval": float(traffic_over_time.mean()),
                "max_packets_per_interval": int(traffic_over_time.max()),
                "min_packets_per_interval": int(traffic_over_time.min()),
            }
            
            # Calculate trend direction (increasing, decreasing, stable)
            if len(traffic_over_time) > 1:
                first_half = traffic_over_time[:len(traffic_over_time)//2].mean()
                second_half = traffic_over_time[len(traffic_over_time)//2:].mean()
                
                if second_half > first_half * 1.2:
                    trend_stats["trend_direction"] = "increasing"
                    trend_stats["trend_percentage"] = float((second_half / first_half - 1) * 100)
                elif first_half > second_half * 1.2:
                    trend_stats["trend_direction"] = "decreasing"
                    trend_stats["trend_percentage"] = float((1 - second_half / first_half) * 100)
                else:
                    trend_stats["trend_direction"] = "stable"
                    trend_stats["trend_percentage"] = 0.0
            else:
                trend_stats["trend_direction"] = "unknown"
                trend_stats["trend_percentage"] = 0.0
            
            # Analyze protocol trends
            if 'protocol' in data.columns:
                # Group by time and protocol
                protocol_trends = data.groupby([pd.Grouper(key='timestamp', freq=freq), 'protocol']).size().unstack(fill_value=0)
                
                # Get the dominant protocol for each time interval
                dominant_protocols = {}
                for interval in protocol_trends.index:
                    if not protocol_trends.loc[interval].empty:
                        dominant_protocol = protocol_trends.loc[interval].idxmax()
                        dominant_protocols[str(interval)] = dominant_protocol
                
                trend_stats["dominant_protocols"] = dominant_protocols
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "trend_stats": trend_stats,
                "insights": self._generate_trend_insights(trend_stats)
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Error analyzing trends: {str(e)}"}
    
    def _generate_trend_insights(self, trend_stats: Dict[str, Any]) -> List[str]:
        """Generate insights from trend analysis"""
        insights = []
        
        try:
            # Traffic trend insights
            direction = trend_stats.get("trend_direction", "unknown")
            percentage = trend_stats.get("trend_percentage", 0)
            
            if direction == "increasing":
                insights.append(f"Network traffic is increasing by approximately {percentage:.1f}% over the {trend_stats.get('timeframe', 'selected')} period.")
                
                if percentage > 50:
                    insights.append("This significant traffic increase may indicate new services, applications, or potential security issues.")
            
            elif direction == "decreasing":
                insights.append(f"Network traffic is decreasing by approximately {percentage:.1f}% over the {trend_stats.get('timeframe', 'selected')} period.")
                
                if percentage > 50:
                    insights.append("This significant traffic decrease may indicate service outages or changes in network usage patterns.")
            
            elif direction == "stable":
                insights.append(f"Network traffic is relatively stable over the {trend_stats.get('timeframe', 'selected')} period.")
            
            # Protocol insights
            if "dominant_protocols" in trend_stats and trend_stats["dominant_protocols"]:
                # Count occurrences of each dominant protocol
                protocol_counts = {}
                for protocol in trend_stats["dominant_protocols"].values():
                    protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
                
                # Find the most common dominant protocol
                if protocol_counts:
                    top_protocol = max(protocol_counts.items(), key=lambda x: x[1])
                    total_intervals = len(trend_stats["dominant_protocols"])
                    
                    insights.append(f"{top_protocol[0]} is the dominant protocol in {top_protocol[1]} out of {total_intervals} time intervals.")
            
            # Traffic volume insights
            if "max_packets_per_interval" in trend_stats and "avg_packets_per_interval" in trend_stats:
                max_packets = trend_stats["max_packets_per_interval"]
                avg_packets = trend_stats["avg_packets_per_interval"]
                
                if max_packets > avg_packets * 2:
                    insights.append(f"Traffic spikes detected with max volume ({max_packets} packets) significantly higher than average ({avg_packets:.1f} packets).")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating trend insights: {str(e)}", exc_info=True)
            return [f"Error generating trend insights: {str(e)}"]
