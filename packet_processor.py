import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
import pandas as pd
from scapy.all import sniff, IP
import logging
from database_handler import DatabaseHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PacketProcessor:
    """Process and analyze network packets"""
    
    def __init__(self, db_path: str = "network_traffic.db"):
        self.protocol_map = {
            1: 'ICMP',
            6: 'TCP',
            17: 'UDP',
            # Add more protocols here
            2: 'IGMP',
            8: 'EGP',
            9: 'IGP',
            41: 'IPv6',
            46: 'RSVP',
            47: 'GRE',
            50: 'ESP',
            51: 'AH',
            88: 'EIGRP',
            89: 'OSPF',
            103: 'PIM',
            132: 'SCTP'
        }
        
        # In-memory buffer for efficiency
        self.packet_buffer: List[Dict[str, any]] = []
        self.buffer_size: int = 100  # Flush to DB when buffer reaches this size
        
        self.start_time: datetime = datetime.now()
        self.lock: threading.Lock = threading.Lock()
        self.running: bool = True
        
        # Initialize database handler
        self.db = DatabaseHandler(db_path)
        
        # For filtering
        self.active_filters: Dict[str, Set[str]] = {
            'protocol': set(),
            'source': set(),
            'destination': set()
        }
        
    def start_capture(self) -> None:
        """Start a thread to capture packets"""
        self.capture_thread = threading.Thread(target=self.capture_packets)
        self.capture_thread.daemon = True  # Allow the thread to exit when the main program exits
        self.capture_thread.start()
        
        # Start a maintenance thread to periodically cleanup old data
        self.maintenance_thread = threading.Thread(target=self.maintenance_routine)
        self.maintenance_thread.daemon = True
        self.maintenance_thread.start()

    def stop_capture(self) -> None:
        """Stop the packet capturing thread"""
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)  # Wait up to 2 seconds
        
        # Final flush of buffer to database
        self.flush_buffer()

    def capture_packets(self) -> None:
        """Capture packets using scapy"""
        try:
            sniff(prn=self.process_packet, stop_filter=lambda x: not self.running)
        except Exception as e:
            logger.error(f"Error in packet capture: {str(e)}", exc_info=True)

    def process_packet(self, packet) -> None:
        """Process a captured packet"""
        try:
            if IP in packet:
                with self.lock:
                    self._add_packet_data(packet)
        except Exception as e:
            logger.error(f'Error processing packet: {str(e)}', exc_info=True)

    def _add_packet_data(self, packet) -> None:
        """Add packet data to the buffer and flush if needed"""
        timestamp = time.time()
        source = packet[IP].src
        destination = packet[IP].dst
        protocol = packet[IP].proto
        size = len(packet)
        
        protocol_name = self.protocol_map.get(protocol, str(protocol))
        
        # Apply filters if active
        if (self.active_filters['protocol'] and 
            protocol_name not in self.active_filters['protocol']):
            return
            
        if (self.active_filters['source'] and 
            source not in self.active_filters['source']):
            return
            
        if (self.active_filters['destination'] and 
            destination not in self.active_filters['destination']):
            return
        
        packet_data = {
            'timestamp': timestamp,
            'source': source,
            'destination': destination,
            'protocol': protocol_name,
            'size': size
        }
        
        self.packet_buffer.append(packet_data)
        
        # Flush buffer to database when it reaches the threshold
        if len(self.packet_buffer) >= self.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self) -> None:
        """Flush the packet buffer to the database"""
        try:
            if self.packet_buffer:
                # Make a copy of the buffer and clear it
                buffer_copy = self.packet_buffer.copy()
                self.packet_buffer = []
                
                # Insert data into database (outside the lock)
                self.db.insert_many_packets(buffer_copy)
        except Exception as e:
            logger.error(f"Error flushing buffer: {str(e)}", exc_info=True)

    def get_dataframe(self, limit: int = 100, protocol: Optional[str] = None) -> pd.DataFrame:
        """Get recent packets as a pandas DataFrame"""
        # First flush any pending packets
        with self.lock:
            self.flush_buffer()
        
        # Get packets from database
        if protocol:
            df = self.db.get_packets_by_timeframe(hours=1, protocol=protocol)
            return df.head(limit)
        else:
            return self.db.get_recent_packets(limit=limit)
    
    def set_filter(self, filter_type: str, values: Set[str]) -> None:
        """Set a filter for packet processing"""
        if filter_type in self.active_filters:
            self.active_filters[filter_type] = values
    
    def clear_filters(self) -> None:
        """Clear all filters"""
        for key in self.active_filters:
            self.active_filters[key] = set()
    
    def get_total_packets(self) -> int:
        """Get total number of packets in the database"""
        return self.db.get_total_packet_count()
    
    def get_top_talkers(self, timeframe: str = 'day') -> Dict[str, pd.DataFrame]:
        """Get top talkers for the specified timeframe"""
        return self.db.get_top_talkers(timeframe=timeframe)
    
    def get_protocol_distribution(self, timeframe: str = 'day') -> pd.DataFrame:
        """Get protocol distribution for the specified timeframe"""
        return self.db.get_protocol_distribution(timeframe=timeframe)
    
    def get_packet_timeline(self, interval: str = '1min', timeframe: str = 'hour') -> pd.DataFrame:
        """Get packet timeline for visualization"""
        return self.db.get_packet_timeline(interval=interval, timeframe=timeframe)
    
    def maintenance_routine(self) -> None:
        """Background routine for database maintenance"""
        while self.running:
            try:
                # Sleep for 6 hours between maintenance runs
                for _ in range(6 * 60 * 60):
                    if not self.running:
                        break
                    time.sleep(1)
                
                if not self.running:
                    break
                
                # Cleanup old data (keep last 30 days by default)
                logger.info("Running database maintenance...")
                self.db.cleanup_old_data(days_to_keep=30)
                
            except Exception as e:
                logger.error(f"Error in maintenance routine: {str(e)}", exc_info=True)
