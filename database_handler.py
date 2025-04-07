import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseHandler:
    """Handle database operations for network packets"""
    
    def __init__(self, db_path: str = "network_traffic.db"):
        """Initialize database connection and create tables if they don't exist"""
        self.db_path = db_path
        self.lock = threading.RLock()  # Use reentrant lock for database access
        self._create_tables()
        
    def _create_tables(self) -> None:
        """Create necessary tables if they don't exist"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create packets table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS packets (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                source TEXT,
                destination TEXT,
                protocol TEXT,
                size INTEGER
            )
            ''')
            
            # Create index on timestamp for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON packets(timestamp)')
            
            # Create index on source and destination for top talkers queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON packets(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_destination ON packets(destination)')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}", exc_info=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings"""
        conn = sqlite3.connect(
            self.db_path, 
            timeout=60.0,  # Longer timeout
            isolation_level=None  # Auto-commit mode
        )
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Better performance
        conn.execute("PRAGMA cache_size=10000")  # Larger cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
        return conn
    
    def insert_packet(self, packet_data: Dict[str, Any]) -> None:
        """Insert a packet into the database"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute(
                    'INSERT INTO packets (timestamp, source, destination, protocol, size) VALUES (?, ?, ?, ?, ?)',
                    (
                        packet_data['timestamp'],
                        packet_data['source'],
                        packet_data['destination'],
                        packet_data['protocol'],
                        packet_data['size']
                    )
                )
                
                conn.close()
        except Exception as e:
            logger.error(f"Error inserting packet: {str(e)}", exc_info=True)
    
    def insert_many_packets(self, packet_data_list: List[Dict[str, Any]]) -> None:
        """Insert multiple packets into the database"""
        if not packet_data_list:
            return
            
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                data = [(
                    p['timestamp'],
                    p['source'],
                    p['destination'],
                    p['protocol'],
                    p['size']
                ) for p in packet_data_list]
                
                cursor.executemany(
                    'INSERT INTO packets (timestamp, source, destination, protocol, size) VALUES (?, ?, ?, ?, ?)',
                    data
                )
                
                conn.close()
        except Exception as e:
            logger.error(f"Error inserting multiple packets: {str(e)}", exc_info=True)
    
    def get_recent_packets(self, limit: int = 10) -> pd.DataFrame:
        """Get the most recent packets"""
        try:
            conn = self._get_connection()
            
            query = f'''
            SELECT timestamp, source, destination, protocol, size 
            FROM packets 
            ORDER BY timestamp DESC 
            LIMIT {limit}
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error getting recent packets: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def get_packets_by_timeframe(self, hours: float = 1, protocol: Optional[str] = None) -> pd.DataFrame:
        """Get packets from the last X hours, optionally filtered by protocol"""
        try:
            conn = self._get_connection()
            
            timestamp_threshold = datetime.now().timestamp() - (hours * 3600)
            
            if protocol:
                query = f'''
                SELECT timestamp, source, destination, protocol, size 
                FROM packets 
                WHERE timestamp > {timestamp_threshold} AND protocol = '{protocol}'
                ORDER BY timestamp DESC
                '''
            else:
                query = f'''
                SELECT timestamp, source, destination, protocol, size 
                FROM packets 
                WHERE timestamp > {timestamp_threshold}
                ORDER BY timestamp DESC
                '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error getting packets by timeframe: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def get_top_talkers(self, timeframe: str = 'day', limit: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Get top talkers (sources and destinations) by specified timeframe
        
        Args:
            timeframe: 'day', 'week', 'month', or 'year'
            limit: Number of top talkers to return
            
        Returns:
            Dictionary with 'sources' and 'destinations' DataFrames
        """
        try:
            conn = self._get_connection()
            
            # Calculate timestamp threshold based on timeframe
            now = datetime.now()
            if timeframe == 'day':
                threshold = (now - timedelta(days=1)).timestamp()
            elif timeframe == 'week':
                threshold = (now - timedelta(weeks=1)).timestamp()
            elif timeframe == 'month':
                threshold = (now - timedelta(days=30)).timestamp()
            elif timeframe == 'year':
                threshold = (now - timedelta(days=365)).timestamp()
            else:
                threshold = (now - timedelta(days=1)).timestamp()  # Default to day
            
            # Get top sources
            source_query = f'''
            SELECT source, COUNT(*) as packet_count, SUM(size) as total_bytes
            FROM packets
            WHERE timestamp > {threshold}
            GROUP BY source
            ORDER BY packet_count DESC
            LIMIT {limit}
            '''
            
            # Get top destinations
            dest_query = f'''
            SELECT destination, COUNT(*) as packet_count, SUM(size) as total_bytes
            FROM packets
            WHERE timestamp > {threshold}
            GROUP BY destination
            ORDER BY packet_count DESC
            LIMIT {limit}
            '''
            
            source_df = pd.read_sql_query(source_query, conn)
            dest_df = pd.read_sql_query(dest_query, conn)
            
            conn.close()
            return {
                'sources': source_df,
                'destinations': dest_df
            }
        except Exception as e:
            logger.error(f"Error getting top talkers: {str(e)}", exc_info=True)
            return {
                'sources': pd.DataFrame(),
                'destinations': pd.DataFrame()
            }
    
    def get_protocol_distribution(self, timeframe: str = 'day') -> pd.DataFrame:
        """Get protocol distribution for the specified timeframe"""
        try:
            conn = self._get_connection()
            
            # Calculate timestamp threshold based on timeframe
            now = datetime.now()
            if timeframe == 'day':
                threshold = (now - timedelta(days=1)).timestamp()
            elif timeframe == 'week':
                threshold = (now - timedelta(weeks=1)).timestamp()
            elif timeframe == 'month':
                threshold = (now - timedelta(days=30)).timestamp()
            elif timeframe == 'year':
                threshold = (now - timedelta(days=365)).timestamp()
            else:
                threshold = (now - timedelta(days=1)).timestamp()  # Default to day
            
            query = f'''
            SELECT protocol, COUNT(*) as count
            FROM packets
            WHERE timestamp > {threshold}
            GROUP BY protocol
            ORDER BY count DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error getting protocol distribution: {str(e)}", exc_info=True)
            return pd.DataFrame()
                
    def get_packet_timeline(self, interval: str = '1min', timeframe: str = 'hour') -> pd.DataFrame:
        """
        Get packet timeline for visualization
        
        Args:
            interval: Time grouping interval ('1min', '5min', 'hour')
            timeframe: 'hour', 'day', 'week'
            
        Returns:
            DataFrame with timestamp and count columns
        """
        try:
            conn = self._get_connection()
            
            # Calculate timestamp threshold based on timeframe
            now = datetime.now()
            if timeframe == 'hour':
                threshold = (now - timedelta(hours=1)).timestamp()
            elif timeframe == 'day':
                threshold = (now - timedelta(days=1)).timestamp()
            elif timeframe == 'week':
                threshold = (now - timedelta(weeks=1)).timestamp()
            else:
                threshold = (now - timedelta(hours=1)).timestamp()  # Default to hour
            
            # Get raw data
            query = f'''
            SELECT timestamp
            FROM packets
            WHERE timestamp > {threshold}
            ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert timestamp to datetime
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Group by interval
                if interval == '1min':
                    df_grouped = df.groupby(pd.Grouper(key='timestamp', freq='1min')).size().reset_index(name='count')
                elif interval == '5min':
                    df_grouped = df.groupby(pd.Grouper(key='timestamp', freq='5min')).size().reset_index(name='count')
                elif interval == 'hour':
                    df_grouped = df.groupby(pd.Grouper(key='timestamp', freq='1H')).size().reset_index(name='count')
                else:
                    df_grouped = df.groupby(pd.Grouper(key='timestamp', freq='1min')).size().reset_index(name='count')
                    
                return df_grouped
            else:
                return pd.DataFrame(columns=['timestamp', 'count'])
        except Exception as e:
            logger.error(f"Error getting packet timeline: {str(e)}", exc_info=True)
            return pd.DataFrame(columns=['timestamp', 'count'])
                
    def get_total_packet_count(self) -> int:
        """Get total number of packets in the database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM packets')
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Error getting total packet count: {str(e)}", exc_info=True)
            return 0
                
    def get_database_size(self) -> float:
        """Get database file size in MB"""
        try:
            # Check for WAL files too
            main_db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            wal_size = os.path.getsize(f"{self.db_path}-wal") if os.path.exists(f"{self.db_path}-wal") else 0
            shm_size = os.path.getsize(f"{self.db_path}-shm") if os.path.exists(f"{self.db_path}-shm") else 0
            
            total_size_bytes = main_db_size + wal_size + shm_size
            size_mb = total_size_bytes / (1024 * 1024)
            return size_mb
        except Exception as e:
            logger.error(f"Error getting database size: {str(e)}", exc_info=True)
            return 0.0
            
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Remove data older than specified days"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                threshold = (datetime.now() - timedelta(days=days_to_keep)).timestamp()
                
                # Check how many records will be deleted
                cursor.execute(f'SELECT COUNT(*) FROM packets WHERE timestamp < {threshold}')
                count_to_delete = cursor.fetchone()[0]
                
                if count_to_delete > 0:
                    # Delete in smaller batches to avoid locking the database for too long
                    batch_size = 1000
                    deleted_count = 0
                    
                    while deleted_count < count_to_delete:
                        cursor.execute(f'''
                        DELETE FROM packets 
                        WHERE id IN (
                            SELECT id FROM packets 
                            WHERE timestamp < {threshold} 
                            LIMIT {batch_size}
                        )
                        ''')
                        deleted_this_batch = cursor.rowcount
                        deleted_count += deleted_this_batch
                        
                        if deleted_this_batch < batch_size:
                            break
                    
                    # Vacuum database to reclaim space
                    conn.execute("PRAGMA optimize")  # Optimize database
                    conn.execute("VACUUM")  # Reclaim space
                
                conn.close()
                logger.info(f"Cleaned up {deleted_count} records older than {days_to_keep} days")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}", exc_info=True)
            
    def clear_all_data(self) -> bool:
        """Clear all data from the database"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Get total record count first
                cursor.execute('SELECT COUNT(*) FROM packets')
                total_count = cursor.fetchone()[0]
                
                if total_count > 10000:
                    # For large databases, delete in batches
                    batch_size = 5000
                    deleted_total = 0
                    
                    while deleted_total < total_count:
                        cursor.execute(f'DELETE FROM packets LIMIT {batch_size}')
                        deleted_batch = cursor.rowcount
                        deleted_total += deleted_batch
                        
                        if deleted_batch < batch_size:
                            break
                else:
                    # For smaller databases, delete all at once
                    cursor.execute('DELETE FROM packets')
                
                # Vacuum database to reclaim space
                conn.execute("PRAGMA optimize")
                conn.execute("VACUUM")
                conn.close()
                
                # Also clean up any WAL files
                try:
                    if os.path.exists(f"{self.db_path}-wal"):
                        os.remove(f"{self.db_path}-wal")
                    if os.path.exists(f"{self.db_path}-shm"):
                        os.remove(f"{self.db_path}-shm")
                except:
                    pass  # Ignore errors from trying to remove these files
                
                logger.info("Cleared all data from the database")
                return True
        except Exception as e:
            logger.error(f"Error clearing data: {str(e)}", exc_info=True)
            return False
            
    def optimize_database(self) -> None:
        """Run database optimization and maintenance"""
        try:
            conn = self._get_connection()
            conn.execute("PRAGMA optimize")
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            conn.close()
            logger.info("Database optimized")
        except Exception as e:
            logger.error(f"Error optimizing database: {str(e)}", exc_info=True)
