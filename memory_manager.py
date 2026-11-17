import os
import gc
import logging
import threading
import time
import psutil
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory usage for the application to prevent crashes"""
    
    def __init__(self, warning_threshold=70, critical_threshold=85, check_interval=300):
        """
        Initialize the memory manager
        
        Args:
            warning_threshold: Percentage of RAM use that triggers a warning
            critical_threshold: Percentage of RAM use that triggers cleanup actions
            check_interval: How often to check memory (in seconds)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.process = psutil.Process(os.getpid())
        self.running = True
        self.last_cleanup_time = 0
        self.lock = threading.RLock()
    
    def start_monitoring(self):
        """Start monitoring memory usage in a background thread"""
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop the memory monitoring thread"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self):
        """Monitor memory usage and take action if needed"""
        while self.running:
            try:
                usage_percent = self.get_memory_usage()
                
                if usage_percent > self.critical_threshold:
                    logger.warning(f"CRITICAL: Memory usage at {usage_percent:.1f}%, performing emergency cleanup")
                    self.perform_cleanup(aggressive=True)
                    
                    # If in Streamlit, show a warning to the user
                    try:
                        st.warning(f"⚠️ High memory usage detected ({usage_percent:.1f}%). Emergency cleanup performed.")
                    except:
                        pass
                
                elif usage_percent > self.warning_threshold:
                    logger.info(f"WARNING: Memory usage high at {usage_percent:.1f}%")
                    self.perform_cleanup(aggressive=False)
            
            except Exception as e:
                logger.error(f"Error in memory monitoring: {str(e)}")
            
            # Sleep for the check interval
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def get_memory_usage(self):
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def perform_cleanup(self, aggressive=False):
        """
        Perform memory cleanup
        
        Args:
            aggressive: If True, perform more aggressive cleanup
        """
        current_time = time.time()
        
        # Don't cleanup too frequently
        if current_time - self.last_cleanup_time < 60 and not aggressive:
            return
        
        with self.lock:
            self.last_cleanup_time = current_time
            
            # Force Python garbage collection
            gc.collect()
            
            if aggressive:
                # Clear Streamlit cache if possible
                try:
                    from dashboard_views import reset_cache
                    reset_cache()
                    logger.info("Reset Streamlit cache")
                except:
                    logger.warning("Could not reset Streamlit cache")
                
                # Clear database connections and optimize
                try:
                    if 'processor' in st.session_state and hasattr(st.session_state.processor, 'db'):
                        st.session_state.processor.db.optimize_database()
                        logger.info("Optimized database")
                except:
                    logger.warning("Could not optimize database")
        
        logger.info(f"Memory cleanup performed (aggressive={aggressive})")
    
    def manual_cleanup(self):
        """Manually trigger a memory cleanup"""
        self.perform_cleanup(aggressive=True)
        return self.get_memory_usage()
