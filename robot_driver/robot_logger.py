"""
Robot Logger Module
Simple logging utility for robot operations
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

class RobotLogger:
    """Simple logger for robot operations"""
    
    def __init__(self, name="RobotDriver", log_file=None, level=logging.INFO):
        """
        Initialize the logger
        
        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'       # Create formatter as an example 2023-10-01 12:00:00 - RobotDriver - INFO - Message
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def write(self, message, level="info"):
        """
        Write a log message
        
        Args:
            message: Message to log
            level: Log level (info, warning, error, debug)
        """
        level = level.lower()
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
        else:
            self.logger.info(message)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)