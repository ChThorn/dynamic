"""
Robot TCP Server Module
Handles TCP server communication for robot motion commands
"""

import socket
import threading
import time
from typing import Callable, Optional, List
from robot_logger import RobotLogger

class RobotTCPServer:
    """TCP Server for robot motion commands"""
    
    def __init__(self, host="0.0.0.0", port=7000, logger=None):
        """
        Initialize TCP server
        
        Args:
            host: Server host address
            port: Server port number
            logger: Logger instance
        """
        self.host = host
        self.port = port
        self.logger = logger or RobotLogger("TCPServer")
        
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        
        self.is_running = False
        self.connection_status = False
        self.server_thread = None
        
        self.data_received = []
        self.data_lock = threading.Lock()
        
        # Callbacks
        self.on_connection_callback: Optional[Callable] = None
        self.on_disconnection_callback: Optional[Callable] = None
        self.on_data_received_callback: Optional[Callable] = None
    
    # Callback setters
    def set_connection_callback(self, callback: Callable):
        """Set callback for new connections"""
        self.on_connection_callback = callback
    
    def set_disconnection_callback(self, callback: Callable):
        """Set callback for disconnections"""
        self.on_disconnection_callback = callback
    
    def set_data_received_callback(self, callback: Callable):
        """Set callback for received data"""
        self.on_data_received_callback = callback
    
    # Lifecycle methods (start/stop)
    def start_server(self) -> bool:
        """
        Start the TCP server
        
        Returns:
            bool: True if server started successfully
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            self.is_running = True
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            
            self.logger.info(f"TCP Server started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start TCP server: {e}")
            return False
    
    def stop_server(self):
        """Stop the TCP server"""
        if not self.is_running:
            return

        self.is_running = False
        
        # First close client socket if connected
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None

        # Close server socket properly
        if self.server_socket:
            try:
                # Shutdown the socket to unblock accept() calls
                self.server_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None

        # Wait for the server thread to terminate completely
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=3.0)  # Increased timeout
            if self.server_thread.is_alive():
                self.logger.warning("Server thread did not terminate cleanly")

        self.connection_status = False
        self.logger.info("TCP Server stopped")
        
        # Small delay to ensure OS releases the port completely
        import time
        time.sleep(0.5)
    
    # Private Internal Methods
    def _server_loop(self):     # Main loop waiting for client connections
        """Main server loop"""
        while self.is_running:
            try:
                if not self.connection_status:
                    self.logger.info("Waiting for client connection...")
                    self.client_socket, self.client_address = self.server_socket.accept()
                    
                    self.connection_status = True
                    self.logger.info(f"Client connected from {self.client_address}")
                    
                    if self.on_connection_callback:
                        self.on_connection_callback()
                    
                    # Start client handler thread
                    client_thread = threading.Thread(
                        target=self._handle_client, 
                        daemon=True
                    )
                    client_thread.start()
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Server loop error: {e}")
                break
    
    def _handle_client(self):
        """Handle client communication"""
        try:
            while self.is_running and self.connection_status:
                try:
                    data = self.client_socket.recv(1024)
                    if not data:
                        break
                    
                    # Store received data
                    with self.data_lock:
                        self.data_received.append(data)
                    
                    # Call callback if set
                    if self.on_data_received_callback:
                        self.on_data_received_callback(data)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Client communication error: {e}")
                    break
        
        except Exception as e:
            self.logger.error(f"Client handler error: {e}")
        
        finally:
            self._disconnect_client()
    
    def _disconnect_client(self):
        """Disconnect current client"""
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        
        self.connection_status = False
        self.client_address = None
        
        self.logger.info("Client disconnected")
        
        if self.on_disconnection_callback:
            self.on_disconnection_callback()
    
    # Communication methods
    def send_data(self, data: bytes) -> bool:       # Send raw binary data to client
        """
        Send data to connected client
        
        Args:
            data: Data to send
            
        Returns:
            bool: True if data sent successfully
        """
        if not self.connection_status or not self.client_socket:
            return False
        
        try:
            self.client_socket.send(data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send data: {e}")
            self._disconnect_client()
            return False
    
    def send_string(self, message: str) -> bool:    # Send text message to client
        """
        Send string message to connected client
        
        Args:
            message: String message to send
            
        Returns:
            bool: True if message sent successfully
        """
        return self.send_data(message.encode('utf-8'))
    
    def get_received_data(self) -> List[bytes]:
        """
        Get all received data from client
        
        Returns:
            List of received data packets
        """
        with self.data_lock:
            data = self.data_received.copy()
            self.data_received.clear()
            return data
    
    # Query methods
    def has_data(self) -> bool:
        """Check if new data arrived from client"""
        with self.data_lock:
            return len(self.data_received) > 0
    
    # Status Methods
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self.connection_status
    
    def get_client_address(self) -> Optional[tuple]:
        """Get IP address of connected client"""
        return self.client_address