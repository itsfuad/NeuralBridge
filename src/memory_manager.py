"""
Memory Manager Module

This module handles the extended memory/context layer for AI models.
It provides functionality to:
- Store conversation history
- Retrieve relevant context from past conversations
- Add metadata to conversations
- Use vector or traditional database for storage
"""

import logging
import uuid
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages conversation memory and context retrieval."""
    
    def __init__(self, db_manager, session_ttl: int = 24*60*60):
        """
        Initialize the memory manager.
        
        Args:
            db_manager: Database manager instance for storage
            session_ttl: Time-to-live for sessions in seconds (default: 24 hours)
        """
        self.db_manager = db_manager
        self.session_ttl = session_ttl
        self.active_sessions = {}
    
    def create_session(self, model_name: str = None, metadata: Dict = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            model_name: Name of the model being used
            metadata: Additional metadata to store with the session
            
        Returns:
            Session ID
        """
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Create session data
        session_data = {
            'id': session_id,
            'created_at': datetime.datetime.now().isoformat(),
            'model': model_name,
            'metadata': metadata or {},
            'messages': []
        }
        
        # Store session in database
        self.db_manager.save_session(session_id, session_data)
        
        # Add to active sessions
        self.active_sessions[session_id] = session_data
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def load_session(self, session_id: str) -> Dict:
        """
        Load an existing session.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            Session data
        """
        # Check if already loaded
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from database
        session_data = self.db_manager.get_session(session_id)
        
        if not session_data:
            raise ValueError(f"Session {session_id} not found")
        
        # Add to active sessions
        self.active_sessions[session_id] = session_data
        
        logger.info(f"Loaded session: {session_id}")
        return session_data
    
    def save_session(self, session_id: str) -> None:
        """
        Save a session to persistent storage.
        
        Args:
            session_id: ID of the session to save
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Cannot save unknown session: {session_id}")
            return
        
        # Get session data
        session_data = self.active_sessions[session_id]
        
        # Update last modified timestamp
        session_data['updated_at'] = datetime.datetime.now().isoformat()
        
        # Save to database
        self.db_manager.save_session(session_id, session_data)
        
        logger.info(f"Saved session: {session_id}")
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None) -> Dict:
        """
        Add a message to a session.
        
        Args:
            session_id: ID of the session to add the message to
            role: Role of the message sender (user, assistant, system)
            content: Message content
            metadata: Additional metadata to store with the message
            
        Returns:
            Added message
        """
        # Load session if not already loaded
        if session_id not in self.active_sessions:
            self.load_session(session_id)
        
        # Create message
        message = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.datetime.now().isoformat(),
            'role': role,
            'content': content,
            'metadata': metadata or {}
        }
        
        # Add to session
        self.active_sessions[session_id]['messages'].append(message)
        
        # Store message in vector database for semantic search
        self.db_manager.add_message(session_id, message)
        
        # Save session
        self.save_session(session_id)
        
        return message
    
    def get_messages(self, session_id: str, limit: int = None) -> List[Dict]:
        """
        Get messages from a session.
        
        Args:
            session_id: ID of the session to get messages from
            limit: Maximum number of messages to return (most recent)
            
        Returns:
            List of messages
        """
        # Load session if not already loaded
        if session_id not in self.active_sessions:
            self.load_session(session_id)
        
        messages = self.active_sessions[session_id]['messages']
        
        # Return most recent messages if limit is specified
        if limit is not None:
            return messages[-limit:]
        
        return messages
    
    def get_context(self, session_id: str, query: str, limit: int = 5) -> List[Dict]:
        """
        Get relevant context from past conversations based on semantic similarity.
        
        Args:
            session_id: ID of the current session
            query: Query to find relevant context for
            limit: Maximum number of context items to return
            
        Returns:
            List of relevant messages as context
        """
        # Get relevant messages from database
        context_messages = self.db_manager.search_messages(query, session_id, limit)
        
        logger.info(f"Retrieved {len(context_messages)} context messages for query: {query[:30]}...")
        return context_messages
    
    def get_session_summary(self, session_id: str) -> Dict:
        """
        Get a summary of a session.
        
        Args:
            session_id: ID of the session to summarize
            
        Returns:
            Session summary
        """
        # Load session if not already loaded
        if session_id not in self.active_sessions:
            self.load_session(session_id)
        
        session_data = self.active_sessions[session_id]
        
        return {
            'id': session_data['id'],
            'created_at': session_data['created_at'],
            'updated_at': session_data.get('updated_at'),
            'model': session_data['model'],
            'message_count': len(session_data['messages']),
            'metadata': session_data['metadata']
        }
    
    def list_sessions(self) -> List[Dict]:
        """
        List all available sessions.
        
        Returns:
            List of session summaries
        """
        session_ids = self.db_manager.list_sessions()
        return [self.get_session_summary(session_id) for session_id in session_ids]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if the session was deleted, False otherwise
        """
        # Remove from active sessions if present
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Remove from database
        result = self.db_manager.delete_session(session_id)
        
        if result:
            logger.info(f"Deleted session: {session_id}")
        
        return result
    
    def clear_context(self, session_id: str) -> None:
        """
        Clear the context/history of a session but keep the session itself.
        
        Args:
            session_id: ID of the session to clear
        """
        # Load session if not already loaded
        if session_id not in self.active_sessions:
            self.load_session(session_id)
        
        # Clear messages
        self.active_sessions[session_id]['messages'] = []
        
        # Save session
        self.save_session(session_id)
        
        # Clear messages from database
        self.db_manager.clear_session_messages(session_id)
        
        logger.info(f"Cleared context for session: {session_id}")
    
    def add_metadata(self, session_id: str, metadata: Dict) -> None:
        """
        Add metadata to a session.
        
        Args:
            session_id: ID of the session to add metadata to
            metadata: Metadata to add
        """
        # Load session if not already loaded
        if session_id not in self.active_sessions:
            self.load_session(session_id)
        
        # Update metadata
        self.active_sessions[session_id]['metadata'].update(metadata)
        
        # Save session
        self.save_session(session_id)
        
        logger.info(f"Added metadata to session: {session_id}")
    
    def create_memory_snapshot(self, session_id: str) -> str:
        """
        Create a snapshot of the current memory state.
        Useful for saving a specific point in the conversation.
        
        Args:
            session_id: ID of the session to snapshot
            
        Returns:
            Snapshot ID
        """
        # Load session if not already loaded
        if session_id not in self.active_sessions:
            self.load_session(session_id)
        
        # Get session data
        session_data = self.active_sessions[session_id]
        
        # Create snapshot
        snapshot_id = f"snapshot_{str(uuid.uuid4())}"
        snapshot_data = {
            'id': snapshot_id,
            'created_at': datetime.datetime.now().isoformat(),
            'session_id': session_id,
            'data': session_data
        }
        
        # Save snapshot
        self.db_manager.save_snapshot(snapshot_id, snapshot_data)
        
        logger.info(f"Created memory snapshot: {snapshot_id} for session: {session_id}")
        return snapshot_id
    
    def load_memory_snapshot(self, snapshot_id: str) -> Dict:
        """
        Load a memory snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to load
            
        Returns:
            Snapshot data
        """
        # Load snapshot from database
        snapshot_data = self.db_manager.get_snapshot(snapshot_id)
        
        if not snapshot_data:
            raise ValueError(f"Snapshot {snapshot_id} not found")
        
        logger.info(f"Loaded memory snapshot: {snapshot_id}")
        return snapshot_data 