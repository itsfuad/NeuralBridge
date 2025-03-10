"""
Database Manager Module

This module handles data storage for the AI model runner system.
It supports both vector databases (for semantic search) and traditional SQL databases.
"""

import os
import json
import logging
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FAISS_INDEX_FILENAME = 'faiss_index.bin'

class DatabaseManager:
    """Manages data storage for sessions, messages, and vector embeddings."""
    
    def __init__(self, db_type: str = 'vector', data_dir: str = 'data'):
        """
        Initialize the database manager.
        
        Args:
            db_type: Type of database to use ('vector' or 'sql')
            data_dir: Directory to store database files
        """
        self.db_type = db_type.lower()
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize database based on type
        if self.db_type == 'vector':
            self._init_vector_db()
        elif self.db_type == 'sql':
            self._init_sql_db()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        logger.info(f"Initialized {self.db_type} database in {data_dir}")
    
    def _init_vector_db(self) -> None:
        """Initialize vector database."""
        try:
            # Import here to avoid dependency if not needed
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            # Create vector index directory
            vector_dir = os.path.join(self.data_dir, 'vector')
            os.makedirs(vector_dir, exist_ok=True)
            
            # Initialize sentence transformer for embedding generation
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize or load FAISS index
            index_path = os.path.join(vector_dir, FAISS_INDEX_FILENAME)
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info(f"Loaded existing FAISS index from {index_path}")
            else:
                # Create a new index - using L2 distance
                embedding_size = self.encoder.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(embedding_size)
                logger.info(f"Created new FAISS index with embedding size {embedding_size}")
            
            # Initialize lookup table for message data
            self.lookup_file = os.path.join(vector_dir, 'lookup.json')
            if os.path.exists(self.lookup_file):
                with open(self.lookup_file, 'r') as f:
                    self.lookup = json.load(f)
            else:
                self.lookup = {
                    'next_id': 0,
                    'messages': {},  # Maps vector ID to message data
                    'sessions': {}   # Maps session ID to session data
                }
            
            self.use_vector = True
            
        except ImportError:
            logger.warning("Vector database dependencies not found. Falling back to SQL database.")
            self.db_type = 'sql'
            self.use_vector = False
            self._init_sql_db()
    
    def _init_sql_db(self) -> None:
        """Initialize SQL database."""
        # Create database file
        self.db_path = os.path.join(self.data_dir, 'chatbot.db')
        self.conn = sqlite3.connect(self.db_path)
        
        # Create tables if they don't exist
        cursor = self.conn.cursor()
        
        # Sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        ''')
        
        # Messages table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        ''')
        
        # Snapshots table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        ''')
        
        self.conn.commit()
        self.use_vector = False
    
    def _save_vector_lookup(self) -> None:
        """Save the vector lookup table to disk."""
        if not hasattr(self, 'lookup_file'):
            return
            
        with open(self.lookup_file, 'w') as f:
            json.dump(self.lookup, f)
    
    def save_session(self, session_id: str, session_data: Dict) -> None:
        """
        Save a session to the database.
        
        Args:
            session_id: ID of the session
            session_data: Session data to save
        """
        if self.use_vector:
            # Save to vector database lookup
            self.lookup['sessions'][session_id] = session_data
            self._save_vector_lookup()
        else:
            # Save to SQL database
            cursor = self.conn.cursor()
            created_at = session_data.get('created_at', '')
            updated_at = session_data.get('updated_at', '')
            
            session_json = json.dumps(session_data)
            
            cursor.execute('''
            INSERT OR REPLACE INTO sessions (id, data, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ''', (session_id, session_json, created_at, updated_at))
            
            self.conn.commit()
        
        logger.debug(f"Saved session: {session_id}")
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get a session from the database.
        
        Args:
            session_id: ID of the session to get
            
        Returns:
            Session data or None if not found
        """
        if self.use_vector:
            # Get from vector database lookup
            return self.lookup['sessions'].get(session_id)
        else:
            # Get from SQL database
            cursor = self.conn.cursor()
            cursor.execute('SELECT data FROM sessions WHERE id = ?', (session_id,))
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from the database.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if the session was deleted, False otherwise
        """
        if self.use_vector:
            # Delete from vector database lookup
            if session_id in self.lookup['sessions']:
                del self.lookup['sessions'][session_id]
                
                # Delete associated message vectors
                # (This is a simplistic approach - in a real system we'd need to track which
                # vector IDs belong to which session and rebuild the index)
                
                self._save_vector_lookup()
                return True
            
            return False
        else:
            # Delete from SQL database
            cursor = self.conn.cursor()
            
            # Delete messages first (foreign key constraint)
            cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            
            # Delete snapshots
            cursor.execute('DELETE FROM snapshots WHERE session_id = ?', (session_id,))
            
            # Delete session
            cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            
            self.conn.commit()
            
            return cursor.rowcount > 0
    
    def list_sessions(self) -> List[str]:
        """
        List all available sessions.
        
        Returns:
            List of session IDs
        """
        if self.use_vector:
            # List from vector database lookup
            return list(self.lookup['sessions'].keys())
        else:
            # List from SQL database
            cursor = self.conn.cursor()
            cursor.execute('SELECT id FROM sessions')
            return [row[0] for row in cursor.fetchall()]
    
    def add_message(self, session_id: str, message: Dict) -> None:
        """
        Add a message to the database.
        
        Args:
            session_id: ID of the session the message belongs to
            message: Message data to add
        """
        if self.use_vector:
            # Add to vector database
            try:
                content = message['content']
                
                # Generate embedding for the message content
                embedding = self.encoder.encode(content)
                
                # Add to FAISS index
                index_id = self.lookup['next_id']
                self.index.add(embedding.reshape(1, -1))
                
                # Store message data with the vector ID
                self.lookup['messages'][str(index_id)] = {
                    'message': message,
                    'session_id': session_id
                }
                
                # Increment next ID
                self.lookup['next_id'] += 1
                
                # Save lookup table
                self._save_vector_lookup()
                
                # Save updated index periodically
                if index_id % 100 == 0:
                    import faiss
                    index_path = os.path.join(self.data_dir, 'vector', FAISS_INDEX_FILENAME)
                    faiss.write_index(self.index, index_path)
                
            except Exception as e:
                logger.error(f"Error adding message to vector database: {e}")
        
        # Always add to SQL/JSON storage for history
        if not self.use_vector:
            # Add to SQL database
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO messages (id, session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                message['id'],
                session_id,
                message['role'],
                message['content'],
                message['timestamp'],
                json.dumps(message.get('metadata', {}))
            ))
            
            self.conn.commit()
        
        logger.debug(f"Added message {message['id']} to session {session_id}")
    
    def _search_vector_db(self, query: str, session_id: Optional[str], limit: int) -> List[Dict]:
        """Search messages in vector database."""
        try:
            query_embedding = self.encoder.encode(query)
            distances, indices = self.index.search(query_embedding.reshape(1, -1), limit * 3)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                    
                message_data = self.lookup['messages'].get(str(idx))
                if not message_data or (session_id and message_data['session_id'] != session_id):
                    continue
                
                message = message_data['message'].copy()
                message['score'] = float(distances[0][i])
                results.append(message)
                
                if len(results) >= limit:
                    break
            
            return results
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []

    def _search_sql_db(self, query: str, session_id: Optional[str], limit: int) -> List[Dict]:
        """Search messages in SQL database."""
        cursor = self.conn.cursor()
        
        if session_id:
            cursor.execute('''
            SELECT id, role, content, timestamp, metadata
            FROM messages
            WHERE session_id = ? AND content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (session_id, f'%{query}%', limit))
        else:
            cursor.execute('''
            SELECT id, role, content, timestamp, metadata
            FROM messages
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (f'%{query}%', limit))
        
        return [{
            'id': row[0],
            'role': row[1],
            'content': row[2],
            'timestamp': row[3],
            'metadata': json.loads(row[4]) if row[4] else {},
            'score': 1.0
        } for row in cursor.fetchall()]

    def search_messages(self, query: str, session_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Search for messages similar to the query."""
        if self.use_vector:
            return self._search_vector_db(query, session_id, limit)
        return self._search_sql_db(query, session_id, limit)
    
    def clear_session_messages(self, session_id: str) -> None:
        """
        Clear all messages for a session.
        
        Args:
            session_id: ID of the session to clear messages for
        """
        if not self.use_vector:
            # Clear from SQL database
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            self.conn.commit()
        
        # For vector database, we'd need a more complex approach to remove
        # vectors by session ID. In a production system, we'd track which
        # vectors belong to which session and rebuild the index or use a
        # database that supports removing vectors.
        # For simplicity, we'll just note that this is a limitation.
        else:
            logger.warning("Clearing messages from vector database not fully supported. Session data may still contain old vectors.")
    
    def save_snapshot(self, snapshot_id: str, snapshot_data: Dict) -> None:
        """
        Save a memory snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            snapshot_data: Snapshot data to save
        """
        if self.use_vector:
            # For vector DB, just add to the lookup for now
            if 'snapshots' not in self.lookup:
                self.lookup['snapshots'] = {}
                
            self.lookup['snapshots'][snapshot_id] = snapshot_data
            self._save_vector_lookup()
        else:
            # Save to SQL database
            cursor = self.conn.cursor()
            session_id = snapshot_data['session_id']
            created_at = snapshot_data['created_at']
            
            cursor.execute('''
            INSERT OR REPLACE INTO snapshots (id, session_id, data, created_at)
            VALUES (?, ?, ?, ?)
            ''', (snapshot_id, session_id, json.dumps(snapshot_data), created_at))
            
            self.conn.commit()
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        """
        Get a memory snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to get
            
        Returns:
            Snapshot data or None if not found
        """
        if self.use_vector:
            # Get from vector database lookup
            if 'snapshots' not in self.lookup:
                return None
                
            return self.lookup['snapshots'].get(snapshot_id)
        else:
            # Get from SQL database
            cursor = self.conn.cursor()
            cursor.execute('SELECT data FROM snapshots WHERE id = ?', (snapshot_id,))
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            
            return None
    
    def close(self) -> None:
        """Close database connections and save any pending data."""
        if hasattr(self, 'conn'):
            self.conn.close()
        
        if self.use_vector:
            self._save_vector_lookup()
            
            # Save FAISS index
            try:
                import faiss
                index_path = os.path.join(self.data_dir, 'vector', FAISS_INDEX_FILENAME)
                faiss.write_index(self.index, index_path)
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}")
        
        logger.info("Closed database connections") 