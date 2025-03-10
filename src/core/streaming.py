from typing import AsyncGenerator, Dict, List
from .interfaces import StreamInterface
import asyncio
import json

class StreamManager(StreamInterface):
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size

    async def stream_response(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream the response from the model."""
        # This is a placeholder implementation
        # In a real implementation, this would connect to the model's streaming API
        response = "This is a streamed response. "
        for i in range(0, len(response), self.chunk_size):
            chunk = response[i:i + self.chunk_size]
            yield chunk
            await asyncio.sleep(0.1)  # Simulate network delay

class StreamProcessor:
    def __init__(self):
        self.stream_manager = StreamManager()

    async def process_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Process a stream of responses."""
        async for chunk in self.stream_manager.stream_response(messages, **kwargs):
            # Process each chunk (e.g., apply formatting, add metadata)
            processed_chunk = self._process_chunk(chunk)
            yield processed_chunk

    def _process_chunk(self, chunk: str) -> str:
        """Process a single chunk of the stream."""
        # Add metadata or formatting to the chunk
        return json.dumps({
            "content": chunk,
            "timestamp": asyncio.get_event_loop().time(),
            "type": "text"
        })

class StreamHandler:
    def __init__(self, processor: StreamProcessor):
        self.processor = processor

    async def handle_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Handle a stream of messages."""
        try:
            async for chunk in self.processor.process_stream(messages, **kwargs):
                yield chunk
        except Exception as e:
            # Handle streaming errors
            yield json.dumps({
                "error": str(e),
                "type": "error"
            }) 