from typing import List, Dict, Any
from abc import ABC, abstractmethod
import time
import json
from .exceptions import RateLimitExceededException

class Middleware(ABC):
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process the request before it reaches the model."""
        pass

    @abstractmethod
    async def process_response(self, response: str) -> str:
        """Process the response before it reaches the user."""
        pass

class MiddlewareManager:
    def __init__(self):
        self._middleware: List[Middleware] = []

    def add_middleware(self, middleware: Middleware) -> None:
        """Add a middleware to the chain."""
        self._middleware.append(middleware)

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process the request through all middleware."""
        for middleware in self._middleware:
            request = await middleware.process_request(request)
        return request

    async def process_response(self, response: str) -> str:
        """Process the response through all middleware."""
        for middleware in reversed(self._middleware):
            response = await middleware.process_response(response)
        return response

class LoggingMiddleware(Middleware):
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Processing request: {request}")
        return request

    async def process_response(self, response: str) -> str:
        print(f"Processing response: {response}")
        return response

class RateLimitMiddleware(Middleware):
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        current_time = time.time()
        
        # Remove old requests outside the window
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.window_seconds]
        
        # Check if we've exceeded the rate limit
        if len(self.requests) >= self.max_requests:
            raise RateLimitExceededException(f"Maximum {self.max_requests} requests per {self.window_seconds} seconds.")
        
        # Add current request
        self.requests.append(current_time)
        return request

    async def process_response(self, response: str) -> str:
        return response

class CacheMiddleware(Middleware):
    def __init__(self, cache: Dict[str, str], ttl_seconds: int = 3600):
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.timestamps = {}

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Create a cache key from the request
        cache_key = self._create_cache_key(request)
        
        # Check if we have a cached response
        if cache_key in self.cache:
            # Check if the cache entry is still valid
            if self._is_cache_valid(cache_key):
                # Add cache hit flag to request
                request['cache_hit'] = True
                return request
        
        request['cache_hit'] = False
        return request

    async def process_response(self, response: str) -> str:
        # If this was a cache hit, return the cached response
        if hasattr(self, '_current_request') and self._current_request.get('cache_hit'):
            return self.cache[self._create_cache_key(self._current_request)]
        
        # Store the response in cache
        cache_key = self._create_cache_key(self._current_request)
        self.cache[cache_key] = response
        self.timestamps[cache_key] = time.time()
        
        return response

    def _create_cache_key(self, request: Dict[str, Any]) -> str:
        """Create a unique cache key from the request."""
        # Create a deterministic string representation of the request
        key_parts = []
        for k, v in sorted(request.items()):
            if isinstance(v, (dict, list)):
                key_parts.append(f"{k}:{json.dumps(v)}")
            else:
                key_parts.append(f"{k}:{str(v)}")
        return "|".join(key_parts)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self.timestamps:
            return False
        return time.time() - self.timestamps[cache_key] < self.ttl_seconds 