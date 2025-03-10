from typing import Dict, Any, Type
from .interfaces import ModelInterface, MemoryInterface, DatabaseInterface, StreamInterface, MultiModalInterface

class Container:
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}

    def register(self, service_name: str, implementation: Any) -> None:
        """Register a service instance."""
        self._services[service_name] = implementation

    def register_factory(self, service_name: str, factory: callable) -> None:
        """Register a service factory."""
        self._factories[service_name] = factory

    def get(self, service_name: str) -> Any:
        """Get a service instance."""
        if service_name in self._services:
            return self._services[service_name]
        elif service_name in self._factories:
            return self._factories[service_name]()
        raise KeyError(f"Service {service_name} not found")

    def register_model(self, model_name: str, model_class: Type[ModelInterface]) -> None:
        """Register a model implementation."""
        self.register(f"model.{model_name}", model_class)

    def register_memory(self, memory_class: Type[MemoryInterface]) -> None:
        """Register a memory implementation."""
        self.register("memory", memory_class)

    def register_database(self, db_class: Type[DatabaseInterface]) -> None:
        """Register a database implementation."""
        self.register("database", db_class)

    def register_stream(self, stream_class: Type[StreamInterface]) -> None:
        """Register a streaming implementation."""
        self.register("stream", stream_class)

    def register_multimodal(self, multimodal_class: Type[MultiModalInterface]) -> None:
        """Register a multimodal implementation."""
        self.register("multimodal", multimodal_class)

# Global container instance
container = Container() 