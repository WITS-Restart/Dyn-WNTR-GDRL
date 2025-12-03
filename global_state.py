import threading
from typing import Any, Callable, Dict, Mapping

class State:
    _state: Dict[str, Any] = {}
    _lock = threading.RLock()

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        with cls._lock:
            return cls._state.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        with cls._lock:
            cls._state[key] = value

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        with cls._lock:
            return dict(cls._state)
    
    def set_all(cls, new_state) -> None:
        with cls._lock:
            cls._state = dict(new_state)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._state.clear()