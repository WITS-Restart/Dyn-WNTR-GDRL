import enum
import os
import threading
import time

DEFAULT_LOGGER_KEY = "default_logger"

class LogType(enum.Enum):
    ENV = "env.txt"
    AGENT = "agent.txt"
    EPISODE = "episode.txt"
    LOSS = "losses_dqn.csv"
    TD_ERRORS = "td_errors_dqn.csv"
    Q_VALUES = "q_values_dqn.csv"
    GRAD_NORM = "grad_norm_dqn.csv"
    TRAINING = "training_log.txt"

class Logger:
    MESSAGE_LIMIT = 50
    _registry = {}
    _registry_lock = threading.Lock()

    def __init__(self, name: str=None, env=True, agent=True, episode=True, loss=True, td_errors=True, q_values=True, grad_norm=True, training=True):
        self.name = name if name else str(time.time())
        self.cache = {
            LogType.ENV: [] if env else None,
            LogType.AGENT: [] if agent else None,
            LogType.EPISODE: [] if episode else None,
            LogType.LOSS: [] if loss else None,
            LogType.TD_ERRORS: [] if td_errors else None,
            LogType.Q_VALUES: [] if q_values else None,
            LogType.GRAD_NORM: [] if grad_norm else None,
            LogType.TRAINING: [] if training else None,
        }
        self.current_episode = 0

        # use an absolute log directory so later flushes don't depend on CWD
        self.log_dir = os.path.abspath(os.path.join("logs", self.name))
        if os.path.exists(self.log_dir):
            os.system(f"rm -r {self.log_dir}")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "single_episodes"), exist_ok=True)

    @classmethod
    def get(cls, name: str = None, env=True, agent=True, episode=True, loss=True, td_errors=True, q_values=True, grad_norm=True, training=True):
        """
        Factory accessor: returns an existing logger with this name or creates one.
        Usage: Logger.get("run1") or Logger.get("run1", env=False)
        """
        key = name or "default"
        with cls._registry_lock:
            if key in cls._registry:
                return cls._registry[key]
            logger = Logger(name, env, agent, episode, loss, td_errors, q_values, grad_norm, training)
            cls._registry[key] = logger
            return logger
    
    def reset_episode(self, episode_id: int):
        self.flush(LogType.EPISODE)
        self.current_episode = episode_id

    def log(self, message: str, msg_type: LogType):
        if msg_type not in self.cache:
            return
        self.cache[msg_type].append(message)
        if len(self.cache[msg_type]) >= Logger.MESSAGE_LIMIT:
            self.flush(msg_type)
        elif msg_type == LogType.TRAINING:
            self.flush(msg_type)

    def flush(self, msg_type: LogType):
        if self.cache[msg_type]:
            if msg_type == LogType.EPISODE:
                path = os.path.join(self.log_dir, "single_episodes", f"episode_{self.current_episode}.txt")
                with open(path, "a") as f:
                    for msg in self.cache[msg_type]:
                        f.write(msg + "\n")
            else:
                path = os.path.join(self.log_dir, msg_type.value)
                with open(path, "a") as f:
                    for msg in self.cache[msg_type]:
                        f.write(msg + "\n")
            self.cache[msg_type] = []
