import json
import logging
import os
from datetime import datetime

class StructuredLogger:
    def __init__(self, name="main", log_dir="logs", level=logging.INFO):
        self.name=name
        os.makedirs(log_dir,exist_ok=True)
        self.log_path=os.path.join(log_dir,f"{name}.log")

        self.logger=logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate=False

        if not self.logger.handlers:
            file_handler=logging.FileHandler(self.log_path)
            console_handler=logging.StreamHandler()

            formatter=logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    
    def _log(self,level,message,**metadata):
        record={
            "timestamp":datetime.utcnow().isoformat(),
            "level":logging.getLevelName(level),
            "message":message,
            "metadata":metadata
        }
        self.logger.log(level,json.dumps(record,ensure_ascii=False))

    def info(self,message,**metadata):
        self._log(logging.INFO,message,**metadata)

    def warning(self,message,**metadata):
        self._log(logging.WARNING,message,**metadata)

    def error(self,message,**metadata):
        self._log(logging.ERROR,message,**metadata)

    def critical(self,message,**metadata):
        self._log(logging.CRITICAL,message,**metadata)
