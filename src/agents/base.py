from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    def __init__(self, brain, memory):
        self.brain = brain
        self.memory = memory
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据并返回结果"""
        pass
    
    @abstractmethod
    async def reflect(self) -> Dict[str, Any]:
        """对自身处理结果进行反思"""
        pass
