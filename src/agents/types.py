from enum import Enum
from typing import Dict, Any, List

class AgentType(Enum):
    """智能体类型枚举"""
    GENERATOR = "generator"
    REFLECTOR = "reflector"
    RANKER = "ranker"
    EVOLVER = "evolver"
    META_REVIEWER = "meta_reviewer"

class TaskType(Enum):
    """任务类型枚举"""
    GENERATE_HYPOTHESIS = "generate_hypothesis"
    EVALUATE_HYPOTHESIS = "evaluate_hypothesis"
    RANK_HYPOTHESIS = "rank_hypothesis"
    EVOLVE_HYPOTHESIS = "evolve_hypothesis"
    META_REVIEW = "meta_review"

class ResearchStage(Enum):
    """研究阶段枚举"""
    INITIAL = "initial"
    HYPOTHESIS = "hypothesis"
    EVALUATION = "evaluation"
    RANKING = "ranking"
    EVOLUTION = "evolution"
    REVIEW = "review"
    COMPLETE = "complete"

class Message:
    """智能体间通信的消息类"""
    def __init__(
        self,
        sender: str,
        receiver: str,
        content: Dict[str, Any],
        msg_type: str
    ):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.msg_type = msg_type
        self.timestamp = None  # 将在发送时设置
