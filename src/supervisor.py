from typing import Dict, Any, List
from .agents.types import AgentType, TaskType, ResearchStage, Message
from .agents.generator import GeneratorAgent
# from .agents.evaluator import EvaluatorAgent
# from .agents.experimenter import ExperimenterAgent
# from .agents.reviewer import ReviewerAgent
from loguru import logger
import asyncio
from collections import deque
from datetime import datetime

class ResearchSession:
    """研究会话类，管理单个研究的完整状态"""
    
    def __init__(self, session_id: str, question: str, background: str):
        self.id = session_id
        self.question = question
        self.background = background
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.stage = ResearchStage.INITIAL
        self.status = "initialized"
        self.hypotheses = []
        self.evaluation = {}
        self.experiments = []
        self.literature = {}
        self.messages = []
        self.artifacts = {}
        
    def update(self, data: Dict[str, Any]):
        """更新会话状态"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        self.updated_at = datetime.now().isoformat()
        
    def add_message(self, role: str, content: str):
        """添加消息"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        self.updated_at = message["timestamp"]
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "question": self.question,
            "background": self.background,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stage": self.stage,
            "status": self.status,
            "hypotheses": self.hypotheses,
            "evaluation": self.evaluation,
            "experiments": self.experiments,
            "literature": self.literature,
            "messages": self.messages,
            "artifacts": self.artifacts
        }

class Supervisor:
    """负责协调多智能体研究过程的主管理器"""
    
    def __init__(self, config: Dict[str, Any], brain, memory):
        self.config = config
        self.brain = brain
        self.memory = memory
        
        # 初始化智能体
        self.agents = {
            AgentType.GENERATOR: GeneratorAgent(brain, memory),
            # AgentType.EVALUATOR: EvaluatorAgent(brain, memory),
            # AgentType.EXPERIMENTER: ExperimenterAgent(brain, memory),
            # AgentType.REVIEWER: ReviewerAgent(brain, memory)
        }
        
        self.current_stage = ResearchStage.INITIAL
        self.update_callback = None
        
        # 任务队列
        self.task_queue = deque()
        self.is_processing = False
        self.current_task = None
        self.task_results = {}
        
        # 会话管理
        self.sessions = {}
        
        self.current_text = []
        self.current_state = None  # 用于存储当前状态
        self._generator_instance = None  # 存储当前生成器实例
        self.should_stop = False  # 停止标志
        self.is_generating = False  # 生成状态标志
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理研究请求的主流程
        
        Args:
            input_data: 包含研究问题和背景的输入数据
            
        Returns:
            Dict: 包含研究结果的字典
        """
        try:
            logger.info(f"开始处理研究请求: {input_data}")
            
            # 初始化研究状态
            research_state = {
                "status": "processing",
                "research_question": input_data.get("content", {}).get("question", ""),
                "background": input_data.get("content", {}).get("background", ""),
                "hypotheses": [],
                "evaluation": {},
                "experiments": [],
                "literature": {},
                "stage": self.current_stage
            }
            
            # 1. 生成假设阶段
            self.current_stage = ResearchStage.HYPOTHESIS_GENERATION
            research_state["stage"] = self.current_stage
            
            generator = self.agents[AgentType.GENERATOR]
            async for update in generator.process(input_data):
                if update["status"] == "generating":
                    # 传递流式更新
                    if self.update_callback:
                        await self.update_callback(update)
                elif update["status"] == "success":
                    research_state["hypotheses"] = update.get("hypotheses", [])
            
            # 如果没有生成假设，则返回错误
            if not research_state["hypotheses"]:
                research_state["status"] = "error"
                research_state["message"] = "未能生成有效的研究假设"
                return research_state
                
            # 2. 评估假设阶段
            if self.config.get("enable_evaluation", True):
                self.current_stage = ResearchStage.HYPOTHESIS_EVALUATION
                research_state["stage"] = self.current_stage
                
                evaluation_input = {
                    "type": "evaluate_hypotheses",
                    "content": {
                        "question": research_state["research_question"],
                        "background": research_state["background"],
                        "hypotheses": research_state["hypotheses"]
                    }
                }
                
                evaluator = self.agents[AgentType.EVALUATOR]
                evaluation_result = await evaluator.process(evaluation_input)
                research_state["evaluation"] = evaluation_result.get("evaluation", {})
            
            # 3. 实验设计阶段
            if self.config.get("enable_experiment_design", False):
                self.current_stage = ResearchStage.EXPERIMENT_DESIGN
                research_state["stage"] = self.current_stage
                
                experiment_input = {
                    "type": "design_experiments",
                    "content": {
                        "question": research_state["research_question"],
                        "background": research_state["background"],
                        "hypotheses": research_state["hypotheses"],
                        "evaluation": research_state["evaluation"]
                    }
                }
                
                experimenter = self.agents[AgentType.EXPERIMENTER]
                experiment_result = await experimenter.process(experiment_input)
                research_state["experiments"] = experiment_result.get("experiments", [])
            
            # 4. 文献综述阶段
            if self.config.get("enable_literature_review", False):
                self.current_stage = ResearchStage.LITERATURE_REVIEW
                research_state["stage"] = self.current_stage
                
                review_input = {
                    "type": "review_literature",
                    "content": {
                        "question": research_state["research_question"],
                        "background": research_state["background"],
                        "hypotheses": research_state["hypotheses"]
                    }
                }
                
                reviewer = self.agents[AgentType.REVIEWER]
                review_result = await reviewer.process(review_input)
                research_state["literature"] = review_result.get("literature", {})
            
            # 完成研究流程
            self.current_stage = ResearchStage.COMPLETED
            research_state["stage"] = self.current_stage
            research_state["status"] = "success"
            
            logger.info(f"研究流程完成: {research_state}")
            return research_state
            
        except Exception as e:
            logger.error(f"研究流程出错: {str(e)}")
            return {
                "status": "error",
                "message": f"处理研究请求失败: {str(e)}",
                "stage": self.current_stage
            }

    def set_update_callback(self, callback):
        """设置更新回调函数"""
        self.update_callback = callback
        # 同时设置Brain的回调
        self.brain.stream_callback = callback

    async def add_task(self, task_id: str, task_type: str, input_data: Dict[str, Any]):
        """添加任务到队列"""
        task = {
            "id": task_id,
            "type": task_type,
            "input": input_data,
            "status": "queued",
            "created_at": datetime.now().isoformat()
        }
        
        self.task_queue.append(task)
        logger.info(f"任务已添加到队列: {task_id}, 类型: {task_type}")
        
        # 如果没有正在处理的任务，启动处理
        if not self.is_processing:
            asyncio.create_task(self._process_queue())
            
        return {"task_id": task_id, "status": "queued"}
        
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        # 检查当前任务
        if self.current_task and self.current_task["id"] == task_id:
            return {
                "task_id": task_id,
                "status": self.current_task["status"],
                "progress": self.current_task.get("progress", 0)
            }
            
        # 检查队列中的任务
        for task in self.task_queue:
            if task["id"] == task_id:
                return {
                    "task_id": task_id,
                    "status": "queued",
                    "position": list(self.task_queue).index(task) + 1
                }
                
        # 检查已完成的任务
        if task_id in self.task_results:
            return {
                "task_id": task_id,
                "status": "completed",
                "result": self.task_results[task_id]
            }
            
        return {"task_id": task_id, "status": "not_found"}
        
    async def _process_queue(self):
        """处理任务队列"""
        if self.is_processing:
            return
            
        self.is_processing = True
        
        try:
            while self.task_queue:
                # 获取下一个任务
                task = self.task_queue.popleft()
                self.current_task = task
                task["status"] = "processing"
                task["started_at"] = datetime.now().isoformat()
                
                logger.info(f"开始处理任务: {task['id']}")
                
                # 根据任务类型处理
                if task["type"] == "research_question":
                    result = await self.process(task["input"])
                else:
                    result = {"status": "error", "message": f"不支持的任务类型: {task['type']}"}
                    
                # 存储结果
                task["status"] = "completed" if result["status"] == "success" else "failed"
                task["completed_at"] = datetime.now().isoformat()
                self.task_results[task["id"]] = result
                
                logger.info(f"任务处理完成: {task['id']}, 状态: {task['status']}")
                
        except Exception as e:
            logger.error(f"处理任务队列时出错: {str(e)}")
            if self.current_task:
                self.current_task["status"] = "failed"
                self.current_task["error"] = str(e)
                
        finally:
            self.is_processing = False
            self.current_task = None
            
            # 如果队列中还有任务，继续处理
            if self.task_queue:
                asyncio.create_task(self._process_queue())

    def create_session(self, question: str, background: str) -> str:
        """创建新的研究会话"""
        session_id = f"{datetime.now().isoformat()}_{question}_{background}"
        self.sessions[session_id] = ResearchSession(session_id, question, background)
        return session_id

    def stop_generation(self):
        """停止当前生成过程"""
        logger.info("正在停止生成过程...")
        self.should_stop = True
        
        # 停止所有代理的处理
        for agent_type, agent in self.agents.items():
            if hasattr(agent, 'stop_generation'):
                logger.info(f"停止 {agent_type} 代理")
                agent.stop_generation()
        
        # 清空任务队列
        self.task_queue = deque()
        
        # 如果有正在运行的任务，尝试取消它
        if hasattr(self, '_current_task') and self._current_task is not None:
            try:
                if not self._current_task.done():
                    logger.info("取消当前任务")
                    self._current_task.cancel()
                self._current_task = None
            except Exception as e:
                logger.error(f"取消任务时出错: {str(e)}")
        
        # 如果有正在运行的处理器，尝试取消它
        if hasattr(self, '_processor_task') and self._processor_task is not None:
            try:
                if not self._processor_task.done():
                    logger.info("取消处理器任务")
                    self._processor_task.cancel()
                self._processor_task = None
            except Exception as e:
                logger.error(f"取消处理器任务时出错: {str(e)}")
        
        logger.info("生成过程已停止")

    def reset_state(self):
        """重置所有状态，准备新的生成过程"""
        logger.info("重置Supervisor状态...")
        self.should_stop = False
        self.task_queue = deque()
        self._current_task = None
        
        # 重置所有代理的状态
        for agent_type, agent in self.agents.items():
            if hasattr(agent, 'reset_state'):
                agent.reset_state()
        
        logger.info("Supervisor状态已重置")
