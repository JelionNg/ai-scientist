from typing import Dict, Any, List
from .base import BaseAgent

class SupervisorAgent(BaseAgent):
    def __init__(self, brain, memory, agents: List[BaseAgent] = None):
        """初始化 Supervisor 智能体
        
        Args:
            brain: LLM 接口
            memory: 向量存储接口
            agents: 其他智能体列表
        """
        super().__init__(brain, memory)
        self.agents = agents or []
        self.name = "supervisor"
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据
        
        Args:
            input_data: 输入数据字典
            
        Returns:
            Dict: 处理结果
        """
        try:
            # 1. 任务分解
            subtasks = await self._decompose_task(input_data)
            
            # 2. 分配并执行子任务
            results = {}
            for subtask in subtasks:
                agent = self._select_agent(subtask)
                if agent:
                    result = await agent.process(subtask)
                    results[agent.name] = result
                    
            # 3. 整合结果
            final_result = await self._integrate_results(results)
            
            return final_result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"处理失败: {str(e)}"
            }
    
    async def reflect(self) -> Dict[str, Any]:
        """反思处理结果
        
        Returns:
            Dict: 反思结果
        """
        try:
            reflection = {
                "agent_status": self._check_agents_status(),
                "performance_metrics": self._calculate_performance_metrics(),
                "improvement_suggestions": await self._generate_improvements()
            }
            return reflection
        except Exception as e:
            return {
                "status": "error",
                "message": f"反思失败: {str(e)}"
            }
    
    async def _decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将任务分解为子任务
        
        Args:
            task: 原始任务
            
        Returns:
            List[Dict]: 子任务列表
        """
        try:
            # 使用 LLM 分解任务
            prompt = f"""
            请将以下研究任务分解为子任务:
            {task['content']}
            
            每个子任务应包含:
            1. 任务类型
            2. 具体目标
            3. 所需资源
            """
            
            response = await self.brain.think(prompt)
            
            # 解析响应为子任务列表
            subtasks = self._parse_subtasks(response)
            
            return subtasks
        except Exception as e:
            raise Exception(f"任务分解失败: {str(e)}")
    
    def _select_agent(self, subtask: Dict[str, Any]) -> BaseAgent:
        """为子任务选择合适的智能体
        
        Args:
            subtask: 子任务信息
            
        Returns:
            BaseAgent: 选中的智能体
        """
        # 根据任务类型选择智能体
        task_type = subtask.get('type', '')
        
        for agent in self.agents:
            if agent.can_handle(task_type):
                return agent
                
        return None
    
    async def _integrate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """整合所有子任务的结果
        
        Args:
            results: 子任务结果字典
            
        Returns:
            Dict: 整合后的结果
        """
        try:
            # 使用 LLM 整合结果
            prompt = f"""
            请整合以下研究子任务的结果:
            {results}
            
            请提供:
            1. 总体结论
            2. 关键发现
            3. 后续建议
            """
            
            response = await self.brain.think(prompt)
            
            return {
                "status": "success",
                "integrated_result": response,
                "sub_results": results
            }
        except Exception as e:
            raise Exception(f"结果整合失败: {str(e)}")
    
    def _check_agents_status(self) -> Dict[str, Any]:
        """检查所有智能体的状态"""
        status = {}
        for agent in self.agents:
            status[agent.name] = {
                "active": True,
                "last_task_success": agent.last_task_success if hasattr(agent, 'last_task_success') else None
            }
        return status
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        return {
            "task_success_rate": 0.0,  # 待实现
            "average_response_time": 0.0,  # 待实现
            "agent_utilization": 0.0  # 待实现
        }
    
    async def _generate_improvements(self) -> List[str]:
        """生成改进建议"""
        return [
            "增加更多专业领域的智能体",
            "优化任务分解策略",
            "改进结果整合方法"
        ]
    
    def _parse_subtasks(self, llm_response: str) -> List[Dict[str, Any]]:
        """解析 LLM 响应为结构化的子任务列表"""
        # 这里需要根据实际的 LLM 输出格式进行解析
        # 当前返回一个示例结构
        return [
            {
                "type": "research",
                "goal": "文献综述",
                "resources": ["papers", "databases"]
            }
        ]
