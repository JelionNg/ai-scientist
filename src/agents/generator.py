from typing import Dict, Any, List, AsyncGenerator
from .base import BaseAgent
from .types import TaskType, Message
from datetime import datetime
from loguru import logger

class GeneratorAgent(BaseAgent):
    def __init__(self, brain, memory):
        super().__init__(brain, memory)
        self.name = "generator"
        self.task_types = [TaskType.GENERATE_HYPOTHESIS]
        self.should_stop = False
        self.current_text = []
        
    async def process(self, input_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """生成研究假设，支持流式输出和停止功能"""
        max_retries = 3
        retry_count = 0
        
        # 重置状态
        self.should_stop = False
        self.current_text = []
        
        while retry_count < max_retries:
            try:
                # 检查是否应该停止
                if self.should_stop:
                    yield {"status": "stopped", "message": "生成已停止"}
                    return
                    
                # 验证输入数据
                if not input_data.get("content", {}).get("question"):
                    raise ValueError("缺少研究问题")
                
                # 构建提示
                prompt = self._build_hypothesis_prompt(input_data)
                
                # 用于累积完整响应
                full_response = ""
                
                # 调用LLM生成假设
                async for chunk in self.brain.think(prompt, TaskType.GENERATE_HYPOTHESIS):
                    # 检查是否应该停止
                    if self.should_stop:
                        yield {"status": "stopped", "message": "生成已停止"}
                        return
                        
                    # 更新完整响应
                    full_response += chunk
                    
                    # 流式输出
                    yield {
                        "status": "generating",
                        "chunk": chunk
                    }
                
                # 检查是否应该停止
                if self.should_stop:
                    yield {"status": "stopped", "message": "生成已停止"}
                    return
                    
                # 解析假设
                try:
                    # 检查是否应该停止
                    if self.should_stop:
                        yield {"status": "stopped", "message": "生成已停止"}
                        return
                        
                    hypotheses = self._parse_hypotheses(full_response)
                    
                    # 检查是否应该停止
                    if self.should_stop:
                        yield {"status": "stopped", "message": "生成已停止"}
                        return
                        
                    # 存储假设
                    if not self.should_stop:
                        await self._store_hypotheses(hypotheses)
                    
                    # 返回成功结果 - 移除评估部分
                    yield {
                        "status": "success",
                        "hypotheses": hypotheses
                    }
                    return
                    
                except Exception as e:
                    logger.error(f"解析假设时出错: {str(e)}")
                    # 如果解析失败但有完整响应，返回原始文本
                    if full_response:
                        yield {
                            "status": "success",
                            "raw_text": full_response,
                            "message": f"解析假设失败，但返回原始文本。错误: {str(e)}"
                        }
                        return
                    else:
                        raise e
                        
            except Exception as e:
                retry_count += 1
                logger.error(f"生成假设失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count >= max_retries:
                    yield {
                        "status": "error",
                        "message": f"多次尝试后生成假设失败: {str(e)}"
                    }
                    return
    
    def _build_hypothesis_prompt(self, input_data: Dict[str, Any]) -> str:
        """构建生成假设的提示"""
        question = input_data["content"]["question"]
        background = input_data["content"].get("background", "")
        
        prompt = f"""# 研究问题
                        {question}

                        # 背景信息
                        {background}

                        # 任务
                        请根据上述研究问题和背景信息，生成3-5个合理的研究假设。每个假设必须包含以下四个部分：
                        1. 假设描述：清晰陈述假设内容
                        2. 理论依据：解释支持该假设的理论基础
                        3. 验证方法：描述如何验证该假设
                        4. 影响因素：列出可能影响假设验证的关键因素

                        # 输出格式
                        请按照以下格式输出：

                        假设1：[直接写出假设描述]
                        理论依据：[直接写出理论依据]
                        验证方法：[直接写出验证方法]
                        影响因素：[直接写出影响因素]

                        假设2：[直接写出假设描述]
                        理论依据：[直接写出理论依据]
                        验证方法：[直接写出验证方法]
                        影响因素：[直接写出影响因素]

                        [继续生成剩余假设...]

                        注意：
                        1. 直接生成假设，不要询问更多信息
                        2. 保持格式统一
                        3. 确保每个假设都完整包含四个部分
                        4. 如果信息不足，基于已有信息做合理推测和扩展
                        """
        return prompt
    
    def _parse_hypotheses(self, response: str) -> List[Dict[str, Any]]:
        """解析生成的假设文本"""
        # 检查是否应该停止
        if self.should_stop:
            logger.info("解析假设被停止")
            return []
            
        logger.info("开始解析假设...")
        
        # 提取内容部分
        content = self._extract_content(response)
        
        # 检查是否应该停止
        if self.should_stop:
            logger.info("解析假设被停止")
            return []
        
        # 初始化结果列表
        hypotheses = []
        
        # 尝试解析假设
        try:
            # 按假设分割文本
            hypothesis_blocks = content.split("假设")
            
            # 跳过第一个元素（通常是空的或者是介绍性文本）
            for i, block in enumerate(hypothesis_blocks[1:], 1):
                # 检查是否应该停止
                if self.should_stop:
                    logger.info("解析假设被停止")
                    return hypotheses
                    
                # 清理文本
                block = block.strip()
                if not block:
                    continue
                    
                # 初始化假设字典
                hypothesis = {
                    "id": f"h{i}",
                    "created_at": datetime.now().isoformat(),
                    "content": {
                        "description": "",
                        "theoretical_basis": "",
                        "verification_method": "",
                        "influencing_factors": ""
                    }
                }
                
                # 提取假设编号和内容
                if block.startswith(str(i) + "：") or block.startswith(str(i) + ":"):
                    block = block[2:].strip()
                elif block.startswith(str(i) + ".") or block.startswith(str(i) + "、"):
                    block = block[2:].strip()
                    
                # 分割各个部分
                parts = block.split("\n")
                
                # 提取假设描述
                if parts and parts[0]:
                    hypothesis["content"]["description"] = parts[0].strip()
                
                # 提取其他部分
                current_key = None
                for part in parts[1:]:
                    part = part.strip()
                    if not part:
                        continue
                        
                    # 检查是否应该停止
                    if self.should_stop:
                        logger.info("解析假设被停止")
                        return hypotheses
                        
                    # 检查是否是新的部分
                    if part.startswith("理论依据：") or part.startswith("理论依据:"):
                        current_key = "theoretical_basis"
                        hypothesis["content"][current_key] = part[5:].strip()
                    elif part.startswith("验证方法：") or part.startswith("验证方法:"):
                        current_key = "verification_method"
                        hypothesis["content"][current_key] = part[5:].strip()
                    elif part.startswith("影响因素：") or part.startswith("影响因素:"):
                        current_key = "influencing_factors"
                        hypothesis["content"][current_key] = part[5:].strip()
                    elif current_key:
                        # 继续添加到当前部分
                        hypothesis["content"][current_key] += " " + part
                
                # 添加到结果列表
                hypotheses.append(hypothesis)
            
            logger.info(f"解析完成，共找到 {len(hypotheses)} 个假设")
            
            # 打印假设内容以便调试
            for h in hypotheses:
                logger.debug(f"假设 {h['id']}:")
                logger.debug(f"  描述: {h['content']['description']}")
                logger.debug(f"  理论依据: {h['content']['theoretical_basis']}")
                logger.debug(f"  验证方法: {h['content']['verification_method']}")
                logger.debug(f"  影响因素: {h['content']['influencing_factors']}")
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"解析假设时出错: {str(e)}")
            raise
    
    def _extract_content(self, text: str) -> str:
        """提取文本中的内容部分"""
        # 简单清理文本
        text = text.strip()
        
        # 移除可能的Markdown标记
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # 跳过Markdown标题行
            if line.strip().startswith("#"):
                continue
            # 跳过任务说明、注意事项等
            if "任务" in line or "注意" in line or "输出格式" in line:
                continue
            cleaned_lines.append(line)
            
        return "\n".join(cleaned_lines)
    
    async def _store_hypotheses(self, hypotheses: List[Dict[str, Any]]):
        """存储生成的假设到向量数据库"""
        # 检查是否应该停止
        if self.should_stop:
            logger.info("存储假设被停止")
            return
            
        # 检查假设是否为空
        if not hypotheses:
            logger.warning("没有假设可存储")
            return
            
        try:
            # 准备存储的文本
            texts = []
            metadatas = []
            
            for h in hypotheses:
                # 检查是否应该停止
                if self.should_stop:
                    logger.info("存储假设被停止")
                    return
                    
                # 构建文本和元数据
                text = f"假设: {h['content'].get('description', '')}\n"
                text += f"理论依据: {h['content'].get('theoretical_basis', '')}\n"
                text += f"验证方法: {h['content'].get('verification_method', '')}\n"
                text += f"影响因素: {h['content'].get('influencing_factors', '')}"
                
                metadata = {
                    "id": h["id"],
                    "type": "hypothesis",
                    "created_at": h["created_at"]
                }
                
                texts.append(text)
                metadatas.append(metadata)
            
            # 检查是否应该停止
            if self.should_stop:
                logger.info("存储假设被停止")
                return
                
            # 存储到向量数据库
            await self.memory.store_embeddings(texts, metadatas)
            
        except Exception as e:
            logger.error(f"存储假设时出错: {str(e)}")
            # 不抛出异常，让流程继续
    
    async def _get_recent_hypotheses(self) -> List[Dict[str, Any]]:
        """获取最近生成的假设"""
        try:
            # 从向量数据库查询
            results = await self.memory.search("type:hypothesis", limit=10)
            
            # 处理结果
            hypotheses = []
            for result in results:
                # 提取元数据
                metadata = result.get("metadata", {})
                
                # 提取文本内容
                text = result.get("text", "")
                
                # 解析文本
                content = {}
                for line in text.split("\n"):
                    if line.startswith("假设:"):
                        content["description"] = line[3:].strip()
                    elif line.startswith("理论依据:"):
                        content["theoretical_basis"] = line[5:].strip()
                    elif line.startswith("验证方法:"):
                        content["verification_method"] = line[5:].strip()
                    elif line.startswith("影响因素:"):
                        content["influencing_factors"] = line[5:].strip()
                
                # 构建假设对象
                hypothesis = {
                    "id": metadata.get("id", "unknown"),
                    "created_at": metadata.get("created_at", ""),
                    "content": content
                }
                
                hypotheses.append(hypothesis)
                
            return hypotheses
            
        except Exception as e:
            logger.error(f"获取假设失败: {str(e)}")
            return []  # 返回空列表而不是抛出异常，使流程更健壮
    
    def stop_generation(self):
        """停止当前生成过程"""
        logger.info("Generator: 停止生成过程")
        self.should_stop = True
        
        # 如果brain有stop_generation方法，调用它
        if hasattr(self.brain, 'stop_generation'):
            self.brain.stop_generation()
    
    def reset_state(self):
        """重置状态，准备新的生成过程"""
        logger.info("Generator: 重置状态")
        self.should_stop = False
        self.current_text = []
        
        # 如果brain有reset_state方法，调用它
        if hasattr(self.brain, 'reset_state'):
            self.brain.reset_state()

    async def reflect(self) -> Dict[str, Any]:
        """反思当前状态和生成的假设"""
        try:
            # 获取最近生成的假设
            hypotheses = await self._get_recent_hypotheses()
            
            # 如果没有假设，返回空结果
            if not hypotheses:
                return {
                    "status": "no_hypotheses",
                    "message": "没有找到最近生成的假设"
                }
            
            # 构建反思结果
            reflection = {
                "status": "success",
                "hypotheses_count": len(hypotheses),
                "latest_hypothesis": hypotheses[0] if hypotheses else None,
                "timestamp": datetime.now().isoformat()
            }
            
            return reflection
            
        except Exception as e:
            logger.error(f"反思失败: {str(e)}")
            return {
                "status": "error",
                "message": f"反思过程中出错: {str(e)}"
            }
