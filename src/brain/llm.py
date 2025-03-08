from typing import Dict, Any, Optional, AsyncGenerator
import os
from openai import AsyncOpenAI
from loguru import logger
import asyncio
from ..agents.types import TaskType

class ModelProvider:
    """模型提供商配置"""
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    # 后续可以添加其他模型提供商

class Brain:
    """与各种 LLM API 交互的大脑类"""
    
    # 模型提供商配置
    PROVIDER_CONFIGS = {
        ModelProvider.DEEPSEEK: {
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
            "stream_required": False,
        },
        ModelProvider.QWEN: {
            "api_key_env": "DASHSCOPE_API_KEY",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "default_model": "qwen-plus",
            "stream_required": True,
        },
        ModelProvider.OPENAI: {
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-3.5-turbo",
            "stream_required": False,
        },
        ModelProvider.ANTHROPIC: {
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": "https://api.anthropic.com",
            "default_model": "claude-2",
            "stream_required": False,
            "is_anthropic": True,  # 标记使用Anthropic专用客户端
        },
        ModelProvider.GEMINI: {
            "api_key_env": "GOOGLE_API_KEY",
            "base_url": "https://generativelanguage.googleapis.com",
            "default_model": "gemini-pro",
            "stream_required": False,
            "is_gemini": True,  # 标记使用Google专用客户端
        }
    }
    
    # 添加模型能力配置
    MODEL_CAPABILITIES = {
        "deepseek-chat": {
            "max_tokens": 8192,
            "supports_functions": True,
            "supports_vision": False,
            "typical_temperature": 0.7
        },
        "qwen-plus": {
            "max_tokens": 6144,
            "supports_functions": True,
            "supports_vision": False,
            "typical_temperature": 0.8
        },
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "supports_functions": True,
            "supports_vision": False,
            "typical_temperature": 0.7
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """初始化大脑
        
        Args:
            config: 配置字典，包含模型参数
        """
        self.config = config
        self.provider = config.get("provider", ModelProvider.QWEN)  # 默认使用通义千问
        
        # 获取提供商配置
        provider_config = self.PROVIDER_CONFIGS.get(self.provider)
        if not provider_config:
            raise ValueError(f"不支持的模型提供商: {self.provider}")
            
        # 获取 API Key
        self.api_key = os.environ.get(provider_config["api_key_env"])
        if not self.api_key:
            raise ValueError(f"未设置 {provider_config['api_key_env']} 环境变量")
        
        # 初始化异步 OpenAI 客户端
        if provider_config.get("is_anthropic"):
            # 使用Anthropic专用客户端
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=self.api_key)
        # elif provider_config.get("is_gemini"):
        #     # 使用Google专用客户端
        #     import google.generativeai as genai
        #     genai.configure(api_key=self.api_key)
        #     self.client = genai
        else:
            # 使用OpenAI兼容客户端
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=provider_config["base_url"]
            )
        
        # 设置默认模型（如果配置中未指定）
        if "model" not in self.config:
            self.config["model"] = provider_config["default_model"]
            
        # 初始化模型能力
        self.capabilities = {
            "max_tokens": 4096,
            "supports_functions": False,
            "supports_vision": False,
            "typical_temperature": 0.7
        }
        
        # 如果模型在预定义能力中，直接使用
        model_name = self.config["model"]
        if model_name in self.MODEL_CAPABILITIES:
            self.capabilities = self.MODEL_CAPABILITIES[model_name]
            logger.info(f"使用预定义的模型能力配置: {model_name}")
        
        self.stream_required = provider_config["stream_required"]
        self.stream_callback = None  # 添加直接回调属性
        logger.info(f"初始化完成，使用模型: {self.config['model']}")
            
        # 检测模型能力
        asyncio.create_task(self.detect_model_capabilities())
            
    async def think(self, prompt: str, task_type=None, callback=None) -> AsyncGenerator[str, None]:
        """思考问题并生成回答，支持流式输出和停止功能"""
        # 重置停止标志
        self.should_stop = False
        
        # 优化参数
        params = self._optimize_params_for_task(task_type, prompt)
        
        # 提取system_prompt（如果存在）
        system_prompt = params.pop("system_prompt", None)
        
        # 创建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # 检查是否应该停止
            if self.should_stop:
                logger.info("Brain: 思考被停止")
                return
            
            # 创建请求任务
            request_task = asyncio.create_task(
                self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=messages,
                    stream=True,
                    **params
                )
            )
            
            # 保存当前请求以便可以在需要时取消它
            self._current_request = request_task
            
            # 等待响应
            response = await request_task
            
            # 检查是否应该停止
            if self.should_stop:
                logger.info("Brain: 思考被停止")
                return
            
            # 处理流式响应
            async for chunk in self._handle_stream_response(response):
                # 检查是否应该停止
                if self.should_stop:
                    logger.info("Brain: 思考被停止")
                    return
                
                # 如果有回调函数，调用它
                if callback:
                    await callback(chunk)
                
                # 产生块
                yield chunk
            
        except asyncio.CancelledError:
            logger.info("Brain: 思考被取消")
            raise
        except Exception as e:
            logger.error(f"Brain: 思考时出错: {str(e)}")
            raise
        finally:
            # 清理
            self._current_request = None

    def _optimize_params_for_task(self, task_type, prompt):
        """根据任务类型优化参数"""
        # 基本参数
        params = {
            "temperature": 0.7,
            "max_tokens": 2000,
        }
        
        # 根据任务类型调整参数
        if task_type == TaskType.GENERATE_HYPOTHESIS:
            params["temperature"] = 0.8
            params["max_tokens"] = 3000
            
            # 添加system_prompt（将在think方法中正确处理）
            params["system_prompt"] = """你是一个专业的科学研究助手，擅长生成研究假设。
                                            请根据提供的研究问题和背景信息，生成3-5个合理的研究假设。
                                            每个假设应包含：假设描述、理论依据、验证方法和影响因素。
                                            保持客观、科学的语言风格，确保假设具有可验证性和创新性。"""
            
        elif task_type == TaskType.EVALUATE_HYPOTHESIS:
            params["temperature"] = 0.3
            params["max_tokens"] = 2000
            
            # 添加system_prompt（将在think方法中正确处理）
            params["system_prompt"] = """你是一个专业的科学研究评估专家，擅长评估研究假设的质量。
                                            请根据提供的研究假设，评估其科学性、创新性、可行性和潜在影响。
                                            提供详细的评分和改进建议，帮助研究者优化假设。"""
            
        elif task_type == TaskType.DESIGN_EXPERIMENT:
            params["temperature"] = 0.5
            params["max_tokens"] = 3000
            
            # 添加system_prompt（将在think方法中正确处理）
            params["system_prompt"] = """你是一个专业的实验设计专家，擅长设计科学实验。
                                            请根据提供的研究假设，设计一个详细的实验方案来验证假设。
                                            包括实验材料、步骤、数据收集方法、分析方法和预期结果。
                                            确保实验设计严谨、可行，并能有效验证假设。"""
        
        # 根据模型提供商调整参数
        if self.provider == ModelProvider.ANTHROPIC:
            # Anthropic Claude 特定参数
            if "max_tokens" in params:
                params["max_tokens_to_sample"] = params.pop("max_tokens")
            
        elif self.provider == ModelProvider.GEMINI:
            # Google Gemini 特定参数
            pass
        
        return params
            
    async def _handle_stream_response(self, response) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        # 保存当前流以便可以在需要时关闭它
        self._current_stream = response
        
        accumulated_text = ""
        
        try:
            async for chunk in response:
                # 检查是否应该停止
                if self.should_stop:
                    logger.info("Brain: 流式响应处理被停止")
                    return
                    
                # 根据不同的模型提供商提取内容
                content = ""
                
                if self.provider == ModelProvider.ANTHROPIC:
                    # Anthropic Claude
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        content = chunk.delta.text or ""
                elif self.provider == ModelProvider.GEMINI:
                    # Google Gemini
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for candidate in chunk.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                for part in candidate.content.parts:
                                    content += part.text or ""
                else:
                    # OpenAI 和其他兼容格式
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            content = delta.content
                
                # 累积文本
                accumulated_text += content
                
                # 如果有内容，则产生
                if content:
                    yield content
                    
        except asyncio.CancelledError:
            logger.info("Brain: 流式响应被取消")
            raise
        except Exception as e:
            logger.error(f"Brain: 处理流式响应时出错: {str(e)}")
            raise
        finally:
            # 清理
            self._current_stream = None
            
    async def close(self):
        """关闭客户端连接"""
        # OpenAI 客户端会自动处理连接的关闭
        pass

    def get_model_name(self) -> str:
        """获取当前使用的模型名称"""
        return self.config["model"]

    async def detect_model_capabilities(self):
        """检测当前模型的能力"""
        model_name = self.config["model"]
        
        # 检查是否有预定义的能力配置
        if model_name in self.MODEL_CAPABILITIES:
            self.capabilities = self.MODEL_CAPABILITIES[model_name]
            logger.info(f"使用预定义的模型能力配置: {model_name}")
            return
            
        # 如果没有预定义配置，尝试通过API获取
        try:
            # 简单的测试提示
            test_prompt = "你能处理多少最大token? 你支持函数调用吗? 你支持视觉输入吗?"
            
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.0,
                max_tokens=100
            )
            
            # 解析响应，提取能力信息
            content = response.choices[0].message.content
            
            # 基于响应内容推断能力
            self.capabilities = {
                "max_tokens": 4096,  # 默认值
                "supports_functions": "函数" in content and "支持" in content,
                "supports_vision": "视觉" in content and "支持" in content,
                "typical_temperature": 0.7  # 默认值
            }
            
            logger.info(f"动态检测的模型能力: {self.capabilities}")
            
        except Exception as e:
            logger.warning(f"模型能力检测失败: {str(e)}，使用默认配置")
            self.capabilities = {
                "max_tokens": 4096,
                "supports_functions": False,
                "supports_vision": False,
                "typical_temperature": 0.7
            }

    def stop_generation(self):
        """停止当前生成过程"""
        logger.info("Brain: 停止生成过程")
        self.should_stop = True
        
        # 如果有正在进行的请求，尝试取消它
        if hasattr(self, '_current_request') and self._current_request is not None:
            try:
                if not self._current_request.done():
                    logger.info("Brain: 取消当前请求")
                    self._current_request.cancel()
                self._current_request = None
            except Exception as e:
                logger.error(f"Brain: 取消请求时出错: {str(e)}")
        
        # 如果有正在进行的流式响应，尝试关闭它
        if hasattr(self, '_current_stream') and self._current_stream is not None:
            try:
                logger.info("Brain: 关闭当前流")
                # 对于不同的客户端可能需要不同的关闭方法
                if hasattr(self._current_stream, 'close'):
                    self._current_stream.close()
                elif hasattr(self._current_stream, 'aclose'):
                    asyncio.create_task(self._current_stream.aclose())
                self._current_stream = None
            except Exception as e:
                logger.error(f"Brain: 关闭流时出错: {str(e)}")

    def reset_state(self):
        """重置状态，准备新的生成过程"""
        self.should_stop = False
        self._current_request = None
