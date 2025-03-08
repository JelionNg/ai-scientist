import gradio as gr
from typing import Dict, Any
from src.agents.types import ResearchStage, AgentType
from loguru import logger
import asyncio

class WebUI:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.current_text = []
        self.current_state = None  # 用于存储当前状态
        self.total_tokens = 0
        self.expected_tokens = 1000  # 预估token数
        self.should_stop = False  # 添加停止标志
        self.is_generating = False  # 生成状态标志
        self._generator_instance = None  # 存储当前生成器实例
        self.generation_id = 0  # 添加生成ID来跟踪每次生成
        
    def format_hypothesis(self, hypothesis):
        """格式化假设为Markdown格式"""
        # 获取假设内容
        content = hypothesis.get("content", {})
        
        # 提取各个部分
        description = content.get("description", "未提供")
        theoretical_basis = content.get("theoretical_basis", "未提供")
        verification_method = content.get("verification_method", "未提供")
        influencing_factors = content.get("influencing_factors", "未提供")
        
        # 构建Markdown格式的假设
        markdown = f"""#### 🔬 研究假设 {hypothesis.get('id', 'unknown')}

📌 假设描述
{description}

📚 理论依据
{theoretical_basis}

🔍 验证方法
{verification_method}

⚡ 影响因素
{influencing_factors}

---
"""
        return markdown

    def format_final_report(self, result, question, background):
        """格式化最终研究报告"""
        report = f"""# 📑 研究报告

## 📌 研究概述

**研究问题**
{question}

**研究背景**
{background or '未提供背景信息'}

## 🔬 研究假设

"""
        # 添加假设
        for hypothesis in result.get("hypotheses", []):
            report += self.format_hypothesis(hypothesis)
            
        # 添加评估结果
        if result.get("evaluation"):
            report += "\n## 📊 评估结果\n\n"
            report += str(result["evaluation"])
        
        # 添加实验设计（如果有）
        if result.get("experiments"):
            report += "\n## 🧪 实验设计\n\n"
            report += str(result["experiments"])
        
        # 添加结论和建议
        report += "\n## 💡 结论与建议\n\n"
        report += "1. 建议优先验证最具创新性和可行性的假设\n"
        report += "2. 实验设计应注意控制变量\n"
        report += "3. 建议进行小规模预实验验证方案可行性\n"
        
        return report

    async def handle_chunk(self, data: Dict[str, Any]):
        """直接处理chunk的回调方法"""
        if data.get("status") == "generating":
            if "chunk" in data:
                self.current_text.append(data["chunk"])
                current_content = "".join(self.current_text)
                
                self.current_state = self._get_update_state(
                    progress="### 📊 研究进度\n正在生成假设...",
                    hypothesis_status="🔄 正在生成...",
                    hypothesis_output=f"```\n{current_content}\n```"
                )
                return self.current_state
                
            elif "hypotheses" in data:
                hypotheses = data["hypotheses"]
                if hypotheses:
                    hypotheses_text = "## 🧬 正在生成研究假设...\n\n"
                    for hypothesis in hypotheses:
                        hypotheses_text += self.format_hypothesis(hypothesis)
                    
                    self.current_state = self._get_update_state(
                        progress="### 📊 研究进度\n正在生成假设...",
                        hypothesis_status="🔄 正在生成...",
                        hypothesis_output=hypotheses_text
                    )
                    return self.current_state

    async def process_research(self, question: str, background: str):
        """处理研究请求，支持流式输出和停止功能"""
        try:
            # 重置状态
            self.current_text = []
            self.should_stop = False  # 重置停止标志
            self._generator_instance = None  # 重置生成器实例
            
            # 初始状态
            initial_state = "### ⏳ 正在准备研究..."
            hypothesis_state = "### 🔄 正在准备生成假设..."
            evaluation_state = "### ⏳ 等待假设生成完成..."
            
            yield initial_state, hypothesis_state, evaluation_state
            
            # 准备输入数据
            input_data = {
                "type": "research_question",
                "content": {
                    "question": question,
                    "background": background
                }
            }
            
            # 处理研究流程
            generator = self.supervisor.agents[AgentType.GENERATOR]
            async for update in generator.process(input_data):
                # 检查是否应该停止生成
                if self.should_stop:
                    final_message = "### ⚠️ 生成已停止\n\n" + "".join(self.current_text)
                    yield final_message, hypothesis_state + "\n\n**已停止**", evaluation_state
                    return
                    
                if update["status"] == "generating":
                    # 更新当前文本
                    if "chunk" in update:
                        self.current_text.append(update["chunk"])
                        current_content = "".join(self.current_text)
                        
                        # 更新假设生成标签页
                        hypothesis_state = f"""### 🔄 正在生成假设...

{current_content}
```
"""
                        # 更新主输出
                        progress_state = self._get_update_state(
                            progress="### 📊 研究进度\n正在生成假设...",
                            hypothesis_status="🔄 正在生成...",
                            hypothesis_output=""
                        )
                        
                        yield progress_state, hypothesis_state, evaluation_state
                        
                elif update["status"] == "success":
                    # 生成最终报告
                    final_state = self._get_final_state(update, question, background)
                    
                    # 更新假设标签页
                    hypotheses_md = "### ✅ 生成的研究假设\n\n"
                    for hypothesis in update.get("hypotheses", []):
                        hypotheses_md += self.format_hypothesis(hypothesis)
                    
                    # 更新评估标签页
                    evaluation_md = "### 📊 假设评估\n\n"
                    if update.get("evaluation"):
                        evaluation_md += str(update["evaluation"])
                    else:
                        evaluation_md += "尚未进行评估。"
                    
                    yield final_state, hypotheses_md, evaluation_md
                    return
                    
                elif update["status"] == "error":
                    error_state = self._get_error_state(update.get("message", "未知错误"))
                    yield error_state, f"### ❌ 错误\n\n{update.get('message', '未知错误')}", "### ❌ 无法进行评估"
                    
        except Exception as e:
            error_message = self._get_error_state(str(e))
            yield error_message, f"### ❌ 错误\n\n{str(e)}", "### ❌ 无法进行评估"

    def build(self):
        """构建Gradio界面"""
        with gr.Blocks(theme=gr.themes.Soft(), title="AI科学家") as demo:
            gr.Markdown("# 🧪 AI科学家 - 智能研究助手")
            
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### 📝 研究信息")
                        question_input = gr.Textbox(
                            label="研究问题",
                            placeholder="请输入您的研究问题...",
                            lines=2
                        )
                        background_input = gr.Textbox(
                            label="研究背景",
                            placeholder="请提供相关背景信息...",
                            lines=5
                        )
                        
                        submit_btn = gr.Button("🚀 开始研究", variant="primary")
                
                with gr.Column(scale=3):
                    # 创建标签页组件并存储引用
                    tabs = gr.Tabs()
                    
                    # 调整标签页顺序，将研究结果放在最后
                    with tabs:
                        with gr.TabItem("🔬 假设生成", id="hypothesis_tab"):
                            with gr.Row():
                                hypothesis_output = gr.Markdown()
                                
                            with gr.Row():
                                # 默认隐藏停止按钮
                                stop_btn = gr.Button("🛑 停止生成", variant="stop", visible=False)
                                
                        with gr.TabItem("📈 评估验证", id="evaluation_tab"):
                            evaluation_output = gr.Markdown()
                            
                        with gr.TabItem("📊 研究结果", id="result_tab"):
                            output = gr.Markdown()
            
            # 设置停止生成的事件处理
            self.should_stop = False
            self.is_generating = False
            self.generation_id = 0  # 添加生成ID来跟踪每次生成
            
            def stop_generation():
                self.should_stop = True
                self.is_generating = False
                
                # 确保supervisor停止生成
                if hasattr(self.supervisor, 'stop_generation'):
                    logger.info("WebUI: 调用supervisor停止生成")
                    self.supervisor.stop_generation()
                
                # 如果有正在运行的生成器，尝试取消它
                if hasattr(self, '_generator_task') and self._generator_task is not None:
                    try:
                        if not self._generator_task.done():
                            logger.info("WebUI: 取消生成器任务")
                            self._generator_task.cancel()
                        self._generator_task = None
                    except Exception as e:
                        logger.error(f"WebUI: 取消生成器任务时出错: {str(e)}")
                
                # 隐藏停止按钮
                return gr.update(visible=False), "### ⚠️ 生成已停止\n\n您可以开始新的研究。"
            
            stop_btn.click(fn=stop_generation, outputs=[stop_btn, hypothesis_output])
            
            # 创建一个函数来切换到假设生成标签页并显示停止按钮
            def start_research():
                # 增加生成ID，确保每次都是新的生成过程
                self.generation_id += 1
                self.is_generating = True
                self.should_stop = False
                self.current_text = []  # 重置当前文本
                
                # 确保supervisor的状态也被重置
                if hasattr(self.supervisor, 'reset_state'):
                    self.supervisor.reset_state()
                    
                # 显示停止按钮并切换到假设生成标签页
                return gr.update(selected="hypothesis_tab"), gr.update(visible=True)
            
            # 提交按钮事件 - 首先切换标签页并显示停止按钮
            submit_btn.click(
                fn=start_research,
                inputs=None,
                outputs=[tabs, stop_btn]
            )
            
            # 然后处理各个输出
            submit_btn.click(
                fn=self.process_hypothesis_output,
                inputs=[question_input, background_input],
                outputs=[hypothesis_output, stop_btn]  # 添加stop_btn作为输出
            )
            
            submit_btn.click(
                fn=self.process_result_output,
                inputs=[question_input, background_input],
                outputs=output
            )
            
            submit_btn.click(
                fn=self.process_evaluation_output,
                inputs=[question_input, background_input],
                outputs=evaluation_output
            )
            
        return demo

    def _get_initial_state(self):
        """获取初始状态"""
        return (
            "### 📊 研究进度\n正在启动研究流程...",
            "🔄 准备开始...",
            "",
            "⏳ 等待中...",
            "",
            "⏳ 等待中...",
            "",
            "⏳ 等待中...",
            "",
            ""
        )

    def _get_update_state(self, progress, hypothesis_status, hypothesis_output):
        """获取更新状态"""
        return (
            progress,
            hypothesis_status,
            hypothesis_output,
            "⏳ 等待中...",
            "",
            "⏳ 等待中...",
            "",
            "⏳ 等待中...",
            "",
            ""
        )

    def _get_final_state(self, result, question, background):
        """获取最终状态"""
        hypotheses_text = "## 🧬 生成的研究假设\n\n"
        for hypothesis in result.get("hypotheses", []):
            hypotheses_text += self.format_hypothesis(hypothesis)
        
        evaluation_text = ""
        if result.get("evaluation"):
            evaluation_text = "## 📊 假设评估结果\n\n" + str(result["evaluation"])
        
        final_text = self.format_final_report(result, question, background)
        
        return (
            "### 📊 研究进度\n✅ 研究完成！",
            "✅ 假设生成完成",
            hypotheses_text,
            "✅ 评估完成" if evaluation_text else "⏳ 等待中...",
            evaluation_text,
            "⏳ 等待中...",
            "",
            "⏳ 等待中...",
            "",
            final_text
        )

    def _get_error_state(self, error_msg):
        """获取错误状态"""
        return (
            f"### 📊 研究进度\n❌ {error_msg}",
            f"❌ {error_msg}",
            "",
            "❌ 未开始",
            "",
            "❌ 未开始",
            "",
            "❌ 未开始",
            "",
            ""
        )

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
请严格按照以下格式输出，确保每个部分都在新行开始：

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
2. 保持格式统一，每个部分都必须在新行开始
3. 确保每个假设都完整包含四个部分
4. 如果信息不足，基于已有信息做合理推测和扩展
"""
        return prompt

    async def process_hypothesis_output(self, question: str, background: str):
        """处理假设标签页的内容，并控制停止按钮的显示"""
        try:
            # 记录当前生成ID
            current_generation_id = self.generation_id
            
            # 初始状态
            yield "### 🔄 正在准备生成假设...", gr.update(visible=True)
            
            # 准备输入数据
            input_data = {
                "type": "research_question",
                "content": {
                    "question": question,
                    "background": background
                }
            }
            
            # 重置当前文本
            self.current_text = []
            
            # 获取生成器
            generator = self.supervisor.agents[AgentType.GENERATOR]
            
            # 创建一个新的生成器实例，避免重用
            generator_process = generator.process(input_data)
            
            try:
                # 处理研究流程
                async for update in generator_process:
                    # 检查是否应该停止生成或生成ID是否变化
                    if self.should_stop or current_generation_id != self.generation_id:
                        logger.info("WebUI: 假设生成被停止")
                        yield "### ⚠️ 生成已停止\n\n您可以开始新的研究。", gr.update(visible=False)
                        return
                        
                    if update["status"] == "generating":
                        # 更新当前文本
                        if "chunk" in update:
                            self.current_text.append(update["chunk"])
                            current_content = "".join(self.current_text)
                            
                            # 预处理文本，确保格式正确
                            formatted_content = self._format_streaming_content(current_content)
                            
                            # 更新假设生成标签页，保持停止按钮可见
                            yield f"""### 🔄 正在生成假设...

```
{formatted_content}
```
""", gr.update(visible=True)
                        
                    elif update["status"] == "success":
                        # 检查是否应该停止生成或生成ID是否变化
                        if self.should_stop or current_generation_id != self.generation_id:
                            logger.info("WebUI: 假设生成被停止")
                            yield "### ⚠️ 生成已停止\n\n您可以开始新的研究。", gr.update(visible=False)
                            return
                            
                        # 更新假设标签页，隐藏停止按钮
                        hypotheses_md = "### ✅ 生成的研究假设\n\n"
                        
                        # 检查是否有假设
                        if "hypotheses" in update and update["hypotheses"]:
                            for hypothesis in update["hypotheses"]:
                                # 打印假设内容以便调试
                                logger.debug(f"处理假设 {hypothesis.get('id', 'unknown')}:")
                                logger.debug(f"  内容: {hypothesis.get('content', {})}")
                                
                                # 格式化假设
                                formatted = self.format_hypothesis(hypothesis)
                                hypotheses_md += formatted
                        elif "raw_text" in update:
                            # 如果解析失败但有原始文本，显示原始文本
                            hypotheses_md += f"""### ⚠️ 解析假设失败，显示原始文本

                                                ```
                                                {update["raw_text"]}
                                                ```

                                                {update.get("message", "")}
                                                """
                        else:
                            hypotheses_md += "未能生成有效假设。"
                        
                        self.is_generating = False
                        yield hypotheses_md, gr.update(visible=False)
                        return
                        
                    elif update["status"] == "stopped":
                        # 生成被停止
                        logger.info("WebUI: 收到停止状态")
                        yield "### ⚠️ 生成已停止\n\n您可以开始新的研究。", gr.update(visible=False)
                        return
                        
                    elif update["status"] == "error":
                        # 发生错误，隐藏停止按钮
                        self.is_generating = False
                        yield f"### ❌ 错误\n\n{update.get('message', '未知错误')}", gr.update(visible=False)
                        
            except asyncio.CancelledError:
                logger.info("WebUI: 假设生成被取消")
                yield "### ⚠️ 生成已停止\n\n您可以开始新的研究。", gr.update(visible=False)
                return
                    
        except Exception as e:
            logger.error(f"处理假设生成时出错: {str(e)}")
            self.is_generating = False
            yield f"### ❌ 错误\n\n生成假设时出错: {str(e)}", gr.update(visible=False)
        finally:
            # 清理
            self._generator_task = None

    async def process_result_output(self, question: str, background: str):
        """处理研究结果标签页的内容"""
        try:
            # 初始状态 - 显示一个简单的研究摘要
            yield f"""### 📑 研究摘要

                    **研究问题**：
                    {question}

                    **研究背景**：
                    {background}

                    **研究状态**：
                    正在生成假设，请查看假设生成标签页获取最新进展。
                    """
            
        except Exception as e:
            logger.error(f"处理研究结果时出错: {str(e)}")
            yield f"### ❌ 错误\n\n生成研究结果时出错: {str(e)}"

    async def process_evaluation_output(self, question: str, background: str):
        """处理评估标签页的内容"""
        try:
            # 初始状态 - 由于移除了评估功能，显示一个占位符
            yield "### 📊 假设评估\n\n评估功能将由专门的评估智能体处理，目前尚未实现。"
            
        except Exception as e:
            logger.error(f"处理评估时出错: {str(e)}")
            yield f"### ❌ 错误\n\n处理评估时出错: {str(e)}"

    def _format_streaming_content(self, content: str) -> str:
        """格式化流式生成的内容，确保文本格式正确"""
        # 替换多余的空格和缩进
        content = content.replace("                                    ", "")
        
        # 确保理论依据、验证方法和影响因素前有换行
        content = content.replace("理论依据：", "\n理论依据：")
        content = content.replace("理论依据:", "\n理论依据:")
        content = content.replace("验证方法：", "\n验证方法：")
        content = content.replace("验证方法:", "\n验证方法:")
        content = content.replace("影响因素：", "\n影响因素：")
        content = content.replace("影响因素:", "\n影响因素:")
        
        # 确保假设之间有换行
        content = content.replace("假设", "\n假设")
        
        # 移除开头的换行符
        if content.startswith("\n"):
            content = content[1:]
        
        return content
