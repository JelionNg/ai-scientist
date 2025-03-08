import os
import asyncio
import yaml
from pathlib import Path
from src.brain.llm import Brain
from src.data.vector_store import VectorStore
from src.web.app import WebUI
from src.supervisor import Supervisor
from dotenv import load_dotenv
import warnings
import logging
from loguru import logger
import sys

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 忽略特定的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 配置日志
logger.remove()  # 移除默认的处理器
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)
logger.add(
    "logs/ai_scientist.log",  # 添加文件日志
    rotation="500 MB",  # 日志文件大小超过500MB时轮转
    retention="10 days",  # 保留10天的日志
    level="DEBUG"
)

def load_config():
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def setup_environment():
    """设置环境变量"""
    current_dir = Path(__file__).parent.absolute()
    env_path = current_dir / '.env'
    
    if not env_path.exists():
        raise FileNotFoundError(
            f"找不到 .env 文件! 请在 {current_dir} 目录下创建 .env 文件，"
            "并添加 DASHSCOPE_API_KEY=your_api_key_here"  # 更新为通义千问的环境变量
        )
    
    load_dotenv(env_path)
    
    # 验证 API 密钥
    api_key = os.environ.get('DASHSCOPE_API_KEY')  # 更新为通义千问的环境变量
    if not api_key:
        raise ValueError(
            "环境变量 DASHSCOPE_API_KEY 未设置!\n"  # 更新错误信息
            "请在 .env 文件中添加：\n"
            "DASHSCOPE_API_KEY=your_api_key_here\n"
            "API 密钥获取方法：\n"
            "1. 访问通义千问开发者平台\n"
            "2. 注册/登录账号\n"
            "3. 在控制台创建 API 密钥"
        )
    
    logger.info("已成功加载 API 配置")

async def main():
    try:
        # 1. 设置环境
        setup_environment()
        
        # 2. 加载配置
        config = load_config()
        logger.info("已加载配置文件")
        
        # 3. 初始化组件
        brain = Brain(config['llm'])
        memory = VectorStore(config['vector_store'])
        logger.info("已初始化核心组件")
        
        # 4. 初始化 Supervisor
        supervisor = Supervisor(config, brain, memory)
        logger.info("已初始化 Supervisor")
        
        # 5. 启动 Web 界面
        ui = WebUI(supervisor)
        app = ui.build()
        logger.info("正在启动 Web 界面...")
        app.launch(server_name="localhost", server_port=7860)
        
    except Exception as e:
        logger.error(f"程序启动失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    # 运行主程序
    asyncio.run(main())
