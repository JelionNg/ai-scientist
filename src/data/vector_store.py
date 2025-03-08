from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
import os
import ssl
import urllib3
from loguru import logger
import requests
import time

class VectorStore:
    def __init__(self, config: Dict[str, Any] = None):
        """初始化向量存储"""
        self.config = config or {}
        self.device = self.config.get("device", "cpu")
        self.embedding_model = None
        self.db = None
        
        # 设置重试次数
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 初始化embedding模型
                model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                
                # 计算本地模型路径
                local_model_path = os.path.join("./models/embeddings", model_name.split("/")[-1])
                
                # 初始化embedding模型
                self._initialize_embedding_model(model_name, local_model_path)
                
                # 初始化Chroma数据库 - 使用新的配置方式
                import chromadb
                
                # 确保数据目录存在
                persist_directory = "./data/chroma"
                os.makedirs(persist_directory, exist_ok=True)
                
                # 使用新的客户端初始化方式
                self.db = chromadb.PersistentClient(
                    path=persist_directory
                )
                
                # 创建或获取集合
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                
                # 使用SentenceTransformerEmbeddingFunction
                embedding_function = SentenceTransformerEmbeddingFunction(
                    model_name=local_model_path if os.path.exists(local_model_path) else model_name,
                    device=self.device
                )
                
                self.collection = self.db.get_or_create_collection(
                    name="research_data",
                    embedding_function=embedding_function
                )
                
                logger.info(f"向量存储初始化完成，使用模型: {model_name}")
                break
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"向量存储初始化尝试 {retry_count}/{max_retries} 失败: {str(e)}")
                
                if retry_count < max_retries:
                    # 等待一段时间后重试
                    wait_time = 2 * retry_count
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    # 尝试使用备用方案
                    logger.warning("所有尝试都失败，使用备用方案...")
                    self._initialize_fallback()
    
    def _disable_ssl_verification(self):
        """禁用SSL验证"""
        logger.warning("禁用SSL验证 - 仅用于开发环境")
        
        # 禁用SSL验证警告
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 创建自定义的SSL上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # 配置requests和urllib3使用自定义SSL上下文
        session = requests.Session()
        session.verify = False
        
        # 设置环境变量以禁用HuggingFace的SSL验证
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
    
    def _initialize_embedding_model(self, model_name: str, local_model_path: str):
        """初始化embedding模型"""
        try:
            # 检查本地模型是否存在
            if os.path.exists(local_model_path):
                logger.info(f"尝试从本地加载模型: {local_model_path}")
                
                # 设置环境变量，禁用huggingface的远程检查
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                
                # 从本地加载模型
                try:
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer(local_model_path, device=self.device)
                    logger.info(f"成功从本地加载模型: {local_model_path}")
                    return
                except Exception as e:
                    logger.warning(f"从本地加载模型失败: {str(e)}")
                    # 如果本地加载失败，重置环境变量，尝试从远程加载
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
            
            # 如果本地模型不存在或加载失败，尝试从远程加载
            logger.info(f"尝试从远程加载模型: {model_name}")
            
            # 设置代理（如果需要）
            # os.environ["HTTPS_PROXY"] = "http://your-proxy:port"
            
            # 设置SSL验证选项
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 设置huggingface的下载选项
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
            
            # 下载模型到本地
            try:
                snapshot_download(
                    repo_id=model_name,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False,
                    ssl_context=ssl_context
                )
                logger.info(f"成功下载模型到本地: {local_model_path}")
            except Exception as e:
                logger.warning(f"下载模型失败: {str(e)}")
            
            # 加载模型
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
            
            # 保存模型到本地
            try:
                self.embedding_model.save(local_model_path)
                logger.info(f"成功保存模型到本地: {local_model_path}")
            except Exception as e:
                logger.warning(f"保存模型到本地失败: {str(e)}")
            
        except Exception as e:
            logger.error(f"初始化embedding模型失败: {str(e)}")
            raise
    
    def _initialize_fallback(self):
        """初始化备用方案，当所有尝试都失败时使用"""
        try:
            # 使用简单的余弦相似度作为备用
            logger.info("初始化备用向量存储...")
            
            # 定义简单的embedding函数
            def simple_embedding(text):
                """简单的词袋模型embedding"""
                from collections import Counter
                import numpy as np
                import re
                
                # 简单的分词
                words = re.findall(r'\w+', text.lower())
                # 创建词袋
                bag = Counter(words)
                # 转换为向量
                return np.array(list(bag.values()), dtype=float)
            
            # 设置embedding模型
            class SimpleEmbedder:
                def encode(self, texts):
                    if isinstance(texts, str):
                        return simple_embedding(texts)
                    return [simple_embedding(text) for text in texts]
            
            self.embedding_model = SimpleEmbedder()
            
            # 初始化简单的内存数据库
            class SimpleDB:
                def __init__(self):
                    self.data = []
                    
                def add(self, texts, metadatas=None, ids=None):
                    if metadatas is None:
                        metadatas = [{} for _ in texts]
                    if ids is None:
                        ids = [str(i) for i in range(len(texts))]
                    
                    for text, metadata, id in zip(texts, metadatas, ids):
                        self.data.append({
                            "id": id,
                            "text": text,
                            "metadata": metadata,
                            "embedding": simple_embedding(text)
                        })
                        
                def search(self, query, limit=5):
                    import numpy as np
                    from scipy.spatial.distance import cosine
                    
                    query_embedding = simple_embedding(query)
                    
                    # 计算相似度
                    results = []
                    for item in self.data:
                        try:
                            similarity = 1 - cosine(query_embedding, item["embedding"])
                            results.append((item, similarity))
                        except:
                            continue
                    
                    # 排序并返回结果
                    results.sort(key=lambda x: x[1], reverse=True)
                    return [
                        {
                            "id": item["id"],
                            "text": item["text"],
                            "metadata": item["metadata"],
                            "distance": 1 - similarity
                        }
                        for item, similarity in results[:limit]
                    ]
            
            # 设置集合
            self.collection = SimpleDB()
            logger.info("备用向量存储初始化完成")
            
        except Exception as e:
            logger.error(f"初始化备用方案失败: {str(e)}")
            raise
    
    async def store_embeddings(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """存储文本及其embedding到向量数据库"""
        try:
            if not texts:
                logger.warning("没有文本需要存储")
                return
            
            # 确保metadatas与texts长度一致
            if metadatas is None:
                metadatas = [{} for _ in texts]
            elif len(metadatas) != len(texts):
                logger.warning(f"metadatas长度 ({len(metadatas)}) 与texts长度 ({len(texts)}) 不一致，将使用空元数据")
                metadatas = [{} for _ in texts]
            
            # 生成唯一ID
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
            
            # 存储到向量数据库
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"成功存储 {len(texts)} 条文本")
            
        except Exception as e:
            logger.error(f"存储embedding失败: {str(e)}")
            raise
    
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索与查询最相似的文本"""
        try:
            # 执行搜索
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            # 处理结果
            processed_results = []
            
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0]  # 第一个查询的结果
                metadatas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if 'distances' in results and results['distances'] else [0] * len(documents)
                ids = results['ids'][0] if 'ids' in results and results['ids'] else [""] * len(documents)
                
                for doc, meta, dist, id in zip(documents, metadatas, distances, ids):
                    processed_results.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                        "id": id
                    })
                
            return processed_results
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            count = self.db._collection.count()
            return {
                "count": count,
                "model": self.model_name,
                "collection": self.config['collection_name']
            }
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {str(e)}")
            return {
                "count": 0,
                "model": self.model_name,
                "collection": self.config['collection_name'],
                "error": str(e)
            }
