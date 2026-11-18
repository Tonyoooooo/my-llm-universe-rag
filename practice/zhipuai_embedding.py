from typing import List
from langchain_core.embeddings import Embeddings

class ZhipuAIEmbeddings(Embeddings):
    def __init__(self):
        import httpx
        from zhipuai import ZhipuAI
        # trust_env=False：跳过系统代理直连智谱；timeout=30：防止单批死等
        self.client = ZhipuAI(http_client=httpx.Client(trust_env=False, timeout=30.0))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        for i in range(0, len(texts), 64):
            print(f"正在向量化：{i} ~ {min(i+64, len(texts))} / {len(texts)}")
            embeddings = self.client.embeddings.create(
                model="embedding-3",
                input=texts[i:i+64]
            )
            result.extend([item.embedding for item in embeddings.data])
        return result

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]