
import os
import httpx


class RemoteAISessionEval:
    def __init__(self):
        self.url = os.environ["AI_WORKER_URL"].rstrip("/")
        self.api_key = os.environ["AI_WORKER_KEY"]
        self.client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=3.0)
        )

    def is_session_related(self,meta_data: dict,topic: str) -> bool:
        response = self.client.post(
            f"{self.url}/classify",
            headers={"X-AI-Key": self.api_key},
            json={
                "topic": topic,
                "metadata": meta_data
            }
        )

        response.raise_for_status()
        result = response.json()

        if type(result.get("is_related")) is not bool:
            raise ValueError("Invalid AI worker response")

        return result["is_related"]

    def AI_meta_data_eval(
        self,
        meta_data: dict,
        topic: str
    ) -> bool:
        return self.is_session_related(meta_data, topic)

    def close(self):
        self.client.close()
