from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from docker_utils import DockerManager

app = FastAPI()
docker_manager = DockerManager()

class VLLMConfig(BaseModel):
    model_name: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    port: int

@app.post("/start_vllm")
def start_vllm(config: VLLMConfig):
    try:
        container_id = docker_manager.start_vllm_container(
            model_name=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            port=config.port,
        )
        return {"container_id": container_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_vllm/{container_id}")
def stop_vllm(container_id: str):
    try:
        docker_manager.stop_vllm_container(container_id)
        return {"message": f"Container {container_id} stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_vllm_containers")
def list_vllm_containers():
    return docker_manager.list_vllm_containers()