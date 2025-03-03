import docker

class DockerManager:
    def __init__(self):
        self.client = docker.from_env()

    def start_vllm_container(self, model_name, tensor_parallel_size, gpu_memory_utilization, port):
        container = self.client.containers.run(
            image="vllm/vllm-openai:latest",
            command=[
                "--model", model_name,
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--port", str(port),
            ],
            ports={f"{port}/tcp": port},
            runtime="nvidia",
            shm_size="4gb",
            detach=True,
            environment={
                "NCCL_DEBUG": "INFO",
            },
        )
        return container.id

    def stop_vllm_container(self, container_id):
        container = self.client.containers.get(container_id)
        container.stop()
        container.remove()

    def list_vllm_containers(self):
        containers = self.client.containers.list(all=True)
        return [
            {
                "id": container.id,
                "name": container.name,
                "status": container.status,
                "ports": container.ports,
            }
            for container in containers
            if "vllm" in container.name
        ]