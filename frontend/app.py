import gradio as gr
import requests

BACKEND_URL = "http://backend:8000"

def start_vllm(model_name, tensor_parallel_size, gpu_memory_utilization, port):
    response = requests.post(
        f"{BACKEND_URL}/start_vllm",
        json={
            "model_name": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "port": port,
        },
    )
    return response.json()

def stop_vllm(container_id):
    response = requests.post(f"{BACKEND_URL}/stop_vllm/{container_id}")
    return response.json()

def list_vllm_containers():
    response = requests.get(f"{BACKEND_URL}/list_vllm_containers")
    return response.json()

with gr.Blocks() as demo:
    gr.Markdown("# vLLM Manager")
    with gr.Row():
        with gr.Column():
            model_name = gr.Textbox(label="Model Name", value="Qwen/Qwen2.5-1.5B-Instruct")
            tensor_parallel_size = gr.Slider(label="Tensor Parallel Size", minimum=1, maximum=8, step=1, value=2)
            gpu_memory_utilization = gr.Slider(label="GPU Memory Utilization", minimum=0.1, maximum=1.0, step=0.05, value=0.95)
            port = gr.Number(label="Port", value=1370)
            start_button = gr.Button("Start vLLM")
        with gr.Column():
            container_id = gr.Textbox(label="Container ID")
            stop_button = gr.Button("Stop vLLM")
    with gr.Row():
        containers = gr.JSON(label="Running Containers")
        list_button = gr.Button("Refresh Containers")

    start_button.click(
        start_vllm,
        inputs=[model_name, tensor_parallel_size, gpu_memory_utilization, port],
        outputs=container_id,
    )
    stop_button.click(stop_vllm, inputs=container_id, outputs=container_id)
    list_button.click(list_vllm_containers, outputs=containers)

demo.launch(server_name="0.0.0.0", server_port=7860)