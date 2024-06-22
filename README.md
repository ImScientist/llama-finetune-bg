# llama-finetune-bg
Apply the Stanford LLaMA fine-tuning using low-rank adaptation (LoRA) and the Alpaca dataset translated into Bulgarian


### Local development

To prevent loading the LLaMA multiple times we map the cache directory in the image with the local cache.

```shell
MODEL_CACHE=${HOME}/cache
NOTEBOOKS_DIR=$(pwd)/notebooks

docker build -t llama_tune -f Dockerfile .

docker run -it --rm \
  --runtime=nvidia --gpus=all --name=llama_tune -p 8888:8888 \
  -v "$(pwd)/src:/workspace/src" \
  -v "${NOTEBOOKS_DIR}:/workspace/notebooks" \
  -v "${MODEL_CACHE}:/root/.cache" \
  llama_tune:latest
```
