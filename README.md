# llama-finetune-bg

Apply the Stanford LLaMA fine-tuning using low-rank adaptation (LoRA) and the Alpaca dataset translated into Bulgarian.

### Local development

- The base model and the training datasets are fetched from [Hugging Face Hub](https://huggingface.co/models).
Besides a token to fetch content from the hub you have to submit an extra form to fetch
the [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) base model.


- To prevent fetching the LLaMA from Hugging Face Hub multiple times we map the cache directory in the image to a local
cache.

```shell
HF_TOKEN=<hugging face token>
MODEL_CACHE=${HOME}/cache
NOTEBOOKS_DIR=$(pwd)/notebooks

docker build -t llama_tune -f Dockerfile .

docker run -it --rm \
  --runtime=nvidia --gpus=all --name=llama_tune -p 8888:8888 \
  -v "$(pwd)/src:/workspace/src" \
  -v "${NOTEBOOKS_DIR}:/workspace/notebooks" \
  -v "${MODEL_CACHE}:/root/.cache" \
  -e HF_TOKEN=HF_TOKEN \
  llama_tune:latest
```
