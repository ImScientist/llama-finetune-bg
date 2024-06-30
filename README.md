# llama-finetune-bg

Apply the Stanford LLaMA fine-tuning using low-rank adaptation (LoRA) and the Alpaca dataset translated into Bulgarian.

### Local development

- The base model and the training datasets are fetched from [Hugging Face Hub](https://huggingface.co/models).
  Besides a token to fetch content from the hub you have to submit an extra form to fetch
  the [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) base model.


- To prevent fetching the LLaMA from Hugging Face Hub multiple times we map the cache directory in the image to a local
  directory `CACHE_DIR`. The Lora weights can be used to reconstruct the weights that can be added to the original model (stored in the cache dir) in order to obtain the fine-tuned model. They are stored in `MODEL_DIR`.

```shell
mkdir model notebooks ${HOME}/cache

HF_TOKEN=<hugging face token>
CACHE_DIR=${HOME}/cache
MODEL_DIR=$(pwd)/model
NOTEBOOKS_DIR=$(pwd)/notebooks

docker build -t llama_tune -f Dockerfile .


# Start a jupyter notebook server
docker run -it --rm \
  --runtime=nvidia --gpus=all --name=llama_tune -p 8888:8888 \
  -v "$(pwd)/src:/workspace/src" \
  -v "${CACHE_DIR}:/root/.cache" \
  -v "${MODEL_DIR}:/workspace/model" \
  -v "${NOTEBOOKS_DIR}:/workspace/notebooks" \
  -e HF_TOKEN=$HF_TOKEN \
  llama_tune:latest


# Fine-tune the model using instructions translated in Bulgarian
# Use the `target-repo` argument if you want to push the LORA weights to a HuggingFace model repo 
docker run -it --rm \
  --runtime=nvidia --gpus=all --name=llama_tune -p 8888:8888 \
  -v "$(pwd)/src:/workspace/src" \
  -v "${CACHE_DIR}:/root/.cache" \
  -v "${MODEL_DIR}:/workspace/model" \
  -v "${NOTEBOOKS_DIR}:/workspace/notebooks" \
  -e HF_TOKEN=$HF_TOKEN \
  llama_tune:latest -- python src/main.py train \
    --dataset=ImScientist/alpaca-cleaned-bg


# Inference with the model
# `repo-fine-tuned` can point to a local directory or to the HuggingFace repo where the LORA weights are pushed
docker run -it --rm \
  --runtime=nvidia --gpus=all --name=llama_tune -p 8888:8888 \
  -v "$(pwd)/src:/workspace/src" \
  -v "${CACHE_DIR}:/root/.cache" \
  -v "${MODEL_DIR}:/workspace/model" \
  -v "${NOTEBOOKS_DIR}:/workspace/notebooks" \
  -e HF_TOKEN=$HF_TOKEN \
  llama_tune:latest -- python src/main.py infer \
    --usr-instruction="Коя е най-голямата страна в света?" \
    --repo-fine-tuned=ImScientist/Llama-2-7b-hf-finetuned \
    --lang=en
```
