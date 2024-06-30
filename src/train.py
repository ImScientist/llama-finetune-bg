import os
import logging
from datasets import load_dataset
from transformers import (
    Trainer,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training)

from typing import Literal
import prompter

logger = logging.getLogger()

HF_TOKEN = os.environ['HF_TOKEN']
CUTOFF_LEN = 256


def tokenize(prompt, tokenizer):
    """ Generate input_ids, attention_mask, labels """

    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point, tokenizer):
    """ Generate and tokenize prompt """

    full_prompt = prompter.generate_prompt(
        usr_instruction=data_point["instruction"],
        usr_input=data_point["input"],
        label=data_point["output"])

    tokenized_full_prompt = tokenize(full_prompt, tokenizer)

    return tokenized_full_prompt


def tr_va_split(ds, val_size, tokenizer):
    """ Train/validation split """

    if val_size > 0:
        args = dict(test_size=val_size, shuffle=True, seed=42)
        ds_tr_va = ds.train_test_split(**args)

        ds_tr = (ds_tr_va["train"]
                 .shuffle()
                 .map(lambda x: generate_and_tokenize_prompt(x, tokenizer)))
        ds_va = (ds_tr_va["test"]
                 .shuffle()
                 .map(lambda x: generate_and_tokenize_prompt(x, tokenizer)))
    else:
        ds_tr = (ds
                 .shuffle()
                 .map(lambda x: generate_and_tokenize_prompt(x, tokenizer)))
        ds_va = None

    return ds_tr, ds_va


def load_model(
        base_model: str = 'meta-llama/Llama-2-7b-hf',
        device_map: str = 'auto'
):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.
    The default padding token is unset as there is no padding token in
    the original model.
    """

    tokenizer = LlamaTokenizer.from_pretrained(
        base_model,
        add_eos_token=True,
        add_bos_token=True,
        padding_side='left',  # allow batched inference
        token=HF_TOKEN)

    tokenizer.pad_token_id = 2  # change from None to 2, which refers to EOS

    config_quantization = BitsAndBytesConfig(load_in_8bit=True)

    config_lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=config_quantization,
        device_map=device_map,
        # torch_dtype=torch.float16,  # this decreases the model size
        token=HF_TOKEN)

    model = prepare_model_for_kbit_training(model)

    peft_model = get_peft_model(model, config_lora)

    return tokenizer, peft_model


def train(
        dataset_name: Literal['yahma/alpaca-cleaned', 'ImScientist/alpaca-cleaned-bg'],
        target_repo_name: str | None = None
):
    """ Fine-tune llama model """

    base_model = 'meta-llama/Llama-2-7b-hf'
    device_map = 'auto'

    val_set_size = 200  # 2_000
    batch_size = 128
    micro_batch_size = 4

    model_dir = './model'
    model_checkpoints_dir = './model_checkpoints'

    resume_from_checkpoint = None

    tokenizer, peft_model = load_model(
        base_model=base_model,
        device_map=device_map)

    ds = load_dataset(dataset_name, split="train[:500]")
    # ds = load_dataset(dataset_name)['train']

    ds_tr, ds_va = tr_va_split(
        ds=ds,
        val_size=val_set_size,
        tokenizer=tokenizer)

    peft_model.print_trainable_parameters()

    trainer = Trainer(
        model=peft_model,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        args=TrainingArguments(
            output_dir=model_checkpoints_dir,

            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=batch_size // micro_batch_size,
            warmup_steps=100,
            num_train_epochs=1,  # 3,
            learning_rate=3e-4,
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            logging_steps=10,

            fp16=True,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            save_total_limit=3,

            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=None,
            group_by_length=False,  # faster, but produces an odd training loss curve
            report_to=None,
            run_name=None),

        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True))

    train_results = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info(f'Training results:\n{train_results}')

    logger.info('Save the lora weights locally')
    peft_model.save_pretrained(model_dir, token=HF_TOKEN)

    if target_repo_name:
        logger.info(f'Push the LORA weights to the hugging-face '
                    f'repo: {target_repo_name}')
        peft_model.push_to_hub(target_repo_name)
