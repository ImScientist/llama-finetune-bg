import os
import torch
import logging
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList)
from peft import (
    PeftConfig,
    PeftModel)

import prompter
from typing import Literal

logger = logging.getLogger()

HF_TOKEN = os.environ['HF_TOKEN']
CUTOFF_LEN = 256


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops: list, tokenizer, encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False


def load_finetuned_model(repo_name, device_map='auto'):
    """ Load fine-tuned model and tokenizer """

    # When you wrap a base model with PeftModel, modifications are done in-place.
    config = PeftConfig.from_pretrained(repo_name)

    tokenizer = LlamaTokenizer.from_pretrained(
        config.base_model_name_or_path,
        token=HF_TOKEN)

    tokenizer.pad_token_id = 2

    model = LlamaForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map=device_map,
        token=HF_TOKEN)

    # Load the LoRA model
    peft_model = PeftModel.from_pretrained(model, repo_name)

    return tokenizer, peft_model


def inference(
        model,
        tokenizer,
        usr_instruction: str,
        usr_input: str = None,
        lang: Literal['en', 'de', 'bg'] = 'en'
) -> str:
    # Set stopping criteria
    stop_words = ["#"]
    stop_words_ids = [
        tokenizer(s,
                  return_tensors='pt',
                  add_special_tokens=False)['input_ids'].squeeze()
        for s in stop_words]

    stopping_criteria = StoppingCriteriaList([
        StoppingCriteriaSub(stops=stop_words_ids,
                            tokenizer=tokenizer)
    ])

    prompt = prompter.generate_prompt(
        usr_instruction=usr_instruction,
        usr_input=usr_input,
        lang=lang)

    prompt_tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors='pt',
        add_special_tokens=False
    ).to('cuda')

    res = model.generate(
        **prompt_tokenized,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria)

    res_str = tokenizer.decode(res[0].tolist())

    return res_str


def infer(
        repo_name: str = 'ImScientist/Llama-2-7b-hf-finetuned',
        usr_instruction: str = 'Коя е най-голямата страна в света?',
        usr_input: str = None,
        lang: Literal['en', 'de', 'bg'] = 'en'
):
    """ Generate model responses to prompts """

    tokenizer, peft_model = load_finetuned_model(repo_name)

    res = inference(
        peft_model,
        tokenizer,
        usr_instruction=usr_instruction,
        usr_input=usr_input,
        lang=lang)

    logger.info(res)
