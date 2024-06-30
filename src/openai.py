"""
    This part contains some functions that are used to translate the
    training dataset from English into Bulgarian. As you may see, the
    translation was not always successful. For example, instead of translating
    a question a translated answer to the question was provided.

    The openai dependency is not part of requirements.txt

    Links:
        https://cookbook.openai.com/

        https://openai.com/api/pricing/

        https://platform.openai.com/usage
        https://platform.openai.com/examples
        https://platform.openai.com/docs/api-reference/introduction?lang=curl
        https://platform.openai.com/docs/models/gpt-3-5-turbo
        https://platform.openai.com/settings/organization/limits
        https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-one

        https://cookbook.openai.com/examples/how_to_handle_rate_limits
"""

import os
import re
import json
import glob
import time
import logging
from datasets import load_dataset

import numpy as np
import pandas as pd
from openai import OpenAI

OPEN_AI_KEY = ''

logger = logging.getLogger()


def matches_regex(regex, text):
    return bool(re.compile(regex).search(text))


def contains_code(text):
    """ Filter based on keywords that indicate code """

    code_blacklist = ['&&', '|', '<html>', 'SELECT', '{', '}', '</']

    return (
            any(code_keyword in text for code_keyword in code_blacklist) |
            # e.g. fn_name(
            matches_regex(r'\w+\(', text) |
            # e.g. this.language
            matches_regex(r'\[A-z]+\.[A-z]+', text))


def create_translatable_subset(save_path: str):
    """ Crate a translatable subset of the input data """

    data = load_dataset('yahma/alpaca-cleaned')
    data = [el for el in data['train']]

    df_data = pd.DataFrame(data)

    cond = (df_data['input'].map(contains_code) |
            df_data['output'].map(contains_code) |
            df_data['output'].map(contains_code))

    df_data.loc[~cond].to_parquet(save_path)


def generate_chat_completion_input(idx: str, text: str, max_tokens: int, lang: str = 'Bulgarian'):
    """ Generate input that will be passed to a chat completion model """

    # f"Translate the following text from English into {lang}"
    # f"You will be provided with a sentence in English, "
    # f"and your task is to translate it into {lang}."

    res = {
        "custom_id": f'request-{idx:05d}',
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo-0125",
            "messages": [
                {"role": "system",
                 "content": f"Translate the following text from English into {lang}"},
                {"role": "user",
                 "content": text}],
            "max_tokens": max_tokens
        }
    }

    return res


def generate_requests_jsonl(
        save_path: str,
        df: pd.DataFrame,
        words_tokens_multiplier: int = 8
):
    """ Generate requests jsonl """

    requests_all = []

    for col in ['instruction', 'input', 'output']:
        requests = (
            df[['idx', col, f'words_{col}']]
            .loc[lambda x: x[f'words_{col}'] > 0]
            .apply(lambda x: generate_chat_completion_input(
                idx=f'idx-{col}-{x.iloc[0]:05d}',
                text=x.iloc[1],
                max_tokens=words_tokens_multiplier * x.iloc[2]), axis=1)
            .tolist())

        requests_all.extend(requests)

    with open(save_path, 'w') as f:
        for item in requests_all:
            f.write(json.dumps(item) + "\n")


def main(save_dir):
    """ Translate part of the training dataset into Bulgarian """

    requests_dir = os.path.join(save_dir, 'requests')
    translated_dir = os.path.join(save_dir, 'translated')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(requests_dir, exist_ok=True)
    os.makedirs(translated_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'data.parquet')
    create_translatable_subset(save_path=save_path)

    df = pd.read_parquet(save_path)
    for c in ['instruction', 'input', 'output']:
        df[f'words_{c}'] = df[c].map(lambda x: len(x.split(' ')) if x != '' else 0)
    df['idx'] = df.index.values
    df['words'] = df['words_instruction'] + df['words_input'] + df['words_output']

    logger.info(f'Create requests files in {requests_dir}')
    idxs = np.arange(0, 10_001, 500)
    paths_requests = []
    for idx_i, idx_f in zip(idxs[:-1], idxs[1:]):
        path_requests = os.path.join(requests_dir, f'requests_{idx_f:05d}.jsonl')
        paths_requests.append(path_requests)
        generate_requests_jsonl(save_path=path_requests, df=df.iloc[idx_i:idx_f])

    client = OpenAI(api_key=OPEN_AI_KEY)

    # Upload batch input files
    batch_requests_file_ids = []
    for path_requests in paths_requests:
        batch_requests_file = client.files.create(
            file=open(path_requests, "rb"),
            purpose="batch")
        batch_requests_file_ids.append(batch_requests_file.id)

    # Create jobs
    for idx, batch_requests_file_id in enumerate(batch_requests_file_ids, start=1):
        job = client.batches.create(
            input_file_id=batch_requests_file_id,
            metadata={"description": f"new-bg-translate-{idx:05d}"},
            endpoint="/v1/chat/completions",
            completion_window="24h")

        while True:
            job_status = client.batches.retrieve(job.id)
            if job_status.status == 'completed':
                break
            time.sleep(120)

    # Store results
    batches = client.batches.list()
    batches = [b for b in batches if b.status == 'completed']
    batches = [b for b in batches if re.fullmatch(pattern='^new-bg-translate-\d{5}$',
                                                  # '^bg-translate-\d{2}\w*$'
                                                  string=b.metadata.get('description', ''))]

    for idx, b in enumerate(batches):
        logger.info(f'id: {b.id} status: {b.status}')
        path = os.path.join(translated_dir, f'translated-{idx:05d}.jsonl')
        client.files.content(file_id=b.output_file_id).write_to_file(path)

    # Read the content and do something with it
    #
    data_all = []
    for file in glob.glob(f'{translated_dir}/*.jsonl'):
        with open(file) as f:
            data = [json.loads(line) for line in f]
            data = [[el['custom_id'],
                     el['response']['status_code'],
                     el['response']['body']['choices'][0]['message']['content']] for el in data]
            data_all.extend(data)

    df_out = (pd.DataFrame(data_all, columns=['idx', 'status_code', 'text'])
              .loc[lambda x: x['status_code'] == 200]
              .sort_values('idx'))

    df_out['type'] = df_out['idx'].str.removeprefix('request-idx-').map(lambda x: x.split('-')[0])
    df_out['id'] = df_out['idx'].str.removeprefix('request-idx-').map(lambda x: x.split('-')[1])
    df_out['id'] = df_out['id'].astype(int)

    df_out_1 = df_out.loc[lambda x: x['type'] == 'instruction', ['id', 'text']].rename(columns={'text': 'instruction'})
    df_out_2 = df_out.loc[lambda x: x['type'] == 'input', ['id', 'text']].rename(columns={'text': 'input'})
    df_out_3 = df_out.loc[lambda x: x['type'] == 'output', ['id', 'text']].rename(columns={'text': 'output'})

    df_out = df_out_1.merge(df_out_2, how='left', on='id').merge(df_out_3, how='left', on='id')
    df_out = df_out[['instruction', 'input', 'output']].fillna('')

    df_out.to_parquet(os.path.join(save_dir, 'data_translated.parquet'))

    with open(os.path.join(save_dir, 'data_translated.jsonl'), 'w') as f:
        for item in df_out.to_dict(orient='records'):
            f.write(json.dumps(item) + "\n")
