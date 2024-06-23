"""
    A dedicated helper to manage templates and prompt building.
"""

from typing import Union

PROMPT_TEMPLATE_ALPACA_EN = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:"
    ),
    "response_split": "### Response:"
}

PROMPT_TEMPLATE_ALPACA_BG = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": (
        "По-долу е представена инструкция, която описва дадена задача, в "
        "съчетание с входни данни, които осигуряват допълнителен контекст. "
        "Напишете отговор, който да допълва по подходящ начин заявката.\n\n"
        "### Инструкция:\n{instruction}\n\n"
        "### Входни данни:\n{input}\n\n"
        "### Отговор:"
    ),
    "prompt_no_input": (
        "По-долу е представена инструкция, която описва дадена задача. "
        "Напишете отговор, който да допълва по подходящ начин заявката.\n\n"
        "### Инструкция:\n{instruction}\n\n"
        "### Отговор:"
    ),
    "response_split": "### Отговор:"
}

PROMPT_TEMPLATE_ALPACA_DE = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": (
        "Nachfolgend finden Sie eine Anweisung, die eine Aufgabe beschreibt, "
        "gepaart mit einer Eingabe, die weiteren Kontext liefert. "
        "Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.\n\n"

        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Anweisung:\n{instruction}\n\n"
        "### Eingabe:\n{input}\n\n"
        "### Antwort:"
    ),
    "prompt_no_input": (
        "Nachfolgend finden Sie eine Anweisung, die eine Aufgabe beschreibt. "
        "Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.\n\n"
        "### Anweisung:\n{instruction}\n\n"
        "### Antwort:"
    ),
    "response_split": "### Antwort:"
}


def generate_prompt(
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
        lang: str = 'en'
) -> str:
    """ Returns the full prompt from instruction and optional input
    if a label (=response, =output) is provided, it's also appended.
    """

    if lang == 'en':
        prompt_template = PROMPT_TEMPLATE_ALPACA_EN
    elif lang == 'de':
        prompt_template = PROMPT_TEMPLATE_ALPACA_DE
    else:
        prompt_template = PROMPT_TEMPLATE_ALPACA_BG

    if input:
        res = (prompt_template["prompt_input"]
               .format(instruction=instruction, input=input))
    else:
        res = (prompt_template["prompt_no_input"]
               .format(instruction=instruction))
    if label:
        res = f"{res}{label}"

    return res


def isolate_prompt_response(output: str) -> str:
    """ Fetch the response part from the output """

    return output.split(PROMPT_TEMPLATE_ALPACA_EN["response_split"])[1].strip()
