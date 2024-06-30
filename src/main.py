import click
import logging
from train import train
from infer import infer

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option(
    '--dataset',
    type=click.Choice(['yahma/alpaca-cleaned', 'ImScientist/alpaca-cleaned-bg']),
    required=True,
    help='Training dataset')
@click.option(
    '--target-repo',
    type=str,
    default=None,
    required=False,
    help='Hugging face repo where the model will be stored')
def train_fn(dataset, target_repo):
    """ Fine-tune llama model """

    train(dataset_name=dataset,
          target_repo_name=target_repo)


@cli.command(name='infer')
@click.option('--usr-instruction', type=str, required=True)
@click.option('--usr-input', type=str, default=None)
@click.option(
    '--repo-fine-tuned', type=str, required=True,
    help='Hugging face repo from which the LORA weights will be loaded')
@click.option('--lang', type=click.Choice(['en', 'de', 'bg']))
def infer_fn(usr_instruction, usr_input, repo_fine_tuned, lang):
    """ Fine-tune llama model """

    infer(
        repo_name=repo_fine_tuned,
        usr_instruction=usr_instruction,
        usr_input=usr_input,
        lang=lang)


if __name__ == "__main__":
    cli()
