#!/usr/bin/env python3
import shutil
import click
import os
import sys
import threading
import time
import inquirer
from pico_tuner import DatasetsManager, ModelsManager, FinetuneManager, InferenceManager
from pico_tuner.configs import DatasetsConfig, LoraConfig, TrainingConfig, OthersConfig
from pico_tuner.utils import find_and_delete_folder, download_and_unzip
from pico_tuner.llama import LlamaTokenizer
from pico_tuner.mistral import MistralTokenizer


class CLIManager():
    def __init__(self):
        # Configs
        self.dataset_config = DatasetsConfig()
        self.lora_config = LoraConfig()
        self.training_config = TrainingConfig()
        self.others_config = OthersConfig()

        # Managers
        self.datasets_manager = DatasetsManager()
        self.models_manager = ModelsManager()
        self.inference_manager = InferenceManager(
            frozen_dtype=self.training_config.frozen_dtype,
            compute_dtype=self.training_config.compute_dtype,
            device=self.training_config.device,
        )

        self.models_path = "models"
        self.models_data_path = "models_data"

        self.model_dict = {}
        self.models_list_number = 1

        self.models_list = self.models_manager.get_models_list(
            os.path.join(self.models_path, self.models_data_path))
        pass

    def setup(self):
        selected_model_info = self.select_model()
        _, model_base, _ = self.models_manager.get_model_paths(
            models_folder_path=self.models_path,
            model_collection_path=selected_model_info['collection'],
            model=selected_model_info['model_name'],
            create_frozen=False,
        )

        url = selected_model_info.get('url', None)

        if url:
            download_and_unzip(selected_model_info['url'], model_base)
        else:
            print(f"Please put model weights in: {model_base}")
            pass

    # Fancy spinner-style loader
    def loader(self, stop_event):
        spinner = ['-', '\\', '|', '/']
        i = 0

        while not stop_event.is_set():
            sys.stdout.write('\r' + spinner[i % len(spinner)] + ' ')
            sys.stdout.flush()
            i += 1
            time.sleep(0.2)

        # Clearing the spinner and resetting the cursor to the beginning
        sys.stdout.write('\r' + ' ' * 30)
        sys.stdout.write('\r')
        sys.stdout.flush()

    def prepare_dataset(self, dataset: str = None, format: str = None):
        if not dataset:
            dataset = str(
                input("Enter a dataset name(leave blank to select default): "))

        if not dataset:
            dataset = self.dataset_config.default_dataset
            print(f"Using default dataset: {dataset}")

        if not format:
            questions = [
                inquirer.List('selected_format',
                              message="Please choose dataset format: ",
                              choices=["ChatLM", "Mistral instruct"],
                              carousel=True
                              )
            ]
            answers = inquirer.prompt(questions)
            selected_format = answers["selected_format"]
            if selected_format == "ChatLM":
                format = "chatlm"
            elif selected_format == "Mistral instruct":
                format = "mistral_instruct"
            else:
                format = "chatlm"

        print(f"Preparing dataset {dataset}")
        stop_event = threading.Event()

        # Start the loader in a new thread
        loader_thread = threading.Thread(
            target=self.loader, args=(stop_event,))
        loader_thread.start()

        try:
            text = self.datasets_manager.prepare_text_data(
                dataset_name=dataset,
                number_of_samples=100,
                format=format,
            )
        finally:
            # Stop the loader
            stop_event.set()

        # Wait for the loader to print "Loading completed"
        loader_thread.join()
        print()
        return text

    def prepare_model(self, selected_model_info: dict = None):
        if selected_model_info is None:
            selected_model_info = self.select_model()

        model_path, model_base, frozen_model = self.models_manager.get_model_paths(
            models_folder_path=self.models_path,
            model_collection_path=selected_model_info['collection'],
            model=selected_model_info['model_name'],
            create_frozen=True
        )

        print(f"Preparing model at {model_path}")
        stop_event = threading.Event()

        # Start the loader in a new thread
        loader_thread = threading.Thread(
            target=self.loader, args=(stop_event,))
        loader_thread.start()

        try:
            self.models_manager.prepare_model(
                model_base=model_base,
                frozen_model=frozen_model,
                compute_dtype=self.training_config.compute_dtype,
                frozen_dtype=self.training_config.frozen_dtype,
                offload_to=self.others_config.offload_to,
                lora_rank=self.lora_config.lora_rank,
            )
        finally:
            # Stop the loader
            stop_event.set()

        # Wait for the loader to print "Loading completed"
        loader_thread.join()
        print()
        return model_path, frozen_model

    def finetune(self, iterations):
        selected_model_info = self.select_model()

        model_path, frozen_model = self.prepare_model(
            selected_model_info=selected_model_info)

        text = self.prepare_dataset()

        if "llama" in frozen_model:
            tokenizer = LlamaTokenizer(os.path.join(
                frozen_model, 'tokenizer.model'))
        else:
            tokenizer = MistralTokenizer(os.path.join(
                frozen_model, 'tokenizer.model'))

        tokens = tokenizer.encode(text, True, True)

        print(f'\n loaded tokens: {len(tokens)} tokens \n')

        tuner = FinetuneManager(
            model_path=model_path,
            frozen_model=frozen_model,
            frozen_dtype=self.training_config.frozen_dtype,
            seq_len=self.dataset_config.seq_len,
            tokenizer=tokenizer,
            lora_rank=self.lora_config.lora_rank,
            log_lora_weight=self.lora_config.log_lora_weight,
            log_lora_grad=self.lora_config.log_lora_grad,
            batch_size=self.training_config.batch_size,
            tokens=tokens,
            device=self.training_config.device,
            adamw_eps=self.training_config.adamw_eps,
            lr=self.training_config.lr,
            compute_dtype=self.training_config.compute_dtype,
            eval_before_training=self.others_config.eval_before_training,
            eval_period=self.others_config.eval_period,
            gen_tokens=self.others_config.gen_tokens,
            test_prompts=self.others_config.prompt,
        )

        stop_event = threading.Event()

        # Start the loader in a new thread
        loader_thread = threading.Thread(
            target=self.loader, args=(stop_event,))
        loader_thread.start()

        print(
            f"Finetuning model {selected_model_info['model_name'].capitalize()}")

        try:
            tuner.train(iterations=iterations)
        finally:
            # Stop the loader
            stop_event.set()

        # Wait for the loader to print "Loading completed"
        loader_thread.join()
        print()
        pass

    def infer(self, selected_model_info, prompt, length):
        if selected_model_info is None:
            selected_model_info = self.select_model()

        model_path, _, frozen_model = self.models_manager.get_model_paths(
            models_folder_path=self.models_path,
            model_collection_path=selected_model_info['collection'],
            model=selected_model_info['model_name'],
            create_frozen=True
        )

        if os.path.exists(frozen_model) and os.listdir(frozen_model):
            self.prepare_model(selected_model_info=selected_model_info)

        lora_path = os.path.join(model_path, "finetuned")
        lora_weights = self.select_lora(lora_path)

        self.inference_manager.generate(
            frozen_model_path=frozen_model,
            lora_weights=lora_weights,
            prompt=prompt,
            length=length
        )

    def reset(self):
        print("Performing reset operations.")
        inputs_folder = "inputs"
        logs_folder = "logs"

        # Remove the folder
        if os.path.exists(inputs_folder):
            shutil.rmtree(inputs_folder)
        if os.path.exists(logs_folder):
            shutil.rmtree(logs_folder)
        # Add your reset logic here
        prepared_folder_name = "prepared"
        finetuned_folder_name = "finetuned"
        find_and_delete_folder(self.models_path, prepared_folder_name)
        find_and_delete_folder(self.models_path, finetuned_folder_name)
        pass

    def select_model(self):
        print("\n---------------------Models---------------------\n")
        for model_collection in self.models_list:
            print(model_collection.name.capitalize())
            for model in model_collection.versions:
                print("   " + str(self.models_list_number) +
                      ". Model: " + model.model_name)
                self.model_dict[self.models_list_number] = {
                    "model_name": model.model_name, "collection": model_collection.name}
                self.models_list_number += 1
        print()
        selected_model_number = int(
            input("Select a model by entering its number: "))
        selected_model_info = self.model_dict.get(selected_model_number)

        if selected_model_info:
            print(
                f"You have selected: {selected_model_info['model_name']} from collection {selected_model_info['collection']}")
            return selected_model_info
        else:
            print("Invalid selection. Please run the program again.")
            raise SystemExit

    def select_lora(self, lora_path):
        if os.path.exists(lora_path) and os.listdir(lora_path):
            # list all the files in the directory
            files = os.listdir(lora_path)

            # Create inquirer list questions
            questions = [
                inquirer.List('selected_file',
                              message="Please choose a Lora file",
                              choices=files,
                              carousel=True
                              )
            ]
            answers = inquirer.prompt(questions)
            chosen_file = answers["selected_file"]

            # assign chosen file's path to lora_weights
            lora_weights = os.path.join(lora_path, chosen_file)
            return lora_weights
        else:
            print("No Lora weights found.")
            return None


cli = CLIManager()


@click.command(name="init")
def init():
    cli.setup()


@click.command(name="prepare-dataset")
def prepare_dataset_text():
    cli.prepare_dataset()


@click.command(name="prepare-model")
def prepare_model():
    cli.reset()
    cli.prepare_model()


@click.command(name="finetune")
@click.option('--iterations', default=10, help='Number of iterations for training', type=int)
def finetune(iterations):
    cli.finetune(iterations)


@click.command(name="infer")
@click.option('--prompt', default=None, help='Prompt for model', type=str)
@click.option('--length', default=None, help='Length of response', type=int)
def infer(prompt, length):
    cli.infer(selected_model_info=None, prompt=prompt, length=length)


@click.command(name="reset")
def reset():
    cli.reset()


if __name__ == '__main__':
    click.Group(
        commands=[init, prepare_dataset_text, prepare_model, finetune, infer, reset])()
