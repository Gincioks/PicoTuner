import logging

from datasets import load_dataset


class DatasetsManager:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            level=logging.DEBUG, filename='logs/datasets.log')

    def change_key(self, dictionary, old_key, new_key):
        if old_key in dictionary:
            dictionary[new_key] = dictionary.pop(old_key)
            return dictionary
        else:
            logging.error(f"Key {old_key} not found in dictionary")
            return dictionary

    def format_dict(self, dataset_name, train_data):
        if "dolly" in dataset_name:
            train_data = self.change_key(
                train_data, 'instruction', 'system_prompt')
            train_data = self.change_key(train_data, 'context', 'user')
            train_data = self.change_key(
                train_data, 'response', 'assistant')
            return train_data
        if "orca" in dataset_name:
            train_data = self.change_key(train_data, 'question', 'user')
            train_data = self.change_key(
                train_data, 'response', 'assistant')
            return train_data

    def format_chatlm_sample(self, sample, dataset_name: str):
        default_sys_msg = "You are Jarvis, a large language model trained by Decentralabs AI. Write out your reasoning step-by-step to be sure you get the right answers!"

        sample = self.format_dict(dataset_name, sample)

        system_message = f"<|im_start|>system\n{sample['system_prompt']}<|im_end|>\n" if len(
            sample["system_prompt"]) > 0 else f"<|im_start|>system\n{default_sys_msg}<|im_end|>\n"

        user = f"<|im_start|>user\n{sample['user']}<|im_end|>\n"
        assistant = f"<|im_start|>assistant\n{sample['assistant']}<|im_end|>\n"
        return system_message + user + assistant

    def format_mistral_instruct_sample(self, sample, dataset_name: str):
        default_sys_msg = "You are Jarvis, a large language model trained by Decentralabs AI. Write out your reasoning step-by-step to be sure you get the right answers!"

        sample = self.format_dict(dataset_name, sample)

        system_message = f"<s>[INST]\nSYSTEM:\n{sample['system_prompt']}\n" if len(
            sample["system_prompt"]) > 0 else f"<s>[INST]\nSYSTEM:\n{default_sys_msg}\n"

        user = f"USER:\n{sample['user']} [/INST]\n"
        assistant = f"{sample['assistant']}</s>"
        return system_message + user + assistant

    def choose_format(self, sample, format: str, dataset_name: str):
        if format == "chatlm":
            return self.format_chatlm_sample(sample=sample, dataset_name=dataset_name)
        elif format == "mistral_instruct":
            return self.format_mistral_instruct_sample(sample=sample, dataset_name=dataset_name)

    def prepare_text_data(self, dataset_name: str, number_of_samples: int = 100, format: str = "mistral_instruct"):
        train_data = load_dataset(dataset_name, split="train")

        formatted = []
        formatted = [self.choose_format(
            sample=s,
            format=format,
            dataset_name=dataset_name
        ) for s in train_data]

        print(f'\nloaded dataset: {len(formatted)} samples')
        print(f'formated: {number_of_samples} samples')
        return '\n\n'.join(formatted[:number_of_samples])
