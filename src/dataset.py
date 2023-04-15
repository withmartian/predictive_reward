from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def ranking2reward(reward_model, reward_tokenizer, ranking, max_length = 512):
    input = reward_tokenizer(
        "<|startoftext|>" + ranking['prompt'] + ranking['output'] + "<|endoftext|>",
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = input['input_ids']
    attention_mask = input['attention_mask']

    reward_model_output = reward_model(input_ids=input_ids, attention_mask=attention_mask)

    return reward_model_output


def create_reward_dataset(reward_model, reward_tokenizer, dataset):
    result = [{'prompt': sample['prompt'], 'reward': ranking2reward(reward_model, reward_tokenizer, sample)['end_scores'][0][0].detach()} for sample in tqdm(dataset)]
    return result


class RewardDataset(Dataset):
    def __init__(self, reward_dataset, tokenizer, max_length):
        self.items = []
        for example in tqdm(reward_dataset):
            current_items = []
            encodings_dict = tokenizer(
                "<|startoftext|>" + example['prompt'] + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            current_items.append(encodings_dict["input_ids"])
            current_items.append(encodings_dict["attention_mask"])
            current_items.append(example['reward'])
            self.items.append(current_items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DataCollator():
    def __init__(self, max_ranks_per_batch=2, max_sequence_length=512, padding_token=0):
        self.max_ranks_per_batch = max_ranks_per_batch
        self.max_sequence_length = max_sequence_length
        self.padding_token = padding_token

    def __call__(self, data):
        batch = {}
        input_ids = []
        attention_mask = []
        reward = []
        for i in range(self.max_ranks_per_batch):
            input_ids.extend([
                f[i * 3] if i * 3 < len(f) else torch.tensor([[self.padding_token] * self.max_sequence_length])
                for f in data
            ])
            attention_mask.extend([
                f[i * 3 + 1] if i * 3 < len(f) else torch.tensor([[2] * self.max_sequence_length])
                for f in data
            ])
            reward.extend([
                torch.unsqueeze(f[i * 3 + 2], 0) if i * 3 < len(f) else torch.tensor([0])
                for f in data
            ])
        batch["input_ids"] = torch.cat(input_ids)
        batch["attention_mask"] = torch.cat(attention_mask)
        batch["rewards"] = torch.cat(reward)
        return batch
