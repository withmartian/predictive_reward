from transformers import AutoModelForSeq2SeqLM, T5EncoderModel, AutoTokenizer
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, model_path, tokenizer_path=None, max_ranks_per_batch=2):
        super().__init__()
        model = T5EncoderModel.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        self.max_ranks_per_batch = max_ranks_per_batch

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
            rewards=None
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = transformer_outputs[0]

        predicted_rewards = self.v_head(hidden_states).squeeze(-1)

        bs = input_ids.shape[0] // self.max_ranks_per_batch

        ranked = [input_ids[i:i + bs] for i in range(0, len(input_ids), bs)]
        ranked_rewards = [predicted_rewards[i:i + bs] for i in range(0, len(predicted_rewards), bs)]

        predicted_end_scores = torch.stack([
            input_rewards[self.get_start_of_padding(ranked[i][j])]
            for i, batch_rewards in enumerate(ranked_rewards)
            for j, input_rewards in enumerate(batch_rewards)
        ])

        criterion = nn.MSELoss()
        loss = criterion(rewards, predicted_end_scores) if rewards is not None else None

        return {
            "end_scores": predicted_end_scores,
            "loss": loss
        }

    def get_start_of_padding(self, input1):
        inds = (input1 == self.PAD_ID).nonzero()
        return inds[0].item() if len(inds) > 0 else len(input1) - 1