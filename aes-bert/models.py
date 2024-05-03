import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class AESModel(nn.Module):
    def __init__(self, model_name, if_llm=False):
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(
            model_name,
        )

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.max_position_embeddings = 4096
        self.config.num_labels = 1
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.in_dim = self.config.hidden_size

        self.bert_model = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        self.if_llm = if_llm
        if if_llm:

            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=self.config
            )
            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"
                    ,"down_proj",],
                inference_mode=False,
                r=6,
                lora_alpha=32,
                lora_dropout=0.,
            )
            self.bert_model = get_peft_model(self.bert_model, peft_config)
        self.bilstm = nn.LSTM(self.in_dim, self.in_dim, num_layers=1,
                              dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        self.pool = MeanPooling()
        self.last_fc = nn.Linear(self.in_dim * 2, self.config.num_labels)
        # self.fc = nn.LazyLinear(num_classes)
        torch.nn.init.normal_(self.last_fc.weight, std=0.02)
        self.loss_function = nn.MSELoss()
        self.num_labels = 1

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if self.if_llm:
            x = self.bert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        else:
            x = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        x, _ = self.bilstm(x)
        x = self.pool(x, attention_mask)
        logits = self.last_fc(x)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits.view(-1), labels.view(-1))

        output = (logits,)
        return ((loss,) + output) if loss is not None else output

if __name__ == '__main__':

    print('')
