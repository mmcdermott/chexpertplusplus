import numpy as np

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn

def multi_task_accuracy(out_by_task, labels_by_task):
    acc = 0
    for task in out_by_task.keys():
        out, labels = out_by_task[task], labels_by_task[task]
        outputs = np.argmax(out, axis=1)
        acc += np.sum(outputs == labels)
    return acc / len(out_by_task)

# TODO(mmd): should probably not do via a dictionary. Possible to do all as tensor.
class BertForMultitaskSequenceClassification(BertPreTrainedModel):
    """BERT model for multitask sequence classification.
    This module is composed of the BERT model with a linear layer per task on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels_per_task`: A dictionary from task_name to the number of classes for the classifier.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels_per_task`: A dictionary of task name to labels for the classification output: 
            {task: torch.LongTensor of shape [batch_size] with indices selected in [0, ..., num_labels_per_task[task]]}.
    Outputs:
    TODO(mmd): update outputs & example usage.
            Outputs the classification logits per task as a dictionary of shape {task: [batch_size, num_labels]}.
            Outputs the CrossEntropy classification loss as a dictionary of shape {task: loss} for tasks with provided labels
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels_per_task):
        super().__init__(config)
        self.num_labels_per_task = num_labels_per_task
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleDict(
            {task_name: nn.Linear(config.hidden_size, task_dim) for task_name, task_dim in num_labels_per_task.items()}
        )
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels_per_task={}):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = {task: layer(pooled_output) for task, layer in self.classifiers.items()}

        losses = {}
        for task, labels in labels_per_task.items():
            losses[task] = nn.CrossEntropyLoss()(logits[task].view(-1, self.num_labels_per_task[task]), labels.view(-1))
        
        return logits, losses
