# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, tqdm_notebook

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

#added
import json
from random import shuffle
from sklearn.model_selection import KFold
import math

from .data_processor import *
from .model import *

def reload_and_get_logits(
    df,
    bert_model,
    processor,
    task_dimensions,
    output_dir,
    processor_args              = {},
    gpu                         = None,
    seed                        = 42,
    do_lower_case               = False,
    max_seq_length              = 128,
    batch_size                  = 8,
    learning_rate               = 5e-5,
    num_train_epochs            = 3,
    cache_dir                   = None,
    tqdm                        = tqdm_notebook,
    model                       = None,
):
    print(bert_model)
   
    if gpu is not None and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu  = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        raise ValueError("Output directory ({}) doesn't exist or is not empty.".format(output_dir))

    label_map = processor.get_labels()
    task_list = list(label_map.keys())
    num_tasks = len(label_map)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    # Prepare model
    ### TODO(mmd): this is where to reload the model properly.
    cache_dir = cache_dir if cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_-1')
    # Load a trained model and config that you have fine-tuned
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    config = BertConfig(output_config_file)

    if model is None:
        model = BertForMultitaskSequenceClassification(
            config,
            num_labels_per_task = {task: len(labels) for task, labels in label_map.items()},
        )
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    eval_examples = processor.get_examples(df, **processor_args)
    eval_features = convert_examples_to_features(
        eval_examples, task_list, max_seq_length, tokenizer
    )

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    all_logits = {t: [] for t in task_list}
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        label_ids = {t: label_ids[:, i] for i, t in enumerate(task_list)}

        with torch.no_grad():
            logits, tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
#             logits = model(input_ids, segment_ids, input_mask)

        logits = {t: lt.detach().cpu().numpy() for t, lt in logits.items()}
        for t in task_list: all_logits[t].extend(list(logits[t]))
        label_ids = {t: l.to('cpu').numpy() for t, l in label_ids.items()}

        tmp_eval_accuracy = multi_task_accuracy(logits, label_ids)

        tmp_eval_loss = sum(tmp_eval_loss.values())
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()): writer.write("%s = %s\n" % (key, str(result[key])))
    return model, all_logits, result
