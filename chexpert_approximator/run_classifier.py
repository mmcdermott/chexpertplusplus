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

def build_and_train(
    df,
    bert_model,
    processor,
    task_dimensions,
    output_dir,
    gradient_accumulation_steps = 1,
    gpu                         = None,
    do_train                    = True,
    do_eval                     = False,
    seed                        = 42,
    do_lower_case               = False,
    max_seq_length              = 128,
    train_batch_size            = 32,
    eval_batch_size             = 8,
    learning_rate               = 5e-5,
    num_train_epochs            = 3,
    warmup_proportion           = 0.1,
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

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            gradient_accumulation_steps))

    train_batch_size = train_batch_size // gradient_accumulation_steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_map = processor.get_labels()
    task_list = list(label_map.keys())
    num_tasks = len(label_map)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    #TESTING
    #splits = processor.create_cv_examples(data_dir)
    

    train_examples = None
    num_train_optimization_steps = None
    if do_train:
        train_examples = processor.get_train_examples(df)
        num_train_optimization_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs

    # Prepare model
    cache_dir = cache_dir if cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_-1')

    if model is None:
        model = BertForMultitaskSequenceClassification.from_pretrained(
            bert_model,
            cache_dir=cache_dir,
            num_labels_per_task = {task: len(labels) for task, labels in label_map.items()},
        )

    model.to(device)
    if n_gpu > 1: model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(
        optimizer_grouped_parameters,
         lr=learning_rate,
         warmup=warmup_proportion,
         t_total=num_train_optimization_steps
    )

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if do_train:
        train_features = convert_examples_to_features(
            train_examples, task_list, max_seq_length, tokenizer
        )

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        model.train()
        for _ in tqdm(range(int(num_train_epochs)), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                label_ids = {t: label_ids[:, i] for i, t in enumerate(task_list)}

                _, losses = model(input_ids, segment_ids, input_mask, label_ids)
                loss = sum(losses.values()) / num_tasks

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
        #### HERE NEED TO MODIFY.

    if do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForMultitaskSequenceClassification(
            config,
            num_labels_per_task = {task: len(labels) for task, labels in label_map.items()},
        )
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForMultitaskSequenceClassification.from_pretrained(
            bert_model,
            num_labels_per_task = {task: len(labels) for task, labels in label_map.items()},
        )
    model.to(device)

    if do_eval:
        eval_examples = processor.get_dev_examples(df)
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
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
 
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            
            label_ids = {t: label_ids[:, i] for i, t in enumerate(task_list)}

            with torch.no_grad():
                logits, tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
#                 logits = model(input_ids, segment_ids, input_mask)

            logits = {t: lt.detach().cpu().numpy() for t, lt in logits.items()}
            label_ids = {t: l.to('cpu').numpy() for t, l in label_ids.items()}
            
            tmp_eval_accuracy = multi_task_accuracy(logits, label_ids)

            tmp_eval_loss = sum(tmp_eval_loss.values())
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss}

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()): writer.write("%s = %s\n" % (key, str(result[key])))
        return result
