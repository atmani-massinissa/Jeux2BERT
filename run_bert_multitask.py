# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

from __future__ import absolute_import, division, print_function
import sys
import argparse
import logging
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, ConcatDataset, DistributedSampler)
from tqdm import tqdm, trange
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MarginRankingLoss, TripletMarginWithDistanceLoss
from sklearn import metrics
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW, BertTokenizer, FlaubertConfig ,BertConfig,FlaubertTokenizer,FlaubertModel,AlbertTokenizer,AlbertModel,AlbertConfig
from models import BertForSequenceClassification
from utils import *
import numpy as np
from dataloaders import BidirectionalOneShotIterator, BertTrainDataset, BertTrainDataset2, BertTrainDataset3
from torch.utils.tensorboard import SummaryWriter
from packaging import version
import gc
logger = logging.getLogger(__name__)


try:
    from apex import amp  # noqa: F401
    _has_apex = True
except ImportError:
    _has_apex = False

_use_native_amp = False
_use_apex = False


def is_apex_available():
    return _has_apex


# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="/data/matmani/JDM/data/",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
                        #flaubert/flaubert_base_cased#76361#135363./model/fralbert-base_tokenizer #/data/matmani/JDM/model/fralbert-base        
    parser.add_argument("--bert_model", default="/data/matmani/JDM/output/13", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")#/data/matmani/JDM/model/flaubert_tokenizer_small
    parser.add_argument("--bert_tokenizer", default="/data/matmani/JDM/model/flaubert_tokenizer_sm", type=str, required=False,
                        help="Bert pre-trained tokenizer selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--rel_tokenizer", default="/data/matmani/JDM/model/flaubert_tokenizer_sm", type=str, required=False,
                        help="Bert pre-trained tokenizer selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_list",
                        default="lp,rr",
                        type=str,
                        required=False,
                        help="The list of name of the tasks to train.")
    parser.add_argument("--output_dir",
                        default="/data/matmani/JDM/output/",
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_debug",
                        action='store_true')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default= 480,#474,#725,#482,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=740,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=30.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--eval_range', type=str, default='1:34')
    parser.add_argument('--eval_task', type=str, default="rp", choices=["lp", "rp","rr"])
    parser.add_argument('--tb_log_dir', type=str, default="runs/null")
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--debug_index', type=int, default=0)
    parser.add_argument('--negative_sample_size', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    args = parser.parse_args()

    summary = SummaryWriter(log_dir=args.tb_log_dir)
    task_list = args.task_list.split(",")

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    n_gpu = torch.cuda.device_count()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #torch.distributed.init_process_group(backend='nccl')

    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    lp_processor = LPProcessor(args.data_dir)
    rp_processor = RPProcessor(args.data_dir)
    rr_processor = RRProcessor()
    rf_processor = RFProcessor(args.data_dir)

    lp_label_list = lp_processor.get_labels(args.data_dir)
    lp_num_labels = len(lp_label_list)
    rp_label_list = rp_processor.get_labels(args.data_dir)
    rp_num_labels = len(rp_label_list)

    entity_list = lp_processor.get_entities(args.data_dir)

    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        tokenizer = FlaubertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=args.do_lower_case)
        rel_tokenizer = FlaubertTokenizer.from_pretrained(args.rel_tokenizer, do_lower_case=args.do_lower_case)
        
        #tokenizer = AlbertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=args.do_lower_case)
        #rel_tokenizer = AlbertTokenizer.from_pretrained(args.rel_tokenizer, do_lower_case=args.do_lower_case)

        # Prepare model
        config = FlaubertConfig.from_pretrained(args.bert_model)
        #config = AlbertConfig.from_pretrained(args.bert_model)
        setattr(config, "lp_num_labels", lp_num_labels)
        setattr(config, "rp_num_labels", rp_num_labels)
        setattr(config, "initializer_range", 0.02)
        #setattr(config, "output_hidden_states", True)
        #model = BertForSequenceClassification.from_pretrained(args.bert_model, config=config)
        model = BertForSequenceClassification.from_pretrained(args.bert_model, config=config)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model,device_ids=[0, 1, 2, 3])
        model.to(device)
        print(device)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))


        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        optimizer = optim.Adam(params =  model.parameters(), lr=args.learning_rate)

        if args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

            #model = torch.nn.DataParallel(
            #    model,
            #    device_ids=[args.local_rank]
            #)

        task_total_dataset = dict()
        train_dataloader = {}
        #ent2input = lp_processor.get_ent2input(args.data_dir, rel_tokenizer, tokenizer)
        #print(len(ent2input))
        #print("lp_train_examples ent2input",ent2input)
        #rel2input = lp_processor.get_rel2input(args.data_dir, rel_tokenizer)
        if "sym" in task_list:  # load link prediction data
            train_sym_bin_path = os.path.join(args.data_dir, 'train-sym_s.pt')
            #print("lp_train_examples ent2input",ent2input)
            if os.path.exists(train_sym_bin_path):
                task_total_dataset["sym"] = torch.load(train_sym_bin_path)
                #sampler = RandomSampler(torch.load(train_lp_bin_path))
                logger.info("load %s" % train_sym_bin_path)
            else:
                train_triples_sym = lp_processor.get_train_sym_triples(args.data_dir)
                train_triples_asym = lp_processor.get_train_asym_triples(args.data_dir)

                sym = BertTrainDataset(train_triples_sym, ent2input, rel2input, args.max_seq_length, lp_processor.num_entity,
                                           lp_processor.num_relation, 1,"sym")
                asym = BertTrainDataset(train_triples_asym, ent2input, rel2input, args.max_seq_length, lp_processor.num_entity,
                                           lp_processor.num_relation, 1,"asym")

                task_total_dataset["sym"] = ConcatDataset([sym,asym])
                torch.save(task_total_dataset["sym"], train_sym_bin_path)
            train_dataloader["sym"] = DataLoader(task_total_dataset["sym"],
                                             batch_size=args.train_batch_size,
                                             sampler= SequentialSampler(task_total_dataset["sym"]),
                                             collate_fn=BertTrainDataset.collate_fn_rr,
                                             pin_memory=False,
                                             shuffle=False,   
                                             num_workers=5)
            
        if "rf" in task_list:  # load link prediction data
            train_rf_bin_path = os.path.join(args.data_dir, 'train-rf.pt')
            #print("lp_train_examples ent2input",ent2input)
            if os.path.exists(train_rf_bin_path):
                task_total_dataset["rf"] = torch.load(train_rf_bin_path)
                #sampler = RandomSampler(torch.load(train_lp_bin_path))
                logger.info("load %s" % train_rf_bin_path)
            else:
                train_triples = rf_processor.get_train_triples(args.data_dir)
                ds = BertTrainDataset3(train_triples, ent2input, rel2input, args.max_seq_length, lp_processor.num_entity,
                                           lp_processor.num_relation,"normal")
                syn_ds = BertTrainDataset3(train_triples, ent2input, rel2input, args.max_seq_length, lp_processor.num_entity,
                                           lp_processor.num_relation,"syn")
                ant_ds = BertTrainDataset3(train_triples, ent2input, rel2input, args.max_seq_length, lp_processor.num_entity,
                                           lp_processor.num_relation,"ant")

                task_total_dataset["rf"] = ConcatDataset([ds,syn_ds,ant_ds])
                torch.save(task_total_dataset["rf"], train_rf_bin_path)
            #sampler = SequentialSampler#RandomSampler(task_total_dataset["lp"])#DistributedSampler(task_total_dataset["lp"])
            train_dataloader["rf"] = DataLoader(task_total_dataset["rf"],
                                             batch_size=args.train_batch_size,
                                             sampler= SequentialSampler(task_total_dataset["rf"]),
                                             collate_fn=BertTrainDataset3.collate_fn_bert,
                                             pin_memory=False,
                                             shuffle=False,   
                                             num_workers=5)
            #train_dataloader["lp"] = lp_train_dataloader
            #del task_total_dataset["lp"]
            #gc.collect()
            logger.info("  [Retrofitting] Num examples = %d", len(train_dataloader["rf"]))
        #ent2input = lp_processor.get_ent2input(args.data_dir, rel_tokenizer, tokenizer)
        #print("lp_train_examples ent2input",ent2input)
        #rel2input = lp_processor.get_rel2input(args.data_dir, rel_tokenizer)
        #import pickle
        #ent2input = pickle.load( open( "/data/matmani/JDM/data/ent2input.p", "rb" ) )
        #rel2input = pickle.load( open( "/data/matmani/JDM/data/rel2input.p", "rb" ) )
        if "lp" in task_list:  # load link prediction data                     
            train_lp_bin_path = os.path.join(args.data_dir, '_train-lp_3.pt') #10
            #ent2input = lp_processor.get_ent2input(args.data_dir, rel_tokenizer, tokenizer)
            #print("lp_train_examples ent2input",ent2input)
            #rel2input = lp_processor.get_rel2input(args.data_dir, rel_tokenizer)
            if os.path.exists(train_lp_bin_path):
                #print("continue")
                task_total_dataset["lp"] = torch.load(train_lp_bin_path)
                #sampler = RandomSampler(torch.load(train_lp_bin_path))
                logger.info("load %s" % train_lp_bin_path)
            else:
                print("starts")
                _train_triples = lp_processor.get_train_triples(args.data_dir)#[:74070107//2]
                train_triples = _train_triples[74070107//2:]
                #train_triples = lp_processor.get_train_triples(args.data_dir)[:200000]
                #print("here lp___ ",train_triples[0])
                print("starts",len(train_triples))
                neg_train_triples = lp_processor.get_neg_train_triples(args.data_dir)
                agg_train_triples = lp_processor.get_agg_train_triples(args.data_dir)
                #args.negative_sample_size
                print("starts")
                head_ds = BertTrainDataset(np.random.permutation(train_triples), ent2input, rel2input, args.max_seq_length, 
                                           lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                #tail_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, lp_processor.num_entity,
                #                           lp_processor.num_relation, 2, 'tail-batch')
                #rel_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, lp_processor.num_entity,
                #                           lp_processor.num_relation, 2, 'rel-batch')
                head_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                                            lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                #rel_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                #                            lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                neg_ds = BertTrainDataset(np.random.permutation(neg_train_triples), ent2input, rel2input, 
                                          args.max_seq_length, lp_processor.num_entity,
                                           lp_processor.num_relation, 0, 'negative')
                #print(head_ds.shape)
                #print(rel2input)
                #print(len(tail_ds))
                task_total_dataset["lp"] = ConcatDataset([head_ds])
                print("starts")
                torch.save(task_total_dataset["lp"], train_lp_bin_path)
                #train_lp_bin_path = os.path.join(args.data_dir, '_train-lp_1.pt') #10

                if os.path.exists(train_lp_bin_path):
                        print("continue")
                        #task_total_dataset["lp"] = torch.load(train_lp_bin_path)
                        #sampler = RandomSampler(torch.load(train_lp_bin_path))
                        #logger.info("load %s" % train_lp_bin_path)
                else:
                        print("starts")
                        #train_triples = lp_processor.get_train_triples(args.data_dir)[:74070107//2]
                        #train_triples = lp_processor.get_train_triples(args.data_dir)[:200000]
                        _train_triples = lp_processor.get_train_triples(args.data_dir)#[:74070107//2]
                        train_triples = _train_triples[74070107//2:]
                        #print("here lp___ ",train_triples[0])
                        print("starts")
                        #neg_train_triples = lp_processor.get_neg_train_triples(args.data_dir)
                        agg_train_triples = lp_processor.get_agg_train_triples(args.data_dir)
                        #args.negative_sample_size
                        print("starts")
                        print(len(train_triples))
                        head_ds = BertTrainDataset(np.random.permutation(train_triples), ent2input, rel2input, args.max_seq_length, 
                                                       lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                        #rel_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                        #lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        head_agg = BertTrainDataset(np.random.permutation(agg_train_triples), ent2input, rel2input, args.max_seq_length, 
                                                        lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                        #rel_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                        #                            lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        #neg_ds = BertTrainDataset(np.random.permutation(neg_train_triples), ent2input, rel2input, 
                        #                          args.max_seq_length, lp_processor.num_entity,
                        #                           lp_processor.num_relation, 0, 'negative')
                        #print(head_ds.shape)
                        #print(rel2input)
                        #print(len(tail_ds))
                        task_total_dataset["lp"] = ConcatDataset([head_ds])
                        print("starts")
                        torch.save(task_total_dataset["lp"], train_lp_bin_path)

                #train_lp_bin_path = os.path.join(args.data_dir, 'train-lp_02.pt') #10

                if os.path.exists(train_lp_bin_path):
                        print("continue")
                        #task_total_dataset["lp"] = torch.load(train_lp_bin_path)
                        #sampler = RandomSampler(torch.load(train_lp_bin_path))
                        #logger.info("load %s" % train_lp_bin_path)
                else:
                        print("starts")
                        _train_triples = lp_processor.get_train_triples(args.data_dir)#[:74070107//2]
                        train_triples = _train_triples[:74070107//2]
                        #train_triples = lp_processor.get_train_triples(args.data_dir)[:74070107//2]
                        #train_triples = lp_processor.get_train_triples(args.data_dir)[:200000]
                        #print("here lp___ ",train_triples[0])
                        print("starts")
                        #neg_train_triples = lp_processor.get_neg_train_triples(args.data_dir)
                        agg_train_triples = lp_processor.get_agg_train_triples(args.data_dir)
                        #args.negative_sample_size
                        print("starts")
                        #head_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                        #                           lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                        rel_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                        lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        #tail_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                        #                            lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                        rel_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                                                        lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        #neg_ds = BertTrainDataset(np.random.permutation(neg_train_triples)[:200000], ent2input, rel2input, 
                        #                          args.max_seq_length, lp_processor.num_entity,
                        #                           lp_processor.num_relation, 0, 'negative')
                        #print(head_ds.shape)
                        #print(rel2input)
                        #print(len(tail_ds))
                        task_total_dataset["lp"] = ConcatDataset([rel_ds,rel_agg])
                        print("starts")
                        torch.save(task_total_dataset["lp"], train_lp_bin_path)                  


                #train_lp_bin_path = os.path.join(args.data_dir, 'train-lp_10.pt') #10
                #ent2input = lp_processor.get_ent2input(args.data_dir, rel_tokenizer, tokenizer)
                #print("lp_train_examples ent2input",ent2input)
                #rel2input = lp_processor.get_rel2input(args.data_dir, rel_tokenizer)
                if os.path.exists(train_lp_bin_path):
                        print("continue")
                        #task_total_dataset["lp"] = torch.load(train_lp_bin_path)
                        #sampler = RandomSampler(torch.load(train_lp_bin_path))
                        #logger.info("load %s" % train_lp_bin_path)
                else:
                        print("starts")
                        #_train_triples = lp_processor.get_train_triples(args.data_dir)#[:74070107//2]
                        train_triples = _train_triples[74070107//2:]
                        #train_triples = lp_processor.get_train_triples(args.data_dir)[:200000]
                        #print("here lp___ ",train_triples[0])
                        print("starts")
                        #neg_train_triples = lp_processor.get_neg_train_triples(args.data_dir)
                        #agg_train_triples = lp_processor.get_agg_train_triples(args.data_dir)
                        #args.negative_sample_size
                        print("starts")
                        head_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                                                   lp_processor.num_entity,lp_processor.num_relation, 2, 'head-batch')
                        #rel_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                        #lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        head_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                                                    lp_processor.num_entity,lp_processor.num_relation, 2, 'head-batch')
                        #rel_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                        #                            lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        #neg_ds = BertTrainDataset(np.random.permutation(neg_train_triples)[:200000], ent2input, rel2input, 
                        #                          args.max_seq_length, lp_processor.num_entity,
                        #                           lp_processor.num_relation, 0, 'negative')
                        #print(head_ds.shape)
                        #print(rel2input)
                        #print(len(tail_ds))
                        task_total_dataset["lp"] = ConcatDataset([head_ds,head_agg])
                        print("starts")
                        torch.save(task_total_dataset["lp"], train_lp_bin_path)

                #train_lp_bin_path = os.path.join(args.data_dir, 'train-lp_11.pt') #10

                if os.path.exists(train_lp_bin_path):
                        print("continue")
                        #task_total_dataset["lp"] = torch.load(train_lp_bin_path)
                        #sampler = RandomSampler(torch.load(train_lp_bin_path))
                        #logger.info("load %s" % train_lp_bin_path)
                else:
                        print("starts")
                        #train_triples = lp_processor.get_train_triples(args.data_dir)[:74070107//2]
                        #train_triples = lp_processor.get_train_triples(args.data_dir)[:200000]
                        #print("here lp___ ",train_triples[0])
                        train_triples = _train_triples[74070107//2:]
                        print("starts")
                        #neg_train_triples = lp_processor.get_neg_train_triples(args.data_dir)
                        #agg_train_triples = lp_processor.get_agg_train_triples(args.data_dir)
                        #args.negative_sample_size
                        print("starts")
                        tail_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                                                       lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                        #rel_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                        #lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        tail_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                                                        lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                        #rel_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                        #                            lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        #neg_ds = BertTrainDataset(np.random.permutation(neg_train_triples)[:200000], ent2input, rel2input, 
                        #                          args.max_seq_length, lp_processor.num_entity,
                        #                           lp_processor.num_relation, 0, 'negative')
                        #print(head_ds.shape)
                        #print(rel2input)
                        #print(len(tail_ds))
                        task_total_dataset["lp"] = ConcatDataset([tail_ds,tail_agg])
                        print("starts")
                        torch.save(task_total_dataset["lp"], train_lp_bin_path)

                #train_lp_bin_path = os.path.join(args.data_dir, 'train-lp_12.pt') #10

                if os.path.exists(train_lp_bin_path):
                        print("continue")
                        #task_total_dataset["lp"] = torch.load(train_lp_bin_path)
                        #sampler = RandomSampler(torch.load(train_lp_bin_path))
                        #logger.info("load %s" % train_lp_bin_path)
                else:
                        print("starts")
                        train_triples = _train_triples[74070107//2:]
                        #train_triples = lp_processor.get_train_triples(args.data_dir)[:74070107//2]
                        #train_triples = lp_processor.get_train_triples(args.data_dir)[:200000]
                        #print("here lp___ ",train_triples[0])
                        print("starts")
                        #neg_train_triples = lp_processor.get_neg_train_triples(args.data_dir)
                        #agg_train_triples = lp_processor.get_agg_train_triples(args.data_dir)
                        #args.negative_sample_size
                        print("starts")
                        #head_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                        #                           lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                        rel_ds = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                        lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        #tail_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                        #                            lp_processor.num_entity,lp_processor.num_relation, 2, 'tail-batch')
                        rel_agg = BertTrainDataset(agg_train_triples, ent2input, rel2input, args.max_seq_length, 
                                                        lp_processor.num_entity,lp_processor.num_relation, 2, 'rel-batch')
                        #neg_ds = BertTrainDataset(np.random.permutation(neg_train_triples)[:200000], ent2input, rel2input, 
                        #                          args.max_seq_length, lp_processor.num_entity,
                        #                           lp_processor.num_relation, 0, 'negative')
                        #print(head_ds.shape)
                        #print(rel2input)
                        #print(len(tail_ds))
                        task_total_dataset["lp"] = ConcatDataset([rel_ds,rel_agg])
                        print("starts")
                        torch.save(task_total_dataset["lp"], train_lp_bin_path)
            #sys.exit()
            train_dataloader["lp"] = DataLoader(task_total_dataset["lp"],
                                             batch_size=args.train_batch_size,
                                             shuffle=False,   
                                             sampler= SequentialSampler(task_total_dataset["lp"]),
                                             collate_fn=BertTrainDataset.collate_fn_bert,
                                             pin_memory=False,
                                             num_workers=16)
            #train_dataloader["lp"] = lp_train_dataloader
            #del task_total_dataset["lp"]
            gc.collect()
            logger.info("  [Link Prediction] Num examples = %d", len(train_dataloader["lp"]))
        if "rp" in task_list:
            # load relation prediction data
            train_bin_path = os.path.join(args.data_dir, '_train-rp_0.pt')
            rp_processor.set_ent2id(lp_processor.ent2id)
            rp_processor.set_rel2id(lp_processor.rel2id)
            if os.path.exists(train_bin_path):
                task_total_dataset["rp"] = torch.load(train_bin_path)
                #sampler = RandomSampler(torch.load(train_bin_path))
                logger.info("load %s" % train_bin_path)
            else:
                print("get_train_triples")
                rng = np.random.default_rng()
                print(rp_label_list)
                #rp_train_triples = rp_processor.get_train_triples(args.data_dir)[:74070107//2]
                _train_triples = rp_processor.get_train_triples(args.data_dir)#[74070107//2:]
                rp_train_triples = _train_triples[:74070107//20]#_rp_train_triples[:74070107//2]
                #rp_train_features = rp_convert_examples_to_features(
                #    rp_train_examples, rp_label_list, args.max_seq_length, tokenizer)
                #logger.info("  [Relation Prediction] Num examples = %d", len(rp_train_examples))
                #all_input_ids = torch.tensor([f.input_ids for f in rp_train_features], dtype=torch.long)
                #all_input_mask = torch.tensor([f.input_mask for f in rp_train_features], dtype=torch.long)
                #all_segment_ids = torch.tensor([f.segment_ids for f in rp_train_features], dtype=torch.long)
                #all_label_ids = torch.tensor([f.label_id for f in rp_train_features], dtype=torch.long)
                #rp_train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
                print("BertTrainDataset2")
                ds = BertTrainDataset2(rp_label_list,
                                       rng.choice(rp_train_triples, size=len(rp_train_triples)//1, replace=False)
                                      , ent2input, rel2input, 
                                      args.max_seq_length, lp_processor.num_entity,lp_processor.num_relation)
                #ds2 = BertTrainDataset2(rp_label_list,
                #                        rng.choice(rp_train_triples, size=len(rp_train_triples)//1, replace=False)
                #                       , ent2input, rel2input, 
                #                      args.max_seq_length, lp_processor.num_entity,lp_processor.num_relation)
                #ds3 = BertTrainDataset2(rp_label_list,
                #                        rng.choice(rp_train_triples, size=len(rp_train_triples)//1, replace=False)
                #                       , ent2input, rel2input, 
                #                      args.max_seq_length, lp_processor.num_entity,lp_processor.num_relation)
                task_total_dataset["rp"] = ConcatDataset([ds])
                print("save")
                torch.save(task_total_dataset["rp"], train_bin_path)
                
                train_bin_path = os.path.join(args.data_dir, '_train-rp_1.pt')
                if os.path.exists(train_bin_path):
                    print("continue")
                    #task_total_dataset["rp"] = torch.load(train_bin_path)
                    #sampler = RandomSampler(torch.load(train_bin_path))
                    #logger.info("load %s" % train_bin_path)
                else:
                    print("get_train_triples")
                    #rp_train_triples = rp_processor.get_train_triples(args.data_dir)[:74070107//2]
                    #_rp_train_triples = rp_processor.get_train_triples(args.data_dir)#[74070107//2:]
                    rp_train_triples = _train_triples[74070107//2:]#_rp_train_triples[74070107//2:]
                    #rp_train_features = rp_convert_examples_to_features(
                    #    rp_train_examples, rp_label_list, args.max_seq_length, tokenizer)
                    #logger.info("  [Relation Prediction] Num examples = %d", len(rp_train_examples))
                    #all_input_ids = torch.tensor([f.input_ids for f in rp_train_features], dtype=torch.long)
                    #all_input_mask = torch.tensor([f.input_mask for f in rp_train_features], dtype=torch.long)
                    #all_segment_ids = torch.tensor([f.segment_ids for f in rp_train_features], dtype=torch.long)
                    #all_label_ids = torch.tensor([f.label_id for f in rp_train_features], dtype=torch.long)
                    #rp_train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
                    print("BertTrainDataset2")
                    ds = BertTrainDataset2(rp_label_list,
                                           rng.choice(rp_train_triples, size=len(rp_train_triples)//1, replace=False)
                                           , ent2input, rel2input, 
                                          args.max_seq_length, lp_processor.num_entity,lp_processor.num_relation)
                    #task_total_dataset["rp"] = ConcatDataset([ds])
                    print("save")
                    torch.save(ConcatDataset([ds]), train_bin_path)    
            
            #sampler = RandomSampler(task_total_dataset["rp"])
            #sys.exit()
            train_dataloader["rp"] = DataLoader(task_total_dataset["rp"],
                                                batch_size=args.train_batch_size, 
                                                sampler=SequentialSampler(task_total_dataset["rp"]),
                                                collate_fn=BertTrainDataset2.collate_fn_bert,
                                                shuffle=False,
                                                pin_memory=False,
                                                num_workers=11)
            #del task_total_dataset["rp"]
            #del ds
            #gc.collect()
            logger.info("  [Relation Prediction] Num examples = %d", len(train_dataloader["rp"]))
        if "rr" in task_list:
            #import random
            train_rr_bin_path = os.path.join(args.data_dir, '_train-rr_1.pt')
            rr_processor.set_ent2id(lp_processor.ent2id)
            rr_processor.set_rel2id(lp_processor.rel2id)
            if os.path.exists(train_rr_bin_path):
                #print("continue")
                task_total_dataset["rr"] = torch.load(train_rr_bin_path)
                logger.info("load %s" % train_rr_bin_path)
            else:
                #train_triples = rr_processor.get_train_triples(args.data_dir)[:83135614//2]
                train_triples__2 = rr_processor.get_train_triples(args.data_dir)[:83135614//2]
                _train_triples_ = rr_processor.get_train_triples_(args.data_dir)#[:74070107//2]
                train_triples_ = _train_triples_[74070107//2:]
                #train_triples_ = rr_processor.get_train_triples_(args.data_dir)[74070107//2:]

                #print("here ___ ",train_triples[0])
                #print("here ___ ",train_triples[0],ent2input[train_triples[0][0]],ent2input[train_triples[0][0]])
                #head_ds_ = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                #lp_processor.num_entity,lp_processor.num_relation, -1, 'head-batch')
                #print(len(head_ds_))
                #random.sample(train_triples_[len(train_triples_)//2:], len(train_triples_)//3)
                rng = np.random.default_rng()
                #result = rng.choice(train_triples_[len(train_triples_)//2:], size=len(train_triples_)//4, replace=False)
                head_ds = BertTrainDataset(np.random.permutation(train_triples_)
                                           ,ent2input,rel2input, args.max_seq_length, 
                                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                head_ds_ = BertTrainDataset(np.random.permutation(train_triples__2)
                                           ,ent2input,rel2input, args.max_seq_length, 
                                           lp_processor.num_entity,lp_processor.num_relation, -1, 'tail-batch')
                #print("hi")                            
                #tail_ds = BertTrainDataset(rng.choice(train_triples_, size=len(_train_triples_)//1, replace=False)
                #                           ,ent2input,rel2input, args.max_seq_length, 
                #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'tail-batch')
                #tail_ds = BertTrainDataset(train_triples_,ent2input,rel2input, args.max_seq_length, 
                #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'tail-batch')
                #head_ds = BertTrainDataset(random.sample(train_triples_[:len(train_triples_)//2], len(train_triples_)//4)
                #                           ,ent2input,rel2input, args.max_seq_length,
                #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                #head_ds = BertTrainDataset(rng.choice(train_triples_, size=len(_train_triples_)//1, replace=False)
                #                           ,ent2input,rel2input, args.max_seq_length,
                #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                #head_ds = BertTrainDataset(train_triples_,ent2input,rel2input, args.max_seq_length,
                #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                #print(len(tail_ds),len(tail_ds))
                task_total_dataset["rr"] = ConcatDataset([head_ds])
                torch.save(task_total_dataset["rr"], train_rr_bin_path)
                #sys.exit()
           
                #train_rr_bin_path = os.path.join(args.data_dir, '_train-rr_2.pt')
                if os.path.exists(train_rr_bin_path):
                        print("continue")
                        #task_total_dataset["rr"] = torch.load(train_rr_bin_path)
                        #logger.info("load %s" % train_rr_bin_path)
                else:
                        #train_triples = rr_processor.get_train_triples(args.data_dir)[:83135614//2]
                        train_triples__2 = rr_processor.get_train_triples(args.data_dir)[83135614//2:]
                        #_train_triples_ = rr_processor.get_train_triples_(args.data_dir)#[:74070107//2]
                        #_train_triples_ = rr_processor.get_train_triples_(args.data_dir)
                        train_triples_ = _train_triples_[74070107//2:]
                        #train_triples_ = rr_processor.get_train_triples_(args.data_dir)[74070107//2:]

                        #print("here ___ ",train_triples[0])
                        #print("here ___ ",train_triples[0],ent2input[train_triples[0][0]],ent2input[train_triples[0][0]])
                        #head_ds_ = BertTrainDataset(train_triples, ent2input, rel2input, args.max_seq_length, 
                        #lp_processor.num_entity,lp_processor.num_relation, -1, 'head-batch')
                        #print(len(head_ds_))
                        #random.sample(train_triples_[len(train_triples_)//2:], len(train_triples_)//3)
                        rng = np.random.default_rng()
                        #result = rng.choice(train_triples_[len(train_triples_)//2:], size=len(train_triples_)//4, replace=False)
                        #tail_ds = BertTrainDataset(random.sample(train_triples_[len(train_triples_)//2:], len(train_triples_)//4)
                        #                           ,ent2input,rel2input, args.max_seq_length, 
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'tail-batch')
                        head_ds = BertTrainDataset(np.random.permutation(train_triples_)
                                                   ,ent2input,rel2input, args.max_seq_length, 
                                                   lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                        head_ds_ = BertTrainDataset(np.random.permutation(train_triples__2)
                                                   ,ent2input,rel2input, args.max_seq_length, 
                                                   lp_processor.num_entity,lp_processor.num_relation, -1, 'tail-batch')
                        #head_ds = BertTrainDataset(random.sample(train_triples_[:len(train_triples_)//2], len(train_triples_)//4)
                        #                           ,ent2input,rel2input, args.max_seq_length,
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                        #head_ds = BertTrainDataset(rng.choice(train_triples_, 
                        #                            size=len(_train_triples_)//4, replace=False)
                        #                           ,ent2input,rel2input, args.max_seq_length,
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                        print(len(head_ds),len(head_ds))
                        task_total_dataset["rr"] = ConcatDataset([head_ds])
                        torch.save(task_total_dataset["rr"], train_rr_bin_path)
                        #sys.exit()

                #train_rr_bin_path = os.path.join(args.data_dir, 'train-rr_2.pt')
                #sys.exit()
                if os.path.exists(train_rr_bin_path):
                        #task_total_dataset["rr"] = torch.load(train_rr_bin_path)
                        logger.info("load %s" % train_rr_bin_path)
                else:
                        _train_triples = rr_processor.get_train_triples(args.data_dir)#[:83135614//2]
                        train_triples = _train_triples[:83135614//2]
                        #train_triples = rr_processor.get_train_triples(args.data_dir)[83135614//2:]
                        #_train_triples_ = rr_processor.get_train_triples_(args.data_dir)#[:74070107//2]
                        #train_triples_ = _train_triples_(args.data_dir)[74070107//2:]
                        #train_triples_ = rr_processor.get_train_triples_(args.data_dir)[74070107//2:]

                        #print("here ___ ",train_triples[0])
                        #print("here ___ ",train_triples[0],ent2input[train_triples[0][0]],ent2input[train_triples[0][0]])
                        head_ds_ = BertTrainDataset(np.random.permutation(train_triples), ent2input, rel2input, args.max_seq_length, 
                                                    lp_processor.num_entity,lp_processor.num_relation, -1, 'head-batch')
                        #print(len(head_ds_))
                        #random.sample(train_triples_[len(train_triples_)//2:], len(train_triples_)//3)
                        rng = np.random.default_rng()
                        #result = rng.choice(train_triples_[len(train_triples_)//2:], size=len(train_triples_)//4, replace=False)
                        #tail_ds = BertTrainDataset(random.sample(train_triples_[len(train_triples_)//2:], len(train_triples_)//4)
                        #                           ,ent2input,rel2input, args.max_seq_length, 
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'tail-batch')
                        #tail_ds = BertTrainDataset(rng.choice(train_triples_[len(train_triples_)//2:], 
                        #                           size=len(train_triples_)//4, replace=False)
                        #                           ,ent2input,rel2input, args.max_seq_length, 
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'tail-batch')
                        #head_ds = BertTrainDataset(random.sample(train_triples_[:len(train_triples_)//2], len(train_triples_)//4)
                        #                           ,ent2input,rel2input, args.max_seq_length,
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                        #head_ds = BertTrainDataset(rng.choice(train_triples_[:len(train_triples_)//2], 
                        #                            size=len(train_triples_)//4, replace=False)
                        #                           ,ent2input,rel2input, args.max_seq_length,
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                        print(len(head_ds_),len(head_ds_))
                        task_total_dataset["rr"] = ConcatDataset([head_ds_])
                        torch.save(task_total_dataset["rr"], train_rr_bin_path)

                #train_rr_bin_path = os.path.join(args.data_dir, 'train-rr_3.pt')
                if os.path.exists(train_rr_bin_path):
                        task_total_dataset["rr"] = torch.load(train_rr_bin_path)
                        logger.info("load %s" % train_rr_bin_path)
                else:
                        #_train_triples = rr_processor.get_train_triples(args.data_dir)#[:83135614//2]
                        train_triples = _train_triples[83135614//2:]
                        #train_triples = rr_processor.get_train_triples(args.data_dir)[83135614//2:]
                        #_train_triples_ = rr_processor.get_train_triples_(args.data_dir)#[:74070107//2]
                        #train_triples_ = _train_triples_(args.data_dir)[74070107//2:]
                        #train_triples_ = rr_processor.get_train_triples_(args.data_dir)[74070107//2:]

                        #print("here ___ ",train_triples[0])
                        #print("here ___ ",train_triples[0],ent2input[train_triples[0][0]],ent2input[train_triples[0][0]])
                        head_ds_ = BertTrainDataset(np.random.permutation(train_triples), ent2input, rel2input, args.max_seq_length, 
                                                    lp_processor.num_entity,lp_processor.num_relation, -1, 'head-batch')
                        #print(len(head_ds_))
                        #random.sample(train_triples_[len(train_triples_)//2:], len(train_triples_)//3)
                        rng = np.random.default_rng()
                        #result = rng.choice(train_triples_[len(train_triples_)//2:], size=len(train_triples_)//4, replace=False)
                        #tail_ds = BertTrainDataset(random.sample(train_triples_[len(train_triples_)//2:], len(train_triples_)//4)
                        #                           ,ent2input,rel2input, args.max_seq_length, 
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'tail-batch')
                        #tail_ds = BertTrainDataset(rng.choice(train_triples_[len(train_triples_)//2:], 
                        #                           size=len(train_triples_)//4, replace=False)
                        #                           ,ent2input,rel2input, args.max_seq_length, 
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'tail-batch')
                        #head_ds = BertTrainDataset(random.sample(train_triples_[:len(train_triples_)//2], len(train_triples_)//4)
                        #                           ,ent2input,rel2input, args.max_seq_length,
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                        #head_ds = BertTrainDataset(rng.choice(train_triples_[:len(train_triples_)//2], 
                        #                            size=len(train_triples_)//4, replace=False)
                        #                           ,ent2input,rel2input, args.max_seq_length,
                        #                           lp_processor.num_entity,lp_processor.num_relation, 1, 'head-batch')
                        print(len(head_ds_),len(head_ds_))
                        task_total_dataset["rr"] = ConcatDataset([head_ds_])
                        torch.save(task_total_dataset["rr"], train_rr_bin_path)
            #sampler = RandomSampler(task_total_dataset["rr"])
            #sys.exit()
            train_dataloader["rr"] = DataLoader(task_total_dataset["rr"],
                                             batch_size=args.train_batch_size,  
                                             sampler=SequentialSampler(task_total_dataset["rr"]),
                                             collate_fn=BertTrainDataset.collate_fn_rr,
                                             shuffle=False,   
                                             pin_memory=False,
                                             num_workers=16)
            #train_dataloader["rr"] = rr_train_dataloader
            #del task_total_dataset["rr"]
            #del head_ds_
            #del head_ds
            #del tail_ds
            #gc.collect()
            logger.info("  [Margin Rank] Num examples = %d", len(train_dataloader["rr"]))
        
        #sys.exit()
        batch_nums = {task: len(train_dataloader[task]) for task in task_total_dataset}
        #task_total_batch_num = sum([batch_nums[k] for k in batch_nums])

        # set batch order
        order = order_selection([task for task in task_total_dataset], batch_nums)

        # set recoders
        loss_recoder = {task: .0 for task in task_total_dataset}

        if args.fp16 and _use_native_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        model.train()
        for k in trange(14,(14+int(args.num_train_epochs)), desc="Epoch"):
            #gc.collect()
            tr_loss = {task: .0 for task in task_total_dataset}
            nb_tr_examples = {task: 0 for task in task_total_dataset}
            nb_tr_steps = 0

            task_iterators = {task: iter(train_dataloader[task]) for task in train_dataloader}
            for step, (cur_batch_task, index) in enumerate(tqdm(order, desc="Iteration over multi-tasks")):
                
                batch_some_task = next(task_iterators[cur_batch_task])
                #optimizer.zero_grad()
                if cur_batch_task in ["lp", "rp"]:
                    if(cur_batch_task=="rp" or  cur_batch_task=="lp"):
                        #print("#####",cur_batch_task)
                        input_ids, segment_ids, input_mask, label_ids = tuple(t.to(device,non_blocking=True) for t in batch_some_task)
                        #print("input_ids",input_ids.shape,input_mask.shape)
                        #print(label_ids[:2],input_ids[:2])
                        logits = model(input_ids, token_type_ids=segment_ids,
                                       attention_mask=input_mask, task=cur_batch_task)
                        #print(logits.shape)
                        #sys.exit()
                        loss_fct = CrossEntropyLoss()
                        if cur_batch_task == "lp":
                            #print("shape lp",logits.shape,label_ids.shape,lp_num_labels)
                            loss = loss_fct(logits.view(-1, lp_num_labels), label_ids.view(-1))
                            loss = loss * 1
                        else:  # task == "rp"
                            #print("shape rp",logits.shape,label_ids.shape)
                            loss = loss_fct(logits.view(-1, rp_num_labels), label_ids.view(-1))
                            loss = loss * 1
                elif cur_batch_task in ["rf"]:
                    #print("#####",cur_batch_task)
                    batch = tuple(t.to(device,non_blocking=True) for t in batch_some_task)
                    input_ids1, seg1, mask1, input_ids2, seg2, mask2, input_ids3, seg3, mask3, label_ids = batch
                    logits1,logits2,e_logits1,e_logits2,e_logits3 = model(input_ids=input_ids1, token_type_ids=seg1, attention_mask=mask1,task=cur_batch_task,input_ids2=input_ids2, token_type_ids2=seg2,attention_mask2=mask2,input_ids3=input_ids3, token_type_ids3=seg3,attention_mask3=mask3)
                    #logits1, logits2,logits3 ,e_logits1,e_logits2,e_logits3 = model(input_ids=input_ids1, token_type_ids=seg1, attention_mask=mask1,task=cur_batch_task,input_ids2=input_ids2, token_type_ids2=seg2,attention_mask2=mask2,input_ids3=input_ids3, token_type_ids3=seg3,attention_mask3=mask3)
                    #loss_fct = nn.TripletMarginLoss(margin=args.margin, p=2)
                    loss_fct = MarginRankingLoss(margin=2*args.margin)
                    #loss_fct = TripletMarginWithDistanceLoss(swap=True) 
                    #print("##########",label_ids.shape,logits1.shape)
                    label_ids = label_ids.new_ones(label_ids.size()).detach()
                    #print("##########",label_ids.view(-1).shape,logits1.shape)
                    #loss = loss_fct(logits1, logits2,logits3)
                    loss = loss_fct(logits2, logits1, label_ids.view(-1))
                    #loss = loss_fct(anchor,positive, negative)
                    #print("##########",logits1[:3], logits2[:3], loss,loss*10)
                    
                    topo_loss = (1-e_logits1)+(1-e_logits2)+(1-e_logits3)
                    loss = loss + 2*topo_loss
                    loss = loss * 1
                    
                elif cur_batch_task in ["rr"]:
                    #print("#####",cur_batch_task)
                    batch = tuple(t.to(device,non_blocking=True) for t in batch_some_task)
                    input_ids1, seg1, mask1, input_ids2, seg2, mask2, label_ids = batch
                    logits1, logits2 = model(input_ids=input_ids1, token_type_ids=seg1, attention_mask=mask1,
                                             task=cur_batch_task,input_ids2=input_ids2, token_type_ids2=seg2,
                                             attention_mask2=mask2)
                    loss_fct = MarginRankingLoss(margin=args.margin)
                    #print("##########",label_ids.shape,logits1.shape)
                    label_ids = label_ids.new_ones(label_ids.size()).detach()
                    #print("##########",label_ids.view(-1).shape,logits1.shape)
                    loss = loss_fct(logits1, logits2, label_ids.view(-1))
                    #print("##########",logits1[:3], logits2[:3], loss,loss*10)
                    loss = loss * 1
                elif cur_batch_task in ["sym"]:
                    #print("#####",cur_batch_task)
                    batch = tuple(t.to(device,non_blocking=True) for t in batch_some_task)
                    input_ids1, seg1, mask1, input_ids2, seg2, mask2, label_ids = batch
                    logits1, logits2 = model(input_ids=input_ids1, token_type_ids=seg1, attention_mask=mask1,
                                             task=cur_batch_task,input_ids2=input_ids2, token_type_ids2=seg2,
                                             attention_mask2=mask2)
                    #pred = torch.eq(torch.argmax(logits1, dim=1),torch.argmax(logits2, dim=1)).long()
                    #loss_fct = MarginRankingLoss(margin=args.margin)
                    #print("##########",label_ids.shape,pred.shape)
                    #print(torch.eq(pred,label_ids).long())
                    #print(torch.sum(torch.eq(pred,label_ids).long()))
                    label_ids = label_ids.long().detach()
                    labels_one = torch.ones(len(label_ids),dtype=torch.long).to(device,non_blocking=True).detach()
                    #xor = torch.logical_xor(pred.long(), label_ids).long()
                    #print(pred[:20])
                    #print(label_ids[:20])
                    #print(xor[:20])
                    loss_fct = CrossEntropyLoss(reduction="mean")
                    loss_1 = loss_fct(logits1.view(-1, lp_num_labels),labels_one.view(-1))
                    loss_2 = loss_fct(logits2.view(-1, lp_num_labels),label_ids.view(-1))
                    loss = loss_1+loss_2
                    #loss = torch.add(torch.mean(torch.mul(torch.add(loss_1,loss_2),xor)),
                    #                 torch.add(torch.mean(loss_1),torch.mean(loss_2)))
                    #print(torch.mul(torch.add(loss_1[:10],loss_2[:10]),xor[:10]))
                    #loss = Variable(torch.div(torch.sub(pred.shape[0],torch.sum(torch.eq(pred,label_ids).long())),pred.shape[0]), requires_grad=True)
                    #print("##########",label_ids.view(-1).shape,logits1.shape)
                    #loss_1 = loss_fct(logits1, logits2, label_ids.view(-1))
                    #print("##########",logits1[:3], logits2[:3], loss,loss*10)
                    loss = loss * 0.5
                    
                else:
                    raise TypeError
                    
                #if(cur_batch_task=="rp" or  cur_batch_task=="lp" or  cur_batch_task=="rr" or  cur_batch_task=="rf"):
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                optimizer.zero_grad()

                #loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                #optimizer.step()

                # backward
                if args.fp16 and _use_native_amp:
                    scaler.scale(loss).backward()
                elif args.fp16 and _use_apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # clip grad norm
                if args.fp16 and _use_native_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                elif args.fp16 and _use_apex:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # step
                if args.fp16 and _use_native_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                #print(loss.item(),loss_recoder[cur_batch_task])

                tr_loss[cur_batch_task] += loss.item()
                loss_recoder[cur_batch_task] += loss.item()
                nb_tr_examples[cur_batch_task] += label_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    
                #break
                #if step%5000==0 and step>0 :
                #    # Save a trained model, configuration and tokenizer
                #    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                #    # If we save using the predefined names, we can load using `from_pretrained`
                #    output_model_file = os.path.join(args.output_dir+str(k+1), WEIGHTS_NAME)
                #    torch.save(model_to_save.state_dict(), output_model_file)
            
                summary.add_scalar('%s_training_loss' % cur_batch_task, loss.item(), global_step)
            # end of epoch
            # eval on dev set
            
            if args.do_train  and k>-1:
                # Save a trained model, configuration and tokenizer
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                # If we save using the predefined names, we can load using `from_pretrained`
                output_model_file = os.path.join(args.output_dir+str(k+0), WEIGHTS_NAME)
                output_config_file = os.path.join(args.output_dir+str(k+0), CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir+str(k+0))
                
            #gc.collect()
            print("eval")
            sys.exit()
            #break
            eval_result = {}
            #eval_result["lp"] = dev_eval(args, device, global_step, lp_label_list, lp_num_labels, lp_processor, model,
            #                              rel_tokenizer,tokenizer, lp_convert_examples_to_features, "lp")
            #eval_result["rp"] = dev_eval(args, device, global_step, rp_label_list, rp_num_labels, rp_processor, model,
            #                              rel_tokenizer,tokenizer, rp_convert_examples_to_features, "rp")

            #for task in task_total_dataset:
            #    logger.info("[%s] Training loss: %.6f, Training examples: %d" % (task, tr_loss[task], nb_tr_examples[task]))
            #    if(task!="rr"):
            #        for key in sorted(eval_result[task].keys()):
            #            if(key!="rr"):
            #                logger.info("%s , %s = %s", k,key, str(eval_result[task][key]))

            #output_eval_file = os.path.join(args.output_dir+str(k+0), "test_results.txt")
            #with open(output_eval_file, "w") as writer:
            #    for task in task_total_dataset:
            #        if(task!="rr"):
            #            for key in sorted(eval_result[task].keys()):
            #                if(key!="rr"):
            #                    writer.write(str(k+0)+","+str(key)+","+str(eval_result[task][key])+"\n")
            #score = 1#eval_result["lp"]["acc"]
            #model_to_save = model.module if hasattr(model, 'module') else model
            #output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + "_%d" % (k+1))
            #torch.save(model_to_save.state_dict(), output_model_file)
            
        # end of whole training


    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.eval_task == "lp":
        # evaluate Link Prediction task 
        
        #test_triples = lp_processor.get_test_triples(args.data_dir, entity=True)#15345518:15345918
        #print(test_triples)
        #sys.exit()
        test_lp_bin_path = os.path.join(args.data_dir, 'test-lp.pt') #10
        
        task_total_dataset = dict()
        test_dataloader = {}
        tokenizer = FlaubertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=args.do_lower_case)

        rel_tokenizer = FlaubertTokenizer.from_pretrained(args.rel_tokenizer, do_lower_case=args.do_lower_case)
        
        """if os.path.exists(test_lp_bin_path):
                task_total_dataset["lp"] = torch.load(test_lp_bin_path)
                logger.info("load %s" % test_lp_bin_path)
        else:
                ent2input = lp_processor.get_ent2input(args.data_dir, rel_tokenizer, tokenizer)
                rel2input = lp_processor.get_rel2input(args.data_dir, rel_tokenizer)
                print("starts")
                test_triples = lp_processor.get_test_triples(args.data_dir)
                print("starts")
                #neg_test_triples = lp_processor.get_test_neg_triples(args.data_dir)
                #neg_train_triples = lp_processor.get_neg_train_triples(args.data_dir)
                #agg_test_triples = lp_processor.get_test_agg_triples(args.data_dir)
                print("starts")
                ds = BertTrainDataset(test_triples, ent2input, rel2input, args.max_seq_length, lp_processor.num_entity,
                                           lp_processor.num_relation, 2, 'positive')
                #agg = BertTrainDataset(agg_test_triples, ent2input, rel2input, args.max_seq_length, 
                #                            lp_processor.num_entity,lp_processor.num_relation, 2, 'positive')
                #neg_ds = BertTrainDataset(neg_test_triples, ent2input, rel2input, 
                #                          args.max_seq_length, lp_processor.num_entity,
                #                           lp_processor.num_relation, 0, 'negative')
                #neg_ds_ = BertTrainDataset(neg_train_triples, ent2input, rel2input, 
                #                          args.max_seq_length, lp_processor.num_entity,
                #                           lp_processor.num_relation, 0, 'negative')

                task_total_dataset["lp"] = ConcatDataset([ds])
                print("starts")
                torch.save(task_total_dataset["lp"], test_lp_bin_path)    
                #sys.exit()
                
        test_dataloader["lp"] = DataLoader(task_total_dataset["lp"],
                                             batch_size=args.eval_batch_size,
                                             shuffle=False,   
                                             sampler= SequentialSampler(task_total_dataset["lp"]),
                                             collate_fn=BertTrainDataset.collate_fn_bert,
                                             pin_memory=False,
                                             num_workers=5)"""
        
        train_triples_ = lp_processor.get_train_triples(args.data_dir, entity=True)
        train_triples__ = []
        #train_triples = np.empty([7102101,3],dtype=tuple) 
        c_i=0
        for tr in train_triples_:
            #print(tr)
            #if c_i==10000000 :
            #    print(len(train_triples),c_i,train_triples[len(train_triples)-1])
            #if c_i%500000==0 :
            #    print(len(train_triples),c_i,train_triples[len(train_triples)-1])
                
            if (tr[1] == "r_associated"):
                #print(tr)
                #train_triples[c_i]=(tr[0], tr[1], tr[2])
                train_triples__.append((tr[0], tr[1], tr[2]))
                c_i+=1
        
        test_triples_ = lp_processor.get_test_triples(args.data_dir, entity=True)
        test_triples__ = []
        #train_triples = np.empty([7102101,3],dtype=tuple) 
        c_i=0
        for tr in test_triples_:
            #print(tr)
            #if c_i==10000000 :
            #    print(len(train_triples),c_i,train_triples[len(train_triples)-1])
            #if c_i%500000==0 :
            #    print(len(train_triples),c_i,train_triples[len(train_triples)-1])
                
            if (tr[1] == "r_associated"):
                #print(tr)
                #train_triples[c_i]=(tr[0], tr[1], tr[2])
                test_triples__.append((tr[0], tr[1], tr[2]))
                c_i+=1
            
        #train_triples = lp_processor.get_train_triples(args.data_dir, entity=True)
        #test_triples = lp_processor.get_test_triples(args.data_dir, entity=True)#[:100]#[15283568:15283968]
        #print(test_triples)
        all_triples = train_triples__[:1000]+test_triples__[:1000]

        all_triples_str_set = list()#np.array([])
        for triple in all_triples:
            #print(triple)
            triple_str = '\t'.join(triple)
            all_triples_str_set.append('\t'.join(triple))#=np.append(all_triples_str_set,'\t'.join(triple))
        
        config = FlaubertConfig.from_pretrained(args.output_dir)
        #setattr(config, "lp_num_labels", lp_num_labels)
        #setattr(config, "rp_num_labels", rp_num_labels)
        model = BertForSequenceClassification.from_pretrained(args.output_dir,config=config)
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model,device_ids=[0, 1, 2, 3])
        model.to(device)
        print(device)
        logger.info("device: {} n_gpu: {}, distributed evaluation: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
        
        #eval_examples = lp_processor.get_test_agg_examples(args.data_dir)#[:50000]
        #eval_features = lp_convert_examples_to_features(
        #    eval_examples, lp_label_list, args.max_seq_length, rel_tokenizer,tokenizer)
        #logger.info("***** Running Prediction *****")
        #logger.info("  Num examples = %d", len(task_total_dataset["lp"]))
        #logger.info("  Batch size = %d", args.eval_batch_size)
        #all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        #all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        #all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        #all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        #eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        #eval_sampler = SequentialSampler(eval_data)
        #eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # Load a trained model and vocabulary that you have fine-tuned
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        all_label_ids = []
        """batch_nums = {task: len(test_dataloader[task]) for task in task_total_dataset}
        order = order_selection([task for task in task_total_dataset], batch_nums)
        task_iterators = {task: iter(test_dataloader[task]) for task in test_dataloader}
        #for step, (cur_batch_task, index) in enumerate(tqdm(order, desc="Iteration over batchs")):
        #    batch_some_task = next(task_iterators[cur_batch_task])
        #    input_ids, segment_ids, input_mask, label_ids = tuple(t.to(device) for t in batch_some_task)
        #    all_label_ids.extend(label_ids.cpu())
        #print(len(all_label_ids))
        #input_ids, segment_ids, input_mask, label_ids = input_ids[0:100], segment_ids[0:100], input_mask[0:100], label_ids[0:100]
        #all_label_ids=all_label_ids[0:100]

        for step, (cur_batch_task, index) in enumerate(tqdm(order, desc="Iteration over batchs")):
            batch_some_task = next(task_iterators[cur_batch_task])
            input_ids, segment_ids, input_mask, label_ids = tuple(t.to(device) for t in batch_some_task)
            all_label_ids.extend(label_ids.cpu())
        #    #print([e.item() for e in all_label_ids])
        
        #for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Testing"):
            input_ids = input_ids.to(device,non_blocking=True)
            input_mask = input_mask.to(device,non_blocking=True)
            segment_ids = segment_ids.to(device,non_blocking=True)
            label_ids = label_ids.to(device,non_blocking=True)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, task="lp")

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, lp_num_labels), label_ids.view(-1))
            #print(tmp_eval_loss)
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        #print(logits)
        preds = preds[0]
        #print(preds, preds.shape)
        #print(all_label_ids)
        #all_label_ids = all_label_ids.numpy()
        all_label_ids = np.array([e.item() for e in all_label_ids])
        preds = np.argmax(preds, axis=1)
        #print(preds[:50], all_label_ids[:50])
        result = compute_metrics("lp", preds, all_label_ids)

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step

        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info("Triple classification acc is : %.4f" % metrics.accuracy_score(all_label_ids, preds))"""
        
        ############################################################################
        # run link prediction
        ############################################################################
        
        print("run link prediction")
        gc.collect()
        ranks = []
        ranks_left = []
        ranks_right = []

        hits_left = []
        hits_right = []
        hits = []

        top_ten_hit_count = 0

        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])
        
        import pickle
        
        #ent2input = lp_processor.get_ent2input(args.data_dir, rel_tokenizer, tokenizer)
        #rel2input = lp_processor.get_rel2input(args.data_dir, rel_tokenizer)
        #pickle.dump(ent2input, open("/data/matmani/JDM/data/ent2input.p","wb"))
        #pickle.dump(rel2input, open("/data/matmani/JDM/data/rel2input.p","wb"))
        
        ent2input = pickle.load( open( "/data/matmani/JDM/data/ent2input.p", "rb" ) )
        rel2input = pickle.load( open( "/data/matmani/JDM/data/rel2input.p", "rb" ) )
        
        
        def _get_tensordataset(corrupt_list,ent2input,rel2input):
            
            ds = BertTrainDataset(lp_processor._get_triples_raw(corrupt_list), ent2input, rel2input, args.max_seq_length, 
                                  lp_processor.num_entity,lp_processor.num_relation, 2, 'negative')
            #tmp_examples = lp_processor._create_examples_q(corrupt_list, "test", args.data_dir)
            # print(len(tmp_examples))
            #tmp_features = lp_convert_examples_to_features(tmp_examples, lp_label_list, args.max_seq_length,
            #                                               rel_tokenizer,tokenizer,print_info=False)
            #all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            #all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
            #all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
            #all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)
            #return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            return ds
        entity2text = lp_processor.get_entity2text(args.data_dir)
        start, end = [int(x) for x in args.eval_range.split(":")]
        start, end = 0,906 #433,1239 #42093 # 0 145
        test_triples = test_triples__
        test = test_triples#[95:906]
        #test =  random.sample(test,len(test))
        print(start,end)
        for idx, test_triple in enumerate(tqdm(test), start):
            head = test_triple[0]
            relation = test_triple[1]
            tail = test_triple[2]
            print(idx,test_triple)

            head_corrupt_list = [test_triple]
            tail_corrupt_list = [test_triple]
            # 4000 is magic number,
            _entity_list = np.random.choice(entity_list, 4000, replace=False)
            for corrupt_ent in _entity_list:
                if corrupt_ent != head:
                    tmp_triple = [corrupt_ent, relation, tail]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                    #    #    # may be slow
                        head_corrupt_list.append(tmp_triple)
                if corrupt_ent != tail:
                    tmp_triple = [head, relation, corrupt_ent]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                    #    #    # may be slow
                        tail_corrupt_list.append(tmp_triple)
            
            #head_corrupt = _get_tensordataset(head_corrupt_list)
            #tail_corrupt = _get_tensordataset(tail_corrupt_list)

            #eval_data = head_corrupt
            #eval_sampler = SequentialSampler(eval_data)
            #print(eval_data)
            #eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            
            head_lp_bin_path = os.path.join(args.data_dir, 'head-lp'+str(idx)+'.pt')
            if os.path.exists(head_lp_bin_path):
                head_corrupt = torch.load(head_lp_bin_path)
                logger.info("load %s" % head_lp_bin_path)
            else:
                head_corrupt = _get_tensordataset(head_corrupt_list,ent2input,rel2input)
                torch.save(head_corrupt, head_lp_bin_path)
                
            test_dataloader["lp"] = DataLoader(head_corrupt,
                                         batch_size=args.eval_batch_size,
                                         shuffle=False,   
                                         sampler= SequentialSampler(head_corrupt),
                                         collate_fn=BertTrainDataset.collate_fn_bert,
                                         pin_memory=False,
                                         num_workers=5)
             
            model.eval()
            preds = []
            #for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            task_total_dataset["lp"] = head_corrupt
            batch_nums = {task: len(test_dataloader[task]) for task in task_total_dataset}
            order = order_selection([task for task in task_total_dataset], batch_nums)
            task_iterators = {task: iter(test_dataloader[task]) for task in test_dataloader}
            for step, (cur_batch_task, index) in enumerate(tqdm(order, desc="Iteration over batchs")):
                batch_some_task = next(task_iterators[cur_batch_task])
                input_ids, segment_ids, input_mask, label_ids = tuple(t.to(device) for t in batch_some_task)
                #if(len(input_ids)<2):
                #    continue
                #print("//////////////////////////////////////////////////////::")
                #print(step)
                input_ids = input_ids.to(device,non_blocking=True)
                input_mask = input_mask.to(device,non_blocking=True)
                segment_ids = segment_ids.to(device,non_blocking=True)
                # label_ids = label_ids.to(device)
                #print(input_ids is not None, input_ids, input_mask, segment_ids, label_ids)
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, task="lp")
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)
                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0)       

            preds = preds[0]
            # get the dimension corresponding to current label 1
            #print(preds, preds.shape)
            rel_values = preds[:, 1]#preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            #print(rel_values, rel_values.shape)
            argvalues1, argsort1 = torch.sort(rel_values, descending=True)
            #print(max_values)
            #print(argsort1)
            argsort1 = argsort1.cpu().numpy()
            argvalues1 = argvalues1.cpu().numpy()
            rank1 = np.where(argsort1 == 0)[0][0]

            ranks.append(rank1+1)
            ranks_left.append(rank1+1)
            if rank1 < 10:
                top_ten_hit_count += 1

            #eval_data = tail_corrupt
            #eval_sampler = SequentialSampler(eval_data)
            #eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=10)
            
            
            file_prefix_ =  "debug" + "_" + str(args.learning_rate) + \
                          "_" + str(args.max_seq_length) + "_" + str(args.num_train_epochs) + "_" + str(args.eval_range)

            """with open(os.path.join(args.output_dir, file_prefix_ + '_ranks.txt'), 'a') as f:
            
                print("====== head corrupt result =======")
                print("<%s, %s, %s>" % (entity2text[head], relation, entity2text[tail]))
                f.write("====== head corrupt result ======="+"\n")
                f.write("<%s, %s, %s>" % (entity2text[head], relation, entity2text[tail])+"\n")
                #print("rank: ", rank1+1)
                scores, words, entities = [], [], []
                for j in range(50):
                    __idx = argsort1[j]
                    h, _, _ = head_corrupt_list[__idx]
                    scores.append(str(round(argvalues1[j], 4)))
                    words.append(entity2text[h])
                    entities.append(h)
                for s,w in zip(scores, words):
                    print(s,w)
                    f.write(s+" "+w+"\n")"""
    
            tail_lp_bin_path = os.path.join(args.data_dir, 'tail-lp'+str(idx)+'.pt')
            if os.path.exists(tail_lp_bin_path):
                tail_corrupt = torch.load(tail_lp_bin_path)
                logger.info("load %s" % tail_lp_bin_path)
            else:
                tail_corrupt = _get_tensordataset(tail_corrupt_list,ent2input,rel2input)
                torch.save(tail_corrupt, tail_lp_bin_path)
            
            test_dataloader["lp"] = DataLoader(tail_corrupt,
                                         batch_size=args.eval_batch_size,
                                         shuffle=False,   
                                         sampler= SequentialSampler(tail_corrupt),
                                         collate_fn=BertTrainDataset.collate_fn_bert,
                                         pin_memory=False,
                                         num_workers=5)
            model.eval()
            preds = []
            #for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            gc.collect()
            task_total_dataset["lp"] = tail_corrupt
            batch_nums = {task: len(test_dataloader[task]) for task in task_total_dataset}
            order = order_selection([task for task in task_total_dataset], batch_nums)
            task_iterators = {task: iter(test_dataloader[task]) for task in test_dataloader}
            for step, (cur_batch_task, index) in enumerate(tqdm(order, desc="Iteration over batchs")):
                batch_some_task = next(task_iterators[cur_batch_task])
               
                input_ids, segment_ids, input_mask, label_ids = tuple(t.to(device) for t in batch_some_task)

                input_ids = input_ids.to(device,non_blocking=True)
                input_mask = input_mask.to(device,non_blocking=True)
                segment_ids = segment_ids.to(device,non_blocking=True)
                # label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, task="lp")
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)
                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0) 

            preds = preds[0]
            # get the dimension corresponding to current label 1
            rel_values = preds[:, 1]#preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            argvalues1, argsort1 = torch.sort(rel_values, descending=True)
            argsort1 = argsort1.cpu().numpy()
            argvalues1 = argvalues1.cpu().numpy()
            rank2 = np.where(argsort1 == 0)[0][0]
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)
            if rank2 < 10:
                top_ten_hit_count += 1
                
                
            """with open(os.path.join(args.output_dir, file_prefix_ + '_ranks.txt'), 'a') as f:

                print("====== tail corrupt result =======")
                print("<%s, %s, %s>" % (entity2text[head], relation, entity2text[tail]))
                f.write("====== tail corrupt result ======="+"\n")
                f.write("<%s, %s, %s>" % (entity2text[head], relation, entity2text[tail])+"\n")
                #print("rank: ", rank2+1)
                scores, words, entities = [], [], []
                for j in range(50):
                    __idx = argsort1[j]
                    _, _, t = tail_corrupt_list[__idx]
                    scores.append(str(round(argvalues1[j], 4)))
                    words.append(entity2text[t])
                    entities.append(t)
                for s,w in zip(scores, words):
                    print(s,w)
                    f.write(s+" "+w+"\n")"""
            
            logging.info('[%d/%d] left: %d, rgiht: %d, mean rank: %.4f, hit@10: %.4f' % (
                idx, len(test_triples), rank1, rank2, np.mean(ranks), (top_ten_hit_count * 1.0 / len(ranks))))

            file_prefix =  str(args.train_batch_size) + "_" + str(args.learning_rate) + \
                          "_" + str(args.max_seq_length) + "_" + str(args.num_train_epochs) + "_" + str(args.eval_range)

            with open(os.path.join(args.output_dir, file_prefix + '_ranks.txt'), 'a') as f:
                f.write('\t'.join([str(idx), str(rank1), str(rank2), str(np.mean(ranks)), str((top_ten_hit_count * 1.0 / len(ranks)))]) + '\n')
            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

        
        
        #file_prefix =  str(args.train_batch_size) + "_" + str(args.learning_rate) + \
        #                  "_" + str(args.max_seq_length) + "_" + str(args.num_train_epochs) + "_" + str(args.eval_range)

        for i in [0,2,9]:
            logger.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
            logger.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
            logger.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
        with open(os.path.join(args.output_dir, file_prefix + '_ranks.txt'), 'a') as f:
                f.write('\t'.join(["MR", str(np.mean(ranks)), str(np.mean(ranks_left)), str(np.mean(ranks_right))]) + '\n')
        with open(os.path.join(args.output_dir, file_prefix + '_ranks.txt'), 'a') as f:
                f.write('\t'.join(["MRR", str(np.mean(1./np.array(ranks_left))), str(np.mean(1./np.array(ranks_right))), str(np.mean(1./np.array(ranks)))]) + '\n')
        with open(os.path.join(args.output_dir, file_prefix + '_ranks.txt'), 'a') as f:
                f.write('\t'.join(["Hits@1", str(np.mean(hits_left[0])), str(np.mean(hits_right[0])), str(np.mean(hits[0]))]) + '\n') 
        with open(os.path.join(args.output_dir, file_prefix + '_ranks.txt'), 'a') as f:
                f.write('\t'.join(["Hits@3", str(np.mean(hits_left[2])), str(np.mean(hits_right[2])), str(np.mean(hits[2]))]) + '\n')
        with open(os.path.join(args.output_dir, file_prefix + '_ranks.txt'), 'a') as f:
                f.write('\t'.join(["Hits@10", str(np.mean(hits_left[9])), str(np.mean(hits_right[9])), str(np.mean(hits[9]))]) + '\n')
        logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
        logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
        logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
        
        
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.eval_task == "rr":
        #print(rp_label_list)
        rr_processor.set_ent2id(lp_processor.ent2id)
        rr_processor.set_rel2id(lp_processor.rel2id)
        test_rp_bin_path = os.path.join(args.data_dir, 'test-rr_s.pt') #10
        
        task_total_dataset = dict()
        test_dataloader = {}
        tokenizer = FlaubertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=args.do_lower_case)
        rel_tokenizer = FlaubertTokenizer.from_pretrained(args.rel_tokenizer, do_lower_case=args.do_lower_case)
        if os.path.exists(test_rp_bin_path):
                task_total_dataset["rr"] = torch.load(test_rp_bin_path)
                logger.info("load %s" % test_rp_bin_path)
        else:
                ent2input = lp_processor.get_ent2input(args.data_dir, rel_tokenizer, tokenizer)
                rel2input = lp_processor.get_rel2input(args.data_dir, rel_tokenizer)
                print("starts")
                test_triples = rr_processor.get_test_triples(args.data_dir)
                print("starts")
                print("starts")
                ds = BertTrainDataset(test_triples, ent2input, rel2input, args.max_seq_length,           lp_processor.num_entity,lp_processor.num_relation, -1, 'rr_positive')
                                

                task_total_dataset["rr"] = ConcatDataset([ds])
                print("starts")
                torch.save(task_total_dataset["rr"], test_rp_bin_path)    
                #sys.exit()
        test_dataloader["rr"] = DataLoader(task_total_dataset["rr"],
                                             batch_size=args.eval_batch_size,
                                             shuffle=False,   
                                             sampler= SequentialSampler(task_total_dataset["rr"]),
                                             collate_fn=BertTrainDataset.collate_fn_rr,
                                             pin_memory=False,
                                             num_workers=25)
        
        #train_triples = rp_processor.get_train_triples(args.data_dir, entity=True)
        #dev_triples = rp_processor.get_dev_triples(args.data_dir)
        test_triples = rp_processor.get_test_triples(args.data_dir, entity=True)#[:100000]
        all_triples = test_triples

        all_triples_str_set = []
        for triple in all_triples:
        #    #print(triple)
        #    triple_str = '\t'.join(triple)
            all_triples_str_set.append('\t'.join(triple))

        model = BertForSequenceClassification.from_pretrained(args.output_dir)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model,device_ids=[0, 1, 2, 3])
        model.to(device)
        print(device)
        logger.info("device: {} n_gpu: {}, distributed evaluation: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
        model.eval() 
        print("A")
        #"eval_examples = rp_processor.get_test_examples(args.data_dir)[:100000]
        print("B")
        #eval_features = rp_convert_examples_to_features(
        #    eval_examples, rp_label_list, args.max_seq_length, tokenizer)
        logger.info("***** [Relavance Ranking] Running Prediction *****")
        logger.info("  Num examples = %d", len(task_total_dataset["rr"]))
        logger.info("  Batch size = %d", args.eval_batch_size)
        #all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        #all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        #all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        #all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        #eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        #eval_sampler = SequentialSampler(eval_data)
        #eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=10)
        # Load a trained model and vocabulary that you have fine-tuned
        
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        all_label_ids = []
        batch_nums = {task: len(test_dataloader[task]) for task in task_total_dataset}
        order = order_selection([task for task in task_total_dataset], batch_nums)
        task_iterators = {task: iter(test_dataloader[task]) for task in test_dataloader}
        for step, (cur_batch_task, index) in enumerate(tqdm(order, desc="Iteration over batchs")):
            #if(len(preds)>1):
            #    break
            batch_some_task = next(task_iterators[cur_batch_task])
            input_ids1, seg1, mask1, input_ids2, seg2, mask2, label_ids = batch_some_task
           
        #for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
            input_ids1 = input_ids1.to(device,non_blocking=True)
            mask1 = mask1.to(device,non_blocking=True)
            seg1 = seg1.to(device,non_blocking=True)
            
            input_ids2 = input_ids2.to(device,non_blocking=True)
            mask2 = mask2.to(device,non_blocking=True)
            seg2 = seg2.to(device,non_blocking=True)
            
            label_ids = label_ids.to(device,non_blocking=True)

            with torch.no_grad():
                logits1, logits2 = model(input_ids=input_ids1, token_type_ids=seg1, attention_mask=mask1,
                                             task="rr",input_ids2=input_ids2, token_type_ids2=seg2,
                                             attention_mask2=mask2)

            loss_fct = MarginRankingLoss(margin=args.margin)
            label_ids = label_ids.new_ones(label_ids.size()).detach()
            all_label_ids.extend(label_ids)
            #print("##########",label_ids.view(-1).shape,logits1.shape)
            tmp_eval_loss = loss_fct(logits1, logits2, label_ids.view(-1))
            #print("##########",logits1[:3], logits2[:3], loss,loss*10)
            tmp_eval_loss = tmp_eval_loss * 1

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            logits = torch.cat((logits1, logits2), 1)
            preds.extend(logits.detach().cpu().numpy())
            #if len(preds) == 0:
            #    preds.append(logits.detach().cpu().numpy())
            #else:
            #    preds[0] = np.append(
            #        preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        print(preds[:20])
        preds = np.array(preds)#[0]
        print( preds.shape)
        print("here")
        #all_label_ids = all_label_ids.numpy()
        all_label_ids = np.array([e.item() for e in all_label_ids])
        
        preds = np.argmax(preds, axis=1)
        print( preds.shape,all_label_ids.shape)

        result = compute_metrics("lp", preds, all_label_ids)
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "rr_test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        # relation prediction, raw
        print("Relation prediction hits@1, raw...")
        print(metrics.accuracy_score(all_label_ids, preds))    

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.eval_task == "rp":
        #print(rp_label_list)
        rp_processor.set_ent2id(lp_processor.ent2id)
        rp_processor.set_rel2id(lp_processor.rel2id)
        test_rp_bin_path = os.path.join(args.data_dir, 'test-rp_s.pt') #10
        
        task_total_dataset = dict()
        test_dataloader = {}
        tokenizer = FlaubertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=args.do_lower_case)
        rel_tokenizer = FlaubertTokenizer.from_pretrained(args.rel_tokenizer, do_lower_case=args.do_lower_case)
        if os.path.exists(test_rp_bin_path):
                task_total_dataset["rp"] = torch.load(test_rp_bin_path)
                logger.info("load %s" % test_rp_bin_path)
        else:
                ent2input = lp_processor.get_ent2input(args.data_dir, rel_tokenizer, tokenizer)
                rel2input = lp_processor.get_rel2input(args.data_dir, rel_tokenizer)
                print("starts")
                test_triples = rp_processor.get_test_triples(args.data_dir)
                print("starts")
                print("starts")
                ds = BertTrainDataset2(rp_label_list,test_triples, ent2input, rel2input, args.max_seq_length, 
                                       lp_processor.num_entity,lp_processor.num_relation)

                task_total_dataset["rp"] = ConcatDataset([ds])
                print("starts")
                torch.save(task_total_dataset["rp"], test_rp_bin_path)    
                #sys.exit()
        test_dataloader["rp"] = DataLoader(task_total_dataset["rp"],
                                             batch_size=args.eval_batch_size,
                                             shuffle=False,   
                                             sampler= SequentialSampler(task_total_dataset["rp"]),
                                             collate_fn=BertTrainDataset2.collate_fn_bert,
                                             pin_memory=False,
                                             num_workers=25)
        
        #train_triples = rp_processor.get_train_triples(args.data_dir, entity=True)
        #dev_triples = rp_processor.get_dev_triples(args.data_dir)
        test_triples = rp_processor.get_test_triples(args.data_dir, entity=True)#[:100000]
        all_triples = test_triples

        all_triples_str_set = []
        for triple in all_triples:
        #    #print(triple)
        #    triple_str = '\t'.join(triple)
            all_triples_str_set.append('\t'.join(triple))

        model = BertForSequenceClassification.from_pretrained(args.output_dir)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model,device_ids=[0, 1, 2, 3])
        model.to(device)
        print(device)
        logger.info("device: {} n_gpu: {}, distributed evaluation: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
        model.eval() 
        print("A")
        #"eval_examples = rp_processor.get_test_examples(args.data_dir)[:100000]
        print("B")
        #eval_features = rp_convert_examples_to_features(
        #    eval_examples, rp_label_list, args.max_seq_length, tokenizer)
        logger.info("***** [Relation Prediction] Running Prediction *****")
        logger.info("  Num examples = %d", len(task_total_dataset["rp"]))
        logger.info("  Batch size = %d", args.eval_batch_size)
        #all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        #all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        #all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        #all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        #eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        #eval_sampler = SequentialSampler(eval_data)
        #eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=10)
        # Load a trained model and vocabulary that you have fine-tuned
        
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        all_label_ids = []
        batch_nums = {task: len(test_dataloader[task]) for task in task_total_dataset}
        order = order_selection([task for task in task_total_dataset], batch_nums)
        task_iterators = {task: iter(test_dataloader[task]) for task in test_dataloader}
        for step, (cur_batch_task, index) in enumerate(tqdm(order, desc="Iteration over batchs")):
            #if(len(preds)>1):
            #    break
            batch_some_task = next(task_iterators[cur_batch_task])
            input_ids, segment_ids, input_mask, label_ids = tuple(t.to(device) for t in batch_some_task)
            all_label_ids.extend(label_ids.cpu())
        #for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
            input_ids = input_ids.to(device,non_blocking=True)
            input_mask = input_mask.to(device,non_blocking=True)
            segment_ids = segment_ids.to(device,non_blocking=True)
            label_ids = label_ids.to(device,non_blocking=True)

            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask, task="rp")
                logits = outputs#[0]  # if labels is None, outputs = (logits, hidden_states, attentions)

            loss_fct = CrossEntropyLoss()
            #print(input_ids)
            #print(logits.view(-1, rp_num_labels),label_ids.view(-1))
            tmp_eval_loss = loss_fct(logits.view(-1, rp_num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            preds.extend(logits.detach().cpu().numpy())
            #if len(preds) == 0:
            #    preds.append(logits.detach().cpu().numpy())
            #else:
            #    preds[0] = np.append(
            #        preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        #print(preds)
        preds = np.array(preds)#[0]
        print( preds.shape)
        print("here")
        #all_label_ids = all_label_ids.numpy()
        all_label_ids = np.array([e.item() for e in all_label_ids])
        ranks = []
        filter_ranks = []
        hits = []
        hits_filter = []
        for i in range(10):
            hits.append([])
            hits_filter.append([])

        
        predictions = np.argmax(preds, axis=1)

        result = compute_metrics("lp", predictions, all_label_ids)
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "rp_test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        # relation prediction, raw
        print("Relation prediction hits@1, raw...")
        print(metrics.accuracy_score(all_label_ids, predictions))
        
        for i, pred in enumerate(preds):
            rel_values = torch.tensor(pred)
            _, argsort1 = torch.sort(rel_values, descending=True)
            argsort1 = argsort1.cpu().numpy()

            rank = np.where(argsort1 == all_label_ids[i])[0][0]
            # print(argsort1, all_label_ids[i], rank)
            ranks.append(rank + 1)
            test_triple = test_triples[i]
            filter_rank = rank
            for tmp_label_id in argsort1[:rank]:
                tmp_label = rp_label_list[tmp_label_id]
                tmp_triple = [test_triple[0], tmp_label, test_triple[2]]
                # print(tmp_triple)
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str in all_triples_str_set:
                    filter_rank -= 1
            filter_ranks.append(filter_rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

                if filter_rank <= hits_level:
                    hits_filter[hits_level].append(1.0)
                else:
                    hits_filter[hits_level].append(0.0)

        print("Raw mean rank: ", np.mean(ranks))
        print("Filtered mean rank: ", np.mean(filter_ranks))
        for i in [0, 2, 9]:
            print('Raw Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
            print('hits_filter Hits @{0}: {1}'.format(i + 1, np.mean(hits_filter[i])))
        preds = np.argmax(preds, axis=1)

        result = compute_metrics("lp", preds, all_label_ids)
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "rp_test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        # relation prediction, raw
        print("Relation prediction hits@1, raw...")
        print(metrics.accuracy_score(all_label_ids, preds))

    if args.do_debug and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.eval_task == "lp":
        ############################################################################
        # DEBUGGING!!
        ############################################################################
        #train_triples = lp_processor.get_train_triples(args.data_dir,entity=True)
        #dev_triples = lp_processor.get_dev_triples(args.data_dir)
        test_triples = lp_processor.get_test_triples(args.data_dir,entity=True)
        all_triples =  test_triples

        all_triples_str_set = []
        for triple in all_triples:
        #    #print(triple)
        #    triple_str = '\t'.join(triple)
            all_triples_str_set.append('\t'.join(triple))

        entity2text = lp_processor.get_entity2text(args.data_dir)

        ranks = []
        ranks_left = []
        ranks_right = []

        hits_left = []
        hits_right = []
        hits = []

        top_ten_hit_count = 0

        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])
            
        model = BertForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = FlaubertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=args.do_lower_case)
        rel_tokenizer = FlaubertTokenizer.from_pretrained(args.rel_tokenizer, do_lower_case=args.do_lower_case)    

        def _get_tensordataset(corrupt_list):
            tmp_examples = lp_processor._create_examples_q(corrupt_list, "test", args.data_dir)
            tmp_features = lp_convert_examples_to_features(tmp_examples, lp_label_list, args.max_seq_length, 
                                                           rel_tokenizer,tokenizer,print_info=False)
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
            return TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        test_triple = test_triples[args.debug_index]

        head, relation, tail = test_triple

        head_corrupt_list = [test_triple]
        tail_corrupt_list = [test_triple]
        _entity_list = np.random.choice(entity_list, 40000, replace=False)

        for corrupt_ent in _entity_list:
            if corrupt_ent != head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    head_corrupt_list.append(tmp_triple)
            if corrupt_ent != tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    tail_corrupt_list.append(tmp_triple)
        head_corrupt = _get_tensordataset(head_corrupt_list)
        tail_corrupt = _get_tensordataset(tail_corrupt_list)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model,device_ids=[0, 1, 2, 3])
        model.to(device)
        print(device)
        logger.info("device: {} n_gpu: {}, distributed evaluation: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
        model.eval()

        eval_data = head_corrupt
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        preds = []
        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, task="lp")
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)
            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0)

        preds = preds[0]
        # get the dimension corresponding to current label 1
        # print(preds, preds.shape)
        # rel_values = preds[:, all_label_ids[0]]
        rel_values = preds[:, 1]
        rel_values = torch.tensor(rel_values)
        # print(rel_values, rel_values.shape)
        argvalues, argsort1 = torch.sort(rel_values, descending=True)
        # print(max_values)
        # print(argsort1)
        argvalues = argvalues.cpu().numpy()
        argsort1 = argsort1.cpu().numpy()
        rank1 = np.where(argsort1 == 0)[0][0]

        ranks.append(rank1 + 1)
        ranks_left.append(rank1 + 1)
        if rank1 < 10:
            top_ten_hit_count += 1

        print("====== head corrupt result =======")
        print("<%s, %s, %s>" % (entity2text[head].split(",")[0], relation, entity2text[tail].split(",")[0]))
        print("rank: ", rank1+1)
        scores, words, entities = [], [], []
        for j in range(10):
            __idx = argsort1[j]
            h, _, _ = head_corrupt_list[__idx]
            scores.append(str(round(argvalues[j], 4)))
            words.append(entity2text[h].split(",")[0])
            entities.append(h)
        for item in [scores, words, entities]:
            for x in item:
                print(x)

        eval_data = tail_corrupt
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        preds = []
        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, task="lp")
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)
            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0)

        preds = preds[0]
        # get the dimension corresponding to current label 1
        rel_values = preds[:, 1]
        rel_values = torch.tensor(rel_values)
        argvalues, argsort1 = torch.sort(rel_values, descending=True)
        argvalues = argvalues.cpu().numpy()
        argsort1 = argsort1.cpu().numpy()
        rank2 = np.where(argsort1 == 0)[0][0]
        ranks.append(rank2 + 1)
        ranks_right.append(rank2 + 1)
        if rank2 < 10:
            top_ten_hit_count += 1

        logging.info('left: %d, rgiht: %d, mean rank: %.4f, hit@10: %.4f' % (
            rank1, rank2, np.mean(ranks), (top_ten_hit_count * 1.0 / len(ranks))))

        # with open("debugging_result_right.txt", "w") as f:
        #     f.write(str(test_triple) + "\n")
        #     f.write("\t".join([entity2text[head], relation, entity2text[tail]]) + "\n")
        #     f.write("rank: %d\n\n" % rank2)
        #     for idx in range(len(argsort1)):
        #         __idx = argsort1[idx]
        #         h, r, t = tail_corrupt_list[__idx]
        #         f.write('\t'.join([str(idx), str(argvalues[idx]), entity2text[t]]) + '\n')

        print("====== tail corrupt result =======")
        print(test_triple)
        print("<%s, %s, %s>" % (entity2text[head].split(",")[0], relation, entity2text[tail].split(",")[0]))
        print("rank: ", rank2+1)
        scores, words, entities = [], [], []
        for j in range(10):
            __idx = argsort1[j]
            _, _, t = tail_corrupt_list[__idx]
            scores.append(str(round(argvalues[j], 4)))
            words.append(entity2text[t].split(",")[0])
            entities.append(t)
        for item in [scores, words, entities]:
            for x in item:
                print(x)


def dev_eval(args, device, global_step, label_list, num_labels, processor, model, rel_tokenizer,tokenizer,
             convert_examples_to_features, task):
    eval_examples = processor.get_test_examples(args.data_dir)
    if(task!="rp"):
        eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, rel_tokenizer,tokenizer)
    else:
        eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask, task=task)
            logits = outputs#[0]

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss() 
        #print(logits.shape,label_ids.shape)
        #print(label_ids)
        #print(logits,outputs)
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    result = compute_metrics("lp", preds, all_label_ids.numpy())
    result['eval_loss'] = eval_loss
    result['global_step'] = global_step
    return result


if __name__ == "__main__":
    main()
