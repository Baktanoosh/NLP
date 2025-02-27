# models.py
import time
import argparse
import random
import numpy as np
from data import *
from models import *
from utils import *
from typing import List

from transformers import pipeline, AutoTokenizer, BertConfig, BertModel, BartConfig, BartForConditionalGeneration, BartModel, TrainingArguments, Trainer


def initialize_seq2seq_model(vocab_size: int):
    """
    :param vocab_size: The size of the vocabulary to use
    :return: A randomly initialized BartForConditionalGeneration model, with relatively small parameter sizes compared
    to what they are by default. You can modify these to experiment if you'd like, but it's outside the scope of
    what's intended in the assignment.
    """
    # The hyperparameters of the BART model are defined here
    # They are smaller than any of the standard BART models,
    # so that we can train it form scratch on our semantic parsing dataset
    config = BartConfig(
        vocab_size=vocab_size,
        max_position_embeddings=100, # was 1024
        encoder_layers=4, # was 12
        encoder_ffn_dim=256, # was 4096
        encoder_attention_heads=4, # was 16
        decoder_layers=4, # was 12
        decoder_ffn_dim=256, # was 4096
        decoder_attention_heads=8, # was 16
        d_model=128) # was 1024

    # The BART model with random weights gets created here
    model = BartForConditionalGeneration(config)
    return config, model


def train_seq2seq_model(model, train_dataset, val_dataset, args):
    """
    :param model: a randomly initialized seq2seq model sharing the BART architecture
    :param train_dataset: the preprocessed train dataset
    :param val_dataset: the preprocessed validation (dev) dataset
    :param args: args bundle from main
    :return: nothing; trains the seq2seq model and updates its parameters in-place
    """
    # We define a set of arguments to be passed to the trainer
    # Most of the arguments are training related hyperparameters
    # some are about storing checkpoint and tensorboard logs
    training_args = TrainingArguments(        
        output_dir=args.model_save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        learning_rate=5e-04, # default seems to be 5e-05
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    # The trainer is a class that handles all aspects of the
    # training for us. We pass to it the model, dataset and other
    # hyperparameters. We don't have to worry about writing the
    # training loop or dealing with checkpointing, logging etc..
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    # We call the train function on the trainer
    # This trains the model we passed through on the dataset
    # we passed through
    trainer.train()


def score_sequence(pred_to_first_eos, gold_labels) -> (int, int, int):
    """
    Evaluates the given sequence and returns the sufficient statistics for accuracy computation.
    DO NOT MODIFY THIS -- we are giving it to you here in models.py for convenience, but we will compute your accuracy
    with an unmodified version of it.
    :param pred_to_first_eos: predicted tokens (real tokens, not indexed) up to and including an <EOS> token
    :param gold_labels: the gold labels (which also include EOS)
    :return: a tuple of exact_match (1 or 0), the token-level recall (the fraction of tokens in the gold that are matched
    by a corresponding token in the prediction), and the total number of tokens
    """
    exact_match = 0
    recall = 0
    total_toks = 0
    total_toks += len(gold_labels)
    if pred_to_first_eos == gold_labels:
        exact_match += 1
    #print("Pred: " + ' '.join(pred_to_first_eos))
    #print("Gold: " + ' '.join(gold_labels))
    for j in range(0, len(gold_labels)):
        if j < len(pred_to_first_eos) and pred_to_first_eos[j] == gold_labels[j]:
            recall += 1
    return (exact_match, recall, total_toks)


def score_decoded_outputs(all_preds, indexer, exs):
    """
    Prints two metrics:
    1. Token-level recall: what fraction of the gold tokens are exactly predicted *at the same position* by
    the model.
    2. Exact match: how often the entire sequence exactly matches the gold standard.
    :param all_preds: list of the model's predictions, must be shorter than or equal to exs in size
    :param indexer:
    :param exs: the examples
    :return:
    """
    (top1_exact_match, top1_recall, top1_total_toks) = (0, 0, 0)
    for i in range(0, len(all_preds)):
        gold_labels = [indexer.get_object(j) for j in exs[i]['labels'] if j >= 0]
        one_best = all_preds[i]
        (top1_em, top1_rec, top1_tt) = score_sequence(one_best, gold_labels)
        top1_exact_match += top1_em
        top1_recall += top1_rec
        top1_total_toks += top1_tt
    print("Recall: " + repr(top1_recall) + "/" + repr(top1_total_toks) + " = " + repr(top1_recall / top1_total_toks))
    print("Exact Match: " + repr(top1_exact_match) + "/" + repr(len(all_preds)) + " = " + repr(top1_exact_match / len(all_preds)))


def pred_indices_to_prediction(raw_pred_indices, indexer):
    """
    :param raw_pred_indices: the raw output of the model
    :param indexer
    :return: The decoded sequence *stopping at the first occurrence of EOS*
    """
    pred_labels = [indexer.get_object(id.item()) for id in raw_pred_indices]
    # Find the first EOS token or take the whole sequence if there's no EOS
    first_eos_tok = pred_labels.index(EOS_SYMBOL) if EOS_SYMBOL in pred_labels else len(pred_labels) - 1
    return pred_labels[0:first_eos_tok + 1] # include the EOS in the return


def decode_basic(model, indexer, exs, num_exs=-1):
    """
    Basic decoding method to show how to use .generate() from a HuggingFace model to get outut
    :param model:
    :param indexer:
    :param exs:
    :param num_exs: -1 if we should use all the examples, otherwise a small number to allow decoding on fewer examples
    :return:
    """
    all_example_preds = []
    num_exs_to_use = min(num_exs, len(exs)) if num_exs > 0 else len(exs)
    for i in range(0, num_exs_to_use):
        ex_length = sum(exs[i]['attention_mask'])
        dev_input_tensor = torch.tensor([exs[i]['input_ids'][0:ex_length]], dtype=torch.long)
        # You can increase this to run "real" beam search
        beam_size = 1
        # The generate method runs decoding with the specified set of
        # hyperparameters and returns a list of possible sequences
        output_ids = model.generate(dev_input_tensor, num_beams=beam_size, max_length=65, early_stopping=True, num_return_sequences=beam_size)
        # [0] extracts the first candidate in the beam for the simple decoding method
        one_best = pred_indices_to_prediction(output_ids.data[0][1:], indexer)
        all_example_preds.append(one_best)
    return all_example_preds


def decode_oracle(model, indexer, exs, num_exs):
    start = time.time()
    all_example_preds = []
    num_exs_to_use = min(num_exs, len(exs)) if num_exs > 0 else len(exs)
    for i in range(0, num_exs_to_use):
        max_ind = 0
        score_list = []
        ex_length = sum(exs[i]['attention_mask'])
        dev_input_tensor = torch.tensor([exs[i]['input_ids'][0:ex_length]], dtype=torch.long)
        beam_size = 7
        for bm in range(0, beam_size):
            output_ids = model.generate(dev_input_tensor, num_beams=beam_size , max_length=65, early_stopping=True, num_return_sequences=beam_size)
            gold_labels = [indexer.get_object(jj) for jj in exs[i]['labels'] if jj >= 0]
            pred = pred_indices_to_prediction(output_ids.data[bm][1:], indexer)
            (top1_em, top1_rec, top1_tt) = score_sequence(pred, gold_labels)
            score_list.append(top1_em)
        max_ind = score_list.index(max(score_list))
        one_best = pred_indices_to_prediction(output_ids.data[max_ind][1:], indexer)
        all_example_preds.append(one_best)
    now = time.time()
    print("Total Traing time", now-start)
    return all_example_preds


def decode_fancy(model, indexer, exs, num_exs):
    start = time.time()
    all_example_preds = []
    num_exs_to_use = min(num_exs, len(exs)) if num_exs > 0 else len(exs)
    common_list = set()
    max_ind = 0
    for i in range(0, num_exs_to_use):
        beam_size = 5
        score_list = []
        best_ones = []
        label = []
        item_list = []
        gold_labels = [indexer.get_object(j) for j in exs[i]['input_ids'] if j >= 0]
        for string_common in gold_labels:
            if string_common in const_list:
                common_list.add(string_common.lower())
        ex_length = sum(exs[i]['attention_mask'])
        dev_input_tensor = torch.tensor([exs[i]['input_ids'][0:ex_length]], dtype=torch.long)
        output_ids = model.generate(dev_input_tensor, num_beams=beam_size , max_length=65, early_stopping=True, num_return_sequences=beam_size)
        best_score = 0
        best_option = []
        for bm in range(1, beam_size):
            pred = pred_indices_to_prediction(output_ids.data[bm][1:], indexer)
            count = 0
            for item in range(len(pred)):
                if pred[item] in common_list:
                    count += 1
            if count > best_score:
                best_score = count
                best_option = pred

        all_example_preds.append(best_option)    
    now = time.time()
    print("Total Traing time", now-start)
    return all_example_preds


# 'east' doesn't actually exist
const_list = ['new', 'north', 'south', 'west', 'east', 'alabama', 'alaska', 'arizona', 'arkansas', 'california',
    'colorado', 'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa','kansas', 
    'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana',
    'nebraska', 'nevada', 'hampshire', 'jersey', 'mexico', 'york', 'carolina', 'dakota', 'ohio', 'oklahoma', 'oregon',
    'pennsylvania', 'rhode', 'island', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'virginia', 'wisconsin',
    'wyoming','smallest' ,'largest', 'longest', 'lowest', 'highest', 'biggest', 'smallest', 'york', 'seattle', 'des', 'moines', 
    'dover', 'mount', 'mckinley', 'rochester', 'orleans', 'rio', 'grande', 'dallas', 'san', 'francisco', 'tucson','average',
    'red', 'potomac', 'springfield', 'sd', 'usa', 'durham', 'boulder', 'houston', 'portland', 'austin', 'united', 'states',
    'plano', 'point', 'points', 'chattahoochee', 'detroit', 'american', 'lake', 'city', 'sacramento','Diego', '	Columbus', 
    'Antonio']


