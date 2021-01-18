# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.platform import gfile
from seq2seq_conversation_model import data_utils, tokenizer, seq2seq_model
from settings import SEQ2SEQ_MODEL_DIR, SEQ2SEQ_SIZE

_LOGGER = logging.getLogger('validation')


class FLAGS(object):
    learning_rate = 0.5  # Learning rate.
    learning_rate_decay_factor = 0.99  # Learning rate decays by this much.
    max_gradient_norm = 5.0  # Clip gradients to this norm.
    batch_size = 64  # Batch size to use during training.
    use_lstm = True  # "use lstm cell or not
    size = SEQ2SEQ_SIZE  # Size of each model layer
    num_samples = 10000  # Size of each model layer.
    num_layers = 3  # Number of layers in the model.
    vocab_size = 100000  # vocabulary size
    data_dir = SEQ2SEQ_MODEL_DIR  # Data directory
    train_dir = SEQ2SEQ_MODEL_DIR + "/train"  # Training directory.
    max_train_data_size = 0  # Limit on the size of training data (0: no limit).
    steps_per_checkpoint = 2  # How many training steps to do per checkpoint.
    decode = False  # Set to True for interactive decoding.
    self_test = False  # Run a self-test if this is set to True.


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.
    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).
    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 1000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(tokenizer.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(
                        _buckets):
                    if len(source_ids) < source_size and len(
                            target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only):
    """Create conversation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.vocab_size, FLAGS.vocab_size, _buckets,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        use_lstm=FLAGS.use_lstm,
        forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    # Prepare conversation data.
    print("Preparing conversation data in %s" % FLAGS.data_dir)
    enquiry_train, answer_train, enquiry_dev, answer_dev, _ = data_utils.prepare_data(
        FLAGS.data_dir, FLAGS.vocab_size)
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    with tf.Session() as sess:
        # Create model.
        print(
            "Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)
        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)
        dev_set = read_data(enquiry_dev, answer_dev)
        train_set = read_data(enquiry_train, answer_train, FLAGS.max_train_data_size)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        print ("Start training ...")
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Log
                log_head = 'current_step: %s' % model.global_step.eval()
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print(
                    "global step %d learning rate %.4f step-time %.2f perplexity "
                    "%.2f" % (
                    model.global_step.eval(), model.learning_rate.eval(),
                    step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(
                        previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir,
                                               "conversation.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs,
                                                 decoder_inputs,
                                                 target_weights, bucket_id,
                                                 True)
                    eval_ppx = math.exp(
                        eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (
                    bucket_id, eval_ppx))
                # Log the answer of validation set: use 20 enquiries from development set
                for bucket_group in dev_set:
                    # decode 4 questions in each bucket
                    for pair in bucket_group[:4]:
                        token_ids = pair[0]
                        log_info = '%s, enquiry: %s' % (log_head, "".join(
                            [rev_vocab[inp] for inp in token_ids]))
                        # Which bucket does it belong to?
                        bucket_id = min([b for b in xrange(len(_buckets))
                                         if _buckets[b][0] > len(token_ids)])
                        # Get a 1-element batch to feed the sentence to the model.
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                            {bucket_id: [(token_ids, [])]}, bucket_id)
                        # Get output logits for the sentence.
                        _, _, output_logits = model.step(sess, encoder_inputs,
                                                         decoder_inputs,
                                                         target_weights,
                                                         bucket_id, True)
                        # This is a greedy decoder - outputs are just argmaxes of output_logits.
                        # Batch_size = 64, we select the first output_logit
                        outputs = [int(np.argmax(logit, axis=1)[0]) for logit in
                                   output_logits]
                        # If there is an EOS symbol in outputs, cut them at that point.
                        # if data_utils.EOS_ID in outputs:
                        #    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                        log_info = '%s, answer: %s' % (log_info, "".join([rev_vocab[output] for output in outputs]))
                        print(log_info)
                        # _LOGGER.info(log_info)
                sys.stdout.flush()


def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.
        # Load vocabularies.
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(sentence, vocab,
                                                         data_utils.ribosome_tokenizer)
            # Which bucket does it belong to?
            bucket_id = min([b for b in xrange(len(_buckets))
                             if _buckets[b][0] > len(token_ids)])
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            print(encoder_inputs[0].shape, decoder_inputs[0].shape,
                  type(decoder_inputs), type(decoder_inputs[0]))
            print(encoder_inputs)
            # Get output logits for the sentence.
            _, average_perplexity, output_logits = model.step(sess,
                                                              encoder_inputs,
                                                              decoder_inputs,
                                                              target_weights,
                                                              bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out response sentence corresponding to outputs.
            print(" ".join([rev_vocab[output] for output in outputs]))
            print("answer perplexity: %s " % average_perplexity)
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()