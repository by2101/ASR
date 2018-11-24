#!/bin/env python

import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
import third_party.kaldi_io as kio

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("prepare_examples")
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description="""
        Prepare tfrecords for training.
        """)

    parser.add_argument("traindir",
                        help="Training data directory, which includes four files: train_feats.scp, train.tra train_utt2dur, wordlist. Use format_data.sh to generate.")
    parser.add_argument("destdir",
                        help="To store generated tfrecord files.")                     
    parser.add_argument('--max-num-one-file', type=int, dest = "max_num_one_file", default = 10000, help='The largest number of examples in one tfrecord file.')                        
    args = parser.parse_args()
    return args


def load_wordlist(fn):
    with open(fn, 'r') as f:
        words = f.read().strip().split('\n')
    word_to_id = {v: k for k,v in enumerate(words)}
    return word_to_id

    
def parse_tra(fn, word_to_id):
    with open(fn, 'r') as f:
        lines = f.read().strip().split('\n')    
    items = [l.split() for l in lines]
    tra_dic = {}
    
    for item in items:
        words = []
        for c in item[1:]:
            if not c in word_to_id:
                logger.warn("utt {}: {} is not in wordlist. Replace it with <UNK>.".format(item[0], c))
                c = "<UNK>"
            words.append(word_to_id[c])    
        tra_dic[item[0]] = words
    return tra_dic

    
def parse_scp(fn):
    with open(fn, 'r') as f:
        lines = f.read().strip().split('\n')
    items = [l.split() for l in lines]    
    scp_dic = {k: v for [k, v] in items}    
    return scp_dic 

    
def parse_dur(fn):
    with open(fn, 'r') as f:
        lines = f.read().strip().split('\n')
    items = [line.split() for line in lines]    
    durs = [(item[0], float(item[1])) for item in items]
    durs = sorted(durs, key=lambda x: x[1])
    return durs

    
def durs_to_frames(durs, frame_shift=10):
    return [ (item[0], int(item[1]*1000/frame_shift)) for item in durs ]    

    
def stat_frames(frames, interval=10):
    shortest_utt = min(frames, key=lambda k:k[1])
    longest_utt = max(frames, key=lambda k:k[1])
    buckets = [(i, i+interval) for i in range(shortest_utt[1], longest_utt[1]+interval, interval)]
    hist = [ 0 for i in buckets] # each bucket contains utt whose length is with [buckets[i], buckes[i+1])  
    for item in frames:
        l = item[1]
        bucket = ((l - shortest_utt[1])//interval)
        hist[bucket] += 1          
    return buckets, hist

    
def filter_buckets(buckets, hist, min_count=5):
    new_buckets = []
    new_hist = []
    for i in range(len(hist)):
        if hist[i] > min_count:
            new_buckets.append(buckets[i])
            new_hist.append(hist[i])
    return new_buckets, new_hist

    
def select_bucket(shortest_len, interval, length):
    bucket_idx = (length - shortest_len) // interval
    return bucket_idx
    
    
def utt_to_bucket_idx(frames, buckets):
    shortest_len = buckets[0][0] # assuming that the buckets are sorted
    interval = buckets[0][1] - buckets[0][0]
    utt_dict = {}
    for utt in frames:
        idx = select_bucket(shortest_len, interval, utt[1])
        if idx < len(buckets) and utt[1] < buckets[idx][1] and utt[1] >= buckets[idx][0]:
            if utt[0] in utt_dict:
                raise ValueError("utterances are duplicated.")
            utt_dict[utt[0]] = idx 
        else: # illegal bucket
            logger.warn("The length of {} is {}, which is out of bucket.".format(utt[0], utt[1]))
    return utt_dict
   
   
def check_wordlist(fn):
    # check wordlist, the first four symbol should be <PAD> <UNK> <S> </S>
    with open(fn, 'r') as f:
        special_syms = f.read().strip().split("\n")[:4]
    status = (["<PAD>", "<UNK>", "<S>", "</S>"] == special_syms)
    if status:
        return True
    else:
        logger.warn("The first four symbols are {}".format(",".join(special_syms)))
        return False
    
    
def process_trainingset_to_tfrecord(srcdir, destdir, max_count_one_file=10000):
    for fn in ["train_feats.scp", "train.tra", "train_utt2dur", "wordlist"]:
        if not os.path.exists(os.path.join(srcdir, fn)):
            raise ValueError("No such file: {}".format(fn))

    if not check_wordlist(os.path.join(srcdir, "wordlist")):
        raise ValueError("The wordlist is not formatted right.")
     
    logger.info("Loading data...")
    word_to_id = load_wordlist(os.path.join(srcdir, "wordlist"))
    durs = parse_dur(os.path.join(srcdir, "train_utt2dur"))
    feats_scp_dic = parse_scp(os.path.join(srcdir, "train_feats.scp"))
    transcripts_dic = parse_tra(os.path.join(srcdir, "train.tra"), word_to_id)
    
    # achive some statistics
    frames = durs_to_frames(durs)
    buckets, hist = stat_frames(frames)
    new_buckets, new_hist = filter_buckets(buckets, hist, min_count=5)
    utt_dict =  utt_to_bucket_idx(frames, new_buckets)        
    valid_utt = utt_dict.keys()
    
    logger.info("Shuffling...")
    np.random.shuffle(valid_utt)
    
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    utt_lists = []
    file_num = len(valid_utt) // max_count_one_file
    if len(valid_utt) % max_count_one_file != 0:
        file_num += 1
    
    for i in range(file_num):
        utt_lists.append(valid_utt[i*max_count_one_file:(i+1)*max_count_one_file])
    
    for i in range(file_num):
        fn_towrite = "train.{}.tfrecord".format(i)
        logger.info("Writing file: {}".format(fn_towrite))
        writer = tf.python_io.TFRecordWriter(os.path.join(destdir, fn_towrite))
        for utt in utt_lists[i]:
            feats = kio.read_mat(feats_scp_dic[utt])
            labels = transcripts_dic[utt]
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'utt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[utt])), 
                    'feats': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feats.tobytes()])),
                    'feat_dim': tf.train.Feature(int64_list =tf.train.Int64List(value=[feats.shape[1]])),
                    'feat_len': tf.train.Feature(int64_list =tf.train.Int64List(value=[feats.shape[0]])),
                    'labels': tf.train.Feature(int64_list =tf.train.Int64List(value=labels)),
                    'labels_len': tf.train.Feature(int64_list =tf.train.Int64List(value=[len(labels)])),
                    }
                    ))     
            writer.write(example.SerializeToString())
        writer.close()
    logger.info("Processing training set has been done.")
    
    
if __name__ == '__main__':
    args = get_args()
    traindata_dir = args.traindir
    destdir = args.destdir
    max_num_one_file = args.max_num_one_file
    
    process_trainingset_to_tfrecord(traindata_dir, destdir, max_count_one_file=max_num_one_file)
    

