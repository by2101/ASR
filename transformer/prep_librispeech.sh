#!/bin/bash

set -e

train_dir=/data1/baiye/Speech/kaldi/egs/librispeech/s5/data/train_clean_100
test_dir=/data1/baiye/Speech/kaldi/egs/librispeech/s5/data/test_clean

librispeech_exp/format_data.sh $train_dir $test_dir librispeech_exp
python prepare_examples.py librispeech_exp librispeech_exp 






