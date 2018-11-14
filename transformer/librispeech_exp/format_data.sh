#!/bin/bash

if [ $# != 3 ];then
    echo "Format kaldi style data for transformer training."
    echo "Usage: $0 <train_dir> <test_dir> <dest_dir>"
    exit 1
fi

train_dir=$1
test_dir=$2
dest_dir=$3

for f in feats.scp text; do
    for d in $train_dir $test_dir; do
        [ ! -f $d/$f ] && echo "No file: "$d/$f && exit 1
    done
done

cp $train_dir/feats.scp $dest_dir/train.scp
cp $train_dir/text $dest_dir/train.tra
cp $test_dir/feats.scp $dest_dir/test.scp
cp $test_dir/text $dest_dir/test.tra

cut -d" " -f2- $dest_dir/train.tra | tr ' ' '\n' | sort | uniq -c | sort -nr > $dest_dir/unigram
cat $dest_dir/unigram | awk '{print $2}' > $dest_dir/wordlist

echo "Formatting done."




