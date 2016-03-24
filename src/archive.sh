#!/usr/bin/env bash
batch_name=$1
scratch_dir=$2
data_dir=$3

tar -zcv -f $3/archive/$1.tgz -C $scratch_dir/scraped-text $1 -C $scratch_dir/worldviews/$1 stanford-parsed --transform="s,^$1/\(.*.json\),$1/AIDA/\1,;s,^stanford-parsed,$1/CoreNLP,;s,^$1/\([^/]*.txt\),$1/raw-text/\1,"
