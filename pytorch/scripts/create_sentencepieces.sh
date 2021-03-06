# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DATA_DIR="/mnt/sdd/LibriSpeech/LibriSpeech"
OUTPUT_DIR=${1:-/datasets/sentencepieces}

mkdir -p $OUTPUT_DIR
# jq为命令行操作json file的工具，输出json中的"transcript"，即音频的文本内容
jq -r '.[]["transcript"]' ${DATA_DIR}/librispeech-train-*-wav.json > /tmp/txt.txt
# sentencepiece是一个分词模型
python -c "import sentencepiece as spm; spm.SentencePieceTrainer.train(input='/tmp/txt.txt', model_prefix='librispeech1023', vocab_size=1023, character_coverage=1.0, bos_id=-1, eos_id=-1, model_type='unigram')"
cp librispeech1023.* $OUTPUT_DIR
