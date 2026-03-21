RUN_ID=baseline_sp1024 
DATA_PATH=./data/datasets/fineweb10B_sp1024/ 
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model 
VOCAB_SIZE=1024 
VAL_LOSS_EVERY=200 
torchrun --standalone --nproc_per_node=1 train_gpt.py