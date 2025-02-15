# CUDA_VISIBLE_DEVICES=1 
python main.py \
	--folder_name AFEW-VA_AL \
	--dataset AFEW-VA \
	--model alexnet \
	--sinkhorn 1 \
	--relevance_weighting 1 \
	--weight_sampling gumbel-softmax \
	--lr 5e-5 \
	--no_domain 1 \
	--topk 5 \
	--latent_dim 64 \
	--erm_input_dim 64 \
	--erm_output_dim 2 \
	--print_check 20 \
	--initial_check 10 \
	--warmup_coef1 10 \
	--warmup_coef2 20 \
	--online_tracker 0 \
	--tr_BS 256
