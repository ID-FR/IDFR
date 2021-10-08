export CUDA_VISIBLE_DEVICES=1
python attack.py \
	--dataset "mnist" \
	--attack_method "bim" \
	--model_path "/home/liyanni/1307/adversarial/models/mnist/mnist_model.008.h5" \
    --eps 0.25