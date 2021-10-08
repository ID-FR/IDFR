export CUDA_VISIBLE_DEVICES=0
python attack.py \
	--dataset "mnist" \
	--attack_method "fgsm" \
	--model_path "/home/liyanni/1307/adversarial/models/mnist/mnist_model.008.h5" \
    --eps 0.25