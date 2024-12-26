# python3 train.py --dataset CUB_200_2011 --split non-overlap --num_step 10000 --alpha 0 --name alpha-0;
# python3 train.py --dataset CUB_200_2011 --split non-overlap --num_step 10000 --alpha 0.2 --name alpha-0.2;
# # python3 train.py --dataset CUB_200_2011 --split non-overlap --num_step 10000 --alpha 0.4 --name alpha-0.4;
# python3 train.py --dataset CUB_200_2011 --split non-overlap --num_step 10000 --alpha 0.6 --name alpha-0.6;

# python3 train.py --dataset CUB_200_2011 --split non-overlap --num_step 10000 --contrastive_loss 0 --name contrastive_loss-0;
# # python3 train.py --dataset CUB_200_2011 --split non-overlap --num_step 10000 --contrastive_loss 1 --name contrastive_loss-1;

python3 train.py --dataset CUB_200_2011 --split non-overlap --num_step 10000 --vit 1 --name vit-non-overlap &&
python3 train.py --dataset CUB_200_2011 --split overlap --num_step 10000 --vit 1 --name vit-overlap &&
python3 train.py --dataset CUB_200_2011 --split non-overlap --num_step 10000 --vit 1 --contrastive_loss 0 --name vit-contrastive_loss-0;