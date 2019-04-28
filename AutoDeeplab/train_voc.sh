CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py --backbone xception --lr 0.01  \
--workers 4 --epochs 100 --batch-size 2 --gpu-ids 0 --checkname  \
auto-deeplab-kitti --eval-interval 1 --dataset kitti
