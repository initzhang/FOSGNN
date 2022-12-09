# amazon
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=20 nohup python train.py --dataset amazon --decomp 8 --dropout 0.2 --eval-every 10 --lr 0.001 --node-budget 20000 --num-epochs 200 --num-hidden 512 > mos_sage_amazon 2>&1 &
# products
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=20 nohup python train.py --dataset ogbn-products --decomp 8 --dropout 0.2 --eval-every 10 --lr 0.01 --node-budget 30000 --num-epochs 200 --num-hidden 512 > mos_sage_products 2>&1 &
# reddit
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=20 nohup python train.py --dataset reddit --decomp 8 --dropout 0.3 --eval-every 5 --lr 0.01 --node-budget 20000 --num-epochs 50 --num-hidden 128 > mos_sage_reddit 2>&1 &
# yelp
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=20 nohup python train.py --dataset yelp --decomp 8 --dropout 0.2 --eval-every 10 --lr 0.01 --node-budget 10000 --num-epochs 100 --num-hidden 512 > mos_sage_yelp 2>&1 &
