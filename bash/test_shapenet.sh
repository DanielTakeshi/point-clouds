# Test on shapenet

# TODO need to also show how to test on multiple categories (we are fixing it to airplane).
python main.py --data shapenet --model pointnet2 --seed 50 --batch_size 12
python main.py --data shapenet --model point_transformer --seed 50 --batch_size 12
