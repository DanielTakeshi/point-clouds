# Test on model net

python main.py --model pointnet2 --data modelnet10 --seed 50
python main.py --model pointnet2 --data modelnet40 --seed 50

python main.py --model point_transformer --data modelnet10 --seed 50
python main.py --model point_transformer --data modelnet40 --seed 50
