python train.py --lr 0.001 --optimizer adam --epochs 40 --batch-size 64 --deterministic \
 --compress policies/schedule.yaml --qat-policy policies/qat_policy_fruits360.yaml \
 --model fruits360 --dataset fruits360 --confusion \
 --param-hist --pr-curves --embedding \
 --device MAX78000 --print-freq 10 --out-dir checkpoints --seed 42