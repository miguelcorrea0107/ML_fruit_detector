python train.py --model fruits360 --dataset fruits360 --evaluate \
 --save-sample 10 \
 --exp-load-weights-from ../ai8x-synthesis/trained/fruits360_trained-q.pth.tar \
 -8 \
 --device MAX78000 "$@"