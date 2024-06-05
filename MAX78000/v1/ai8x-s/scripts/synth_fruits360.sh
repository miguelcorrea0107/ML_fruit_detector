python ai8xize.py --test-dir synthed_net --prefix fruits360 --checkpoint-file trained/fruits360_trained-q.pth.tar --config-file networks/fruits360.yaml \
 --sample-input ../ai8x-training/sample_fruits360.npy --device MAX78000 --board-name FTHR_RevA --compact-data \
 --mexpress --timer 0 --display-checkpoint --verbose --fifo --overwrite "$@"