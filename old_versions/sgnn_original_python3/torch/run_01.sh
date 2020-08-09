#python train.py --gpu 0 --data_path ./data/completion_blocks --train_file_list ../filelists/train_list.txt --val_file_list ../filelists/val_list.txt --save_epoch 1 --save logs/mp --max_epoch 4

python train.py --gpu 1 --data_path ./data/completion_blocks --train_file_list ../filelists/train_list.txt --val_file_list ../filelists/val_list.txt --save_epoch 1 --save logs/mp --max_epoch 4

