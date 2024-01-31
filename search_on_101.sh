conda activate env_name
python ./train/generate_data.py --search_space 101 > generateLog.txt 2>&1 &
wait
python ./train/pretrain.py --search_space 101 --epochs 62 --gpu 0 > pretrainLog.txt 2>&1 &
wait
python ./train/finetune.py --search_space 101 --batch_size 10 --gpu 0 --epochs 200 --train_num 30 --test_num 3000 > finetuneLog.txt 2>&1 &
wait
python ./search/predictor_search.py --search_space 101 --predictor_path ./res/predictor_101_0030.pt --gpu 0 > searchLog.txt 2>&1 &