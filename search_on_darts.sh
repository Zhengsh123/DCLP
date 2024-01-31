conda activate env_name
python ./train/generate_data.py --search_space darts --data_num 1000 > generateLog.txt 2>&1 &
wait
python ./train/pretrain.py --search_space darts --epochs 2 --gpu 0 --train_path ./pkl/darts_data.pkl > pretrainLog.txt 2>&1 &
wait
python ./train/finetune.py --search_space darts --batch_size 10 --gpu 0 --epochs 200 \
--train_num 30 --test_num 3000 --train_path ./pkl/darts_train_data.pkl \
--pretrain_path ./checkpoint/checkpoint_darts_0001.pth.tar > finetuneLog.txt 2>&1 &
wait
python ./search/predictor_search.py --search_space darts --predictor_path ./res/predictor_darts_0030.pt \
  --gpu 0 --save_path ./darts.txt > searchLog.txt 2>&1 &