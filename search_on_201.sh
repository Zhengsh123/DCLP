conda activate env_name
python ./train/generate_data.py --search_space 201 > generateLog.txt 2>&1 &
wait
python ./train/pretrain.py --search_space 201 --epochs 2 --gpu 0 --train_path ./pkl/nasbench201_all_data.pkl > pretrainLog.txt 2>&1 &
wait
python ./train/finetune.py --search_space 201 --batch_size 10 --gpu 0 --epochs 200 \
--train_num 30 --test_num 3000 --train_path ./pkl/nasbench201_all_data.pkl \
--pretrain_path ./checkpoint/checkpoint_201_0001.pth.tar > finetuneLog.txt 2>&1 &
wait
python ./search/predictor_search.py --search_space 201 --predictor_path ./res/predictor_201_0030.pt \
 --train_path ./pkl/nasbench201_all_data.pkl --gpu 0 --save_path ./nasbench_201.txt > searchLog.txt 2>&1 &