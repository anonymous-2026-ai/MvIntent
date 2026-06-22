for LEARNING_RATE in 1e-6
do
for n_word in 1
do
for n_sent in -1
do
for bs in 4
do
for SEED in 0
do 
for ws in 0
do
for wd in 0
do
for alpha in 0.2
do 
	python run_citation_classification.py \
		--model_name_or_path allenai/scibert_scivocab_uncased \
		--model_type bert \
		--task_name ours \
		--do_train --do_eval \
		--data_dir ../../datasets/data_multicite_triple\
		--max_seq_length 512 --per_gpu_train_batch_size ${bs} --per_gpu_eval_batch_size 32 \
		--learning_rate ${LEARNING_RATE} --num_train_epochs 1 \
		--output_dir result_baseline --seed ${SEED} \
		--classification_type multilabel --overwrite_cache \
		--overwrite_output_dir --gradient_accumulation_steps 1 --alpha ${alpha}\
		 --save_steps 200 --k 0 --logging_steps 200 --evaluate_during_training --n_iter_sent ${n_sent} --n_iter_word ${n_word} --warmup_steps ${ws} --weight_decay ${wd} --save_model 1
done
 done
 done
done
done
done 
done 
done
