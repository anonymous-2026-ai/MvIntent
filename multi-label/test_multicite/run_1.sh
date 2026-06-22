for LEARNING_RATE in 1e-5 5e-6
do
for n_word in 2 
do
for n_sent in -1
do 
for bs in 2 4 
do
for weight_decay in 0
do
	python run_citation_classification.py \
		--model_name_or_path allenai/scibert_scivocab_uncased \
		--model_type bert \
		--task_name ours \
		--do_train --do_test \
		--data_dir data_multicite_triple\
		--max_seq_length 512 --per_gpu_train_batch_size ${bs} --per_gpu_eval_batch_size 8 \
		--learning_rate ${LEARNING_RATE} --num_train_epochs 10 \
		--output_dir result_baseline --seed 12 \
		--classification_type multilabel --overwrite_cache \
		--overwrite_output_dir --gradient_accumulation_steps 1 \
		 --save_steps 500 --k 0 --logging_steps 500 --evaluate_during_training --n_iter_sent ${n_sent} --n_iter_word ${n_word} --weight_decay ${weight_decay} --save_model 0
done 
done
done 
done 
done 
