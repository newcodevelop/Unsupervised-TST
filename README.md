# Unsupervised-TST

This describes codes and datasets used for TST project.

#STEPS

1. Train style classifier using train_style_classifier.py by CUDA_VISIBLE_DEVICES=<GPU_ID> python train_style_classifier.py. It'll output a folder named './results_yelp' where all the model checkpoints will be stored.
2. Train the fluency reward generator (a GPT2 model) using train_gpt2.py CUDA_VISIBLE_DEVICES=<GPU_ID> python train_style_classifier.py. It'll output a folder named './gpt2_lm' where model checkpoints will be stored.
3. Train TST model using train_yelp_nll.py using command deepspeed --include=localhost:<GPU_ID> --master_port 60000 train_yelp_nll.py
4. Evaluate TST model using eval_yelp.py using command CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_yelp.py. It will generate a csv file containing model results.
5. To compare with all the SoTA models, use eval_all_models.py script. This will print all the scores necessary to compare TST model developed above to that of existing models.

Necessary folders are attached to this repository as well.
