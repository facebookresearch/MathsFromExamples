# DDSS - Deep differential system stability -  Learning advanced mathematical computations from examples

This is the source code and data sets relevant to the paper Learning advanced mathematical computations from examples, by Amaury hayat, François Charton and Guillaume Lample, published by ICLR 2021. 

We provide code for 
* data generation
* model training
* model evaluation

We also provide
* 7 datasets
* 7 pretrained models
* a Jupyter notebook demonstrating the use

### Dependencies 

* Python (3.8+)
* Numpy (1.16.4+)
* Sympy (1.4+)
* Pytorch (1.7.1+)
* Control library (0.8.4, from conda-forge)
* CUDA (i.e. a NVIDIA chip) if you intend to use a GPU
* Apex for half-precision training


## Important notes

### Learning with and without GPU
All the code can run on CPU only (set parameter --cpu to true). Data generation is to be done on CPU only. Model training and model evaluation can be done on CPU, but training will be extremely slow. To train or evaluate with a GPU, you need a CUDA-enabled GPU (i.e. a NVIDIA chip).

We support: 
* Half-Precision (with NVIDIA Apex library): set parameters `--fp16 true --amp 2`, to disable, set `--fp16 false --amp -1`
* Multi-GPU training: to run an experiment with several GPU on a unique machine, use 
```bash
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py  # parameters for your experiment
```
* Multi-node training: using GPU on different machines is handled by SLURM (see code)

On GPU with limited video memory, you will need to reduce memory usage by adjusting `--batch_size`. Try to set it to the largest value that will fit in your CUDA memory. Since model optimization is performed at the end of each minibatch, smaller batch sizes will gratly slow learning. You can compensate for this by increasing `--accumulate_gradient`, which controls the number of mini-batches the model sees before optimizing the model.

### Dump paths and experiment names
All paths should be absolute : `--dump_path ./mydump` might not work, `--dump_path c:/Users/me/mydump` should be fine.
The directories where your datasets, models, and logfiles will be generated are constructed from the parameters --dump_path --exp_name and --exp_id, as {dump_path}/{exp_name}/{exp_id}/, if you do not specify an exp_id, a random unique name will be created for you. If you reuse the same dump_path/exp/name/exp_id, generation or training will resume there (adding new examples, or loading the previous model for training).


## Data sets

We provide 7 datasets, all can be found on the FAIR cluster (H2) 
 
### Stability : balanced sample of systems of degree 2 to 5 (50% stable), predicting speed of convergence at 0.01 (largest real part of eigenvalue): 
in directory  `/checkpoint/fcharton/dumped/ddss_gen_stab_bal/`
* ddss_stability_balanced.prefix_counts.train : 25,544,975 systems
* ddss_stability_balanced.prefix_counts.valid.final : 10,000 systems
* ddss_stability_balanced.prefix_counts.test.final : 10,000 systems

### Stability : random sample of systems of degree 2 to 6, predicting speed of convergence at 0.01
in directory  `/checkpoint/fcharton/dumped/ddss_gen_stab/`
* ddss_stability.prefix_counts.train : 92,994,423 systems
* ddss_stability.prefix_counts.valid.final : 10,000 systems
* ddss_stability.prefix_counts.test.final : 10,000 systems

### Controllability: balanced sample of systems of degree 3 to 5 (50% stable), predicting controllability (a binary value)
in directory  `/checkpoint/fcharton/dumped/ddss_gen_ctrl/`
* ddss_control.prefix_counts.train : 26,577,934 systems
* ddss_control.prefix_counts.valid.final : 10,000 systems
* ddss_control.prefix_counts.test.final : 10,000 systems

### Controllability: sample of controllable systems of degree 3 to 6, predicting a control matrix
in directory `/checkpoint/fcharton/dumped/ddss_gen_gram/`
* ddss_gram.prefix_counts.train : 53,680,092 systems
* ddss_gram.prefix_counts.valid.final : 10,000 systems
* ddss_gram.prefix_counts.test.final : 10,000 systems

### Non autonomous controllability: random sample (82.4% controllable) of systems of degree 2 and 3, predicting controllability
in directory  `/checkpoint/fcharton/dumped/ddss_gen_ctrl_t/`
* ddss_control_t.prefix_counts.train : 65,754,655 systems
* ddss_control_t.prefix_counts.valid.final : 10,000 systems
* ddss_control_t.prefix_counts.test.final : 10,000 systems

### Non autonomous controllability: balanced sample (50/50) of systems of degree 2 and 3, predicting controllability
in directory  `/checkpoint/fcharton/dumped/ddss_gen_ctrl_t/`
* ddss_control_t_bal.prefix_counts.train : 23,125,016 systems
* ddss_control_t_bal.prefix_counts.valid.final : 10,000 systems
* ddss_control_t_bal.prefix_counts.test.final : 10,000 systems

### Partial differential equations with initial conditions, predicting existence of a solution and behavior at infinity
in directory `/checkpoint/fcharton/dumped/ddss_gen_fourier/`
* ddss_fourier.prefix_counts.train : 52,285,760 systems
* ddss_fourier.prefix_counts.valid.final : 10,000 systems
* ddss_fourier.prefix_counts.test.final : 10,000 systems

## Training a model from a dataset

```bash
python train.py 

# experiment parameters 
# the full path of this experiment will be /checkpoint/fcharton/dumped/ddss_ctrl/exp_1
--dump_path '/checkpoint/fcharton/dumped'   # path for log files and saved models, avoid ./ and other non absolute paths
--exp_name ddss_ctrl                        # name
--exp_id exp_1                              # id : randomly generated if absent

# dataset
--export_data false
--tasks ode_control         # set to `ode_convergence_speed`, `ode_control` or `fourier_cond_init`
# '{tasks},{train_file_path},{valid_file_path},{test_file_path}'
--reload_data 'ode_control,/checkpoint/fcharton/dumped/ddss_gen_ctrl/ddss_control.prefix_counts.train,/checkpoint/fcharton/dumped/ddss_gen_ctrl/ddss_control.prefix_counts.valid.final,/checkpoint/fcharton/dumped/ddss_gen_ctrl/ddss_control.prefix_counts.test.final' 
--reload_size 40000000      # nr of records to load
--max_len 512               # max length of input or output

# model parameters
--emb_dim 512 
--n_enc_layers 6 
--n_dec_layers 6 
--n_heads 8 
--optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01'

# training parameters
--batch_size 256        # minibatch size, reduce to fit available GPU memory
--epoch_size 300000     # how often evaluation on validation set is performed
--beam_eval 0           # use beam search for evaluation (set to 1 for quantitative tasks)
--eval_size 10000       # size of validation set
--batch_size_eval 256   # batchs for validation, reduce to adjust memory

# validation metrics
# valid_{task}_acc or valid_{task}_beam_acc depending on whether beam search is used  
--validation_metrics valid_ode_control_acc 
# stop after no increase in 20 epochs
--stopping_criterion 'valid_ode_control_acc,20' 
```

## Generating your own data sets

To generate a dataset, use the parameters
```bash 
python train.py --cpu true --export_data true  --reload_data '' --env_base_seed -1  --num_workers 20 --task # task specific parameters 
```
Generated data (exported as sequences of tokens) will be written in file data.prefix in the dump path of the experiment. To be used for training, these files need to be post-processed as shown in the examples below.

IMPORTANT NOTE : Data generation is very slow, and sometimes results in errors that cause the program to abort and need to be relaunched. Typical generating speeds are one or a few systems per second. Whereas one might want to use this code to experiment with data generation, creating datasets on which our models can be trained (10 million examples or more) requires a lot of computing power (typically 200-300 experiments, with 20 CPU each, running for several days)

Important parameters for data generation are : 
* `--tasks` : ode_convergence_speed, ode_control or fourier_cond_init
* `--cpu` : always set to true
* `--num_workers` : set to the number of cores you can use
* `--env_base_seed` : set to -1
* `--min_degree` and `--max_degree` : bounds for the size of the systems generated  
For more details, see file 'envs/ode.py' in the source code

### Predicting stability - balanced sample (50% stable), systems of degree 2 to 5
	
```bash
# Generation command
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 false --amp -1 --emb_dim 128 --n_enc_layers 2 --n_dec_layers 2 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --batch_size 32 --optimizer 'adam,lr=0.0001' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --accumulate_gradients 1 --env_name ode --max_int 10 --precision 2 --skip_zero_gradient true --positive false --nonnull true --prob_int 0.3 --min_degree 2 --max_degree 5 --eval_value 0.01 --prob_positive 0.5 --num_workers 20 --cpu true --stopping_criterion '' --validation_metrics '' --export_data true --reload_data '' --tasks ode_convergence_speed --env_base_seed -1 --exp_name ddss_gen_stab_bal

# Post-processing
# assemble raw data file from prefixes
cat */data.prefix \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> ddss_stability_balanced.prefix_counts

# create train, valid and test samples
python ~/DDSS/split_data.py ddss_stability_balanced.prefix_counts 10000

# check valid and test for duplicates and remove them
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_stability_balanced.prefix_counts.train ddss_stability_balanced.prefix_counts.valid > ddss_stability_balanced.prefix_counts.valid.final
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_stability_balanced.prefix_counts.train ddss_stability_balanced.prefix_counts.test > ddss_stability_balanced.prefix_counts.test.final
```

### Predicting stability - random sample, systems of degree 2 to 6

```bash
# Generation command
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 false --amp -1 --emb_dim 128 --n_enc_layers 2 --n_dec_layers 2 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --batch_size 32 --optimizer 'adam,lr=0.0001' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --accumulate_gradients 1 --env_name ode --max_int 10 --precision 2 --skip_zero_gradient true --positive false --nonnull true --prob_int 0.3 --min_degree 2 --max_degree 6 --eval_value 0.01 --num_workers 20 --cpu true --stopping_criterion '' --validation_metrics '' --export_data true --reload_data '' --tasks ode_convergence_speed --env_base_seed -1 --exp_name ddss_gen_stab

# assemble raw data file from prefixes
cat */data.prefix \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> ddss_stability.prefix_counts
 
# create train, valid and test samples 
python ~/DDSS/split_data.py ddss_stability.prefix_counts 10000

# check valid and test for duplicates and remove them
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_stability.prefix_counts.train ddss_stability.prefix_counts.valid > ddss_stability.prefix_counts.valid.final
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_stability.prefix_counts.train ddss_stability.prefix_counts.test > ddss_stability.prefix_counts.test.final
```

### Predicting controllability - balanced sample, systems of degree 3 to 6

```bash
# generation command 
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 false --amp -1 --emb_dim 128 --n_enc_layers 2 --n_dec_layers 2 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --batch_size 32 --optimizer 'adam,lr=0.0001' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --accumulate_gradients 1 --env_name ode --max_int 10 --precision 3 --skip_zero_gradient true --positive false --nonnull true --prob_int 0.3 --min_degree 3 --max_degree 6 --eval_value 0.9 --allow_complex false --jacobian_precision 3 --qualitative true --num_workers 20 --cpu true --stopping_criterion '' --validation_metrics '' --export_data true --reload_data '' --tasks ode_control --env_base_seed -1 --exp_name ddss_gen_ctrl

# assemble non controllable cases from prefixes
cat */data.prefix \
| grep '0$' \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> ddss_control.prefix_counts.0

# count them
wc -l ddss_control.prefix_counts.0   # 13,298,967

# assemble controllable cases from prefixes
cat */data.prefix \
| grep '1$' \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
| head -n 13298967 > ddss_control.prefix_counts.1

# assemble prefix_counts
cat ddss_control.prefix_counts.0 ddss_control.prefix_counts.1 | shuf > ddss_control.prefix_counts

# create train, valid and test samples
python ~/DDSS/split_data.py ddss_control.prefix_counts 10000

# check valid and test for duplicates and remove them
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_control.prefix_counts.train ddss_control.prefix_counts.valid > ddss_control.prefix_counts.valid.final
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_control.prefix_counts.train ddss_control.prefix_counts.test > ddss_control.prefix_counts.test.final
```

### Predicting non autonomous controllability: unbalanced sample, systems of 2 to 3 equations 

```bash
# generation command 
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 false --amp -1 --emb_dim 128 --n_enc_layers 2 --n_dec_layers 2 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --batch_size 32 --optimizer 'adam,lr=0.0001' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --accumulate_gradients 1 --env_name ode --max_int 10 --precision 3 --skip_zero_gradient true --positive false --nonnull true --prob_int 0.3 --min_degree 2 --max_degree 3 --eval_value 0.5 --allow_complex false --jacobian_precision 3 --qualitative false --tau 1 --num_workers 20 --cpu true --stopping_criterion '' --validation_metrics '' --export_data true --reload_data '' --tasks ode_control --env_base_seed -1 --exp_name ddss_gen_ctrl_t

# assemble raw data file from prefixes
cat */data.prefix \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> ddss_control_t.prefix_counts

# create train, valid and test samples
python ~/DDSS/split_data.py ddss_control_t.prefix_counts 10000

# check valid and test for duplicates and remove them
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_control_t.prefix_counts.train ddss_control_t.prefix_counts.valid > ddss_control_t.prefix_counts.valid.final
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_control_t.prefix_counts.train ddss_control_t.prefix_counts.test > ddss_control_t.prefix_counts.test.final
```

### Predicting non autonomous controllability: balanced sample, systems of 2 to 3 equations 

```bash
# generation command 
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 false --amp -1 --emb_dim 128 --n_enc_layers 2 --n_dec_layers 2 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --batch_size 32 --optimizer 'adam,lr=0.0001' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --accumulate_gradients 1 --env_name ode --max_int 10 --precision 3 --skip_zero_gradient true --positive false --nonnull true --prob_int 0.3 --min_degree 2 --max_degree 3 --eval_value 0.5 --allow_complex false --jacobian_precision 3 --qualitative false --tau 1 --num_workers 20 --cpu true --stopping_criterion '' --validation_metrics '' --export_data true --reload_data '' --tasks ode_control --env_base_seed -1 --exp_name ddss_gen_ctrl_t

# assemble non controllable cases from prefixes
cat */data.prefix \
| grep '0$' \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> ddss_control_t.prefix_counts.0

# count them
wc -l ddss_control_t.prefix_counts.0   # 11,572,508

# assemble controllable cases from prefixes
cat */data.prefix \
| grep '1$' \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
| head -n 11572508 > ddss_control_t.prefix_counts.1

# assemble prefix_counts
cat ddss_control_t.prefix_counts.0 ddss_control_t.prefix_counts.1 | shuf > ddss_control_t_bal.prefix_counts

# create train, valid and test samples
python ~/DDSS/split_data.py ddss_control_t_bal.prefix_counts 10000

# check valid and test for duplicates and remove them
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_control_t_bal.prefix_counts.train ddss_control_t_bal.prefix_counts.valid > ddss_control_t_bal.prefix_counts.valid.final
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_control_t_bal.prefix_counts.train ddss_control_t_bal.prefix_counts.test > ddss_control_t_bal.prefix_counts.test.final
```

### Predicting control matrices - sample of controllable systems, of degree 3 to 6

```bash
# generation command
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 false --amp -1 --emb_dim 128 --n_enc_layers 2 --n_dec_layers 2 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --batch_size 32 --optimizer 'adam,lr=0.0001' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --accumulate_gradients 1 --env_name ode --max_int 10 --precision 3 --skip_zero_gradient true --positive false --nonnull true --prob_int 0.3 --min_degree 3 --max_degree 6 --eval_value 0.5 --allow_complex false --jacobian_precision 2 --qualitative false --predict_gramian true --prob_positive 1.0 --num_workers 20 --cpu true --stopping_criterion '' --validation_metrics '' --export_data true --reload_data '' --tasks ode_control --env_base_seed -1 --exp_name ddss_gen_gram

# assemble raw data file from prefixes
cat */data.prefix \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> ddss_gram.prefix_counts
 
# create train, valid and test samples 
python ~/DDSS/split_data.py ddss_gram.prefix_counts 10000

# check valid and test for duplicates and remove them
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_gram.prefix_counts.train ddss_gram.prefix_counts.valid > ddss_gram.prefix_counts.valid.final
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_gram.prefix_counts.train ddss_gram.prefix_counts.test > ddss_gram.prefix_counts.test.final
```

### Predicting the existence of solutions of partial differential equations

```bash
# generation command
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 false --amp -1 --emb_dim 128 --n_enc_layers 2 --n_dec_layers 2 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --batch_size 32 --optimizer 'adam,lr=0.0001' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --accumulate_gradients 1 --env_name ode --max_int 10 --precision 2 --jacobian_precision 2 --positive false --nonnull true --allow_complex false --predict_bounds true --skip_zero_gradient true --prob_int 0.3 --min_degree 2 --max_degree 6 --eval_value 0.01 --prob_positive -1.0 --num_workers 20 --cpu true --stopping_criterion '' --validation_metrics '' --export_data true --reload_data '' --tasks fourier_cond_init --env_base_seed -1 --exp_name ddss_gen_fourier

# assemble raw data file from prefixes
cat */data.prefix \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> ddss_fourier.prefix_counts
 
# create train, valid and test samples 
python ~/DDSS/split_data.py ddss_fourier.prefix_counts 10000

# check valid and test for duplicates and remove them
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_fourier.prefix_counts.train ddss_fourier.prefix_counts.valid > ddss_fourier.prefix_counts.valid.final
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' ddss_fourier.prefix_counts.train ddss_fourier.prefix_counts.test > ddss_fourier.prefix_counts.test.final
```

## Pre-trained models
We provide 7 pretrained models for the various problems. Below are the links, the dataset they were trained on, and the parameters used, and the performance on the validation set (valid.final in the same directory, 10 000 held-out examples).

### Predicting stability (qualitative)
* Model: `/checkpoint/fcharton/dumped/ddss_stab_quali/37185697/best-valid_ode_control.pth`
* Training set: `/checkpoint/fcharton/dumped/ddss_gen_stab_bal/ddss_stability_balanced.prefix_counts.train`
* Accuracy over validation set: 97.1%
* Training parameters (command line)
```bash
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 true --amp 2 --accumulate_gradients 1 --emb_dim 512 --batch_size 128 --batch_size_eval 256 --n_enc_layers 6 --n_dec_layers 6 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --optimizer 'adam,lr=0.0001' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --num_workers 1 --export_data false --env_name ode --max_int 10 --positive false --nonnull true --qualitative true --skip_zero_gradient true --prob_int 0.3 --max_degree 5 --min_degree 2 --eval_verbose 0 --beam_eval 0 --eval_size 10000 --tasks ode_convergence_speed --reload_data 'ode_convergence_speed,/checkpoint/fcharton/dumped/ddss_gen_stab_bal/ddss_stability_balanced.prefix_counts.train,/checkpoint/fcharton/dumped/ddss_gen_stab_bal/ddss_stability_balanced.prefix_counts.valid.final,/checkpoint/fcharton/dumped/ddss_gen_stab_bal/ddss_stability_balanced.prefix_counts.test.final' --reload_size 40000000 --stopping_criterion 'valid_ode_convergence_speed_acc,40' --validation_metrics valid_ode_convergence_speed_acc --env_base_seed -1 --exp_name ddss_stab_quali
```

### Stability:  computing convergence speed

### Predicting autonomous controllability
* Model: `/checkpoint/fcharton/dumped/ddss_ctrl/37056800/best-valid_ode_control.pth`
* Training set: `/checkpoint/fcharton/dumped/ddss_gen_ctrl/ddss_control.prefix_counts.train`
* Accuracy over validation set: 97.4%
* Training parameters (command line)
```bash
 python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 true --amp 2 --accumulate_gradients 1 --emb_dim 512 --batch_size 256 --batch_size_eval 256 --n_enc_layers 6 --n_dec_layers 6 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --num_workers 1 --export_data false --env_name ode --max_int 10 --positive false --nonnull true --skip_zero_gradient true --prob_int 0.3 --max_degree 6 --min_degree 3 --eval_value 0.9 --qualitative true --eval_verbose 0 --beam_eval 0 --eval_size 10000 --tasks ode_control --reload_data 'ode_control,/checkpoint/fcharton/dumped/ddss_gen_ctrl/ddss_control.prefix_counts.train,/checkpoint/fcharton/dumped/ddss_gen_ctrl/ddss_control.prefix_counts.valid.final,/checkpoint/fcharton/dumped/ddss_gen_ctrl/ddss_control.prefix_counts.test.final' --reload_size 40000000 --stopping_criterion 'valid_ode_control_acc,20' --validation_metrics valid_ode_control_acc --env_base_seed -1 --exp_name ddss_ctrl
 ```

### Predicting non-autonomous controllability
* Model: `/checkpoint/fcharton/dumped/ddss_ctrl_t/37185745/best-valid_ode_control.pth`
* Training set: `/checkpoint/fcharton/dumped/ddss_gen_ctrl_t/ddss_control_t.prefix_counts.train`
* Accuracy over validation set: 99.6%
* Training parameters (command line)
```bash
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 true --amp 2 --accumulate_gradients 1 --emb_dim 512 --batch_size 256 --batch_size_eval 256 --n_enc_layers 6 --n_dec_layers 6 --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 512 --optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --num_workers 1 --export_data false --env_name ode --max_int 10 --positive false --nonnull true --skip_zero_gradient true --prob_int 0.3 --max_degree 3 --min_degree 2 --eval_value 0.5 --qualitative false --tau 1 --eval_verbose 0 --beam_eval 0 --eval_size 10000 --tasks ode_control --reload_data 'ode_control,/checkpoint/fcharton/dumped/ddss_gen_ctrl_t/ddss_control_t.prefix_counts.train,/checkpoint/fcharton/dumped/ddss_gen_ctrl_t/ddss_control_t.prefix_counts.valid.final,/checkpoint/fcharton/dumped/ddss_gen_ctrl_t/ddss_control_t.prefix_counts.test.final' --reload_size 40000000 --stopping_criterion 'valid_ode_control_acc,60' --validation_metrics valid_ode_control_acc --env_base_seed -1 --exp_name ddss_ctrl_t
```

### Computing control matrices: predicting solution up to 10% 

### Computing control matrices: predicting a correct mathematical solution

### Predicting the existence of solutions of partial differential equations
* Model: `/checkpoint/fcharton/dumped/ddss_fourier/37062096/best-valid_fourier_cond_init_acc.pth`
* Training set: `/checkpoint/fcharton/dumped/ddss_gen_fourier/ddss_fourier.prefix_counts.train`
* Accuracy over validation set: 98.6%
* Training parameters (command line) 
```bash
python train.py --dump_path '/checkpoint/fcharton/dumped' --save_periodic 0 --fp16 false --amp -1 --accumulate_gradients 1 --emb_dim 512 --n_enc_layers 8 --n_dec_layers 8 --batch_size 64 --batch_size_eval 64 --eval_size 10000 --predict_jacobian false --n_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --max_len 1024 --optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01' --clip_grad_norm 5 --epoch_size 300000 --max_epoch 100000 --num_workers 1 --export_data false --env_name ode --max_int 10 --precision 3 --jacobian_precision 1 --positive false --nonnull true --prob_int '0.3' --max_degree 6 --eval_value 0.5 --allow_complex false --predict_bounds true --skip_zero_gradient true --eval_verbose 0 --beam_eval 0 --tasks fourier_cond_init --reload_data 'fourier_cond_init,/checkpoint/fcharton/dumped/ddss_gen_fourier/ddss_fourier.prefix_counts.train,/checkpoint/fcharton/dumped/ddss_gen_fourier/ddss_fourier.prefix_counts.valid,/checkpoint/fcharton/dumped/ddss_gen_fourier/ddss_fourier.prefix_counts.test' --reload_size 40000000 --stopping_criterion 'valid_fourier_cond_init_acc,20' --validation_metrics valid_fourier_cond_init_acc --env_base_seed -1 --exp_name ddss_fourier
```

## Evaluating trained models


## Citation
This code is released under a ... License. Please cite the following paper in research using the source code, datasets or models.


