import argparse
import json
import os
import torch
from functools import partial

import wandb
from transformers import AutoTokenizer
from transformers import set_seed

from improved_diffusion import dist_util, logger  # improved diffusion是模块文件，open AI写的
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.rounding import load_models, load_tokenizer
from improved_diffusion.script_util import (add_dict_to_argparser, args_to_dict, create_model_and_diffusion,
											model_and_diffusion_defaults)
from improved_diffusion.test_util import compute_logp, get_weights
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.train_util import TrainLoop

"""
Train a diffusion model on images.
"""
#  以下注释和该训练指令配套：
"""python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 
--lr_anneal_steps 200000  --seed 102 --noise_schedule sqrt --in_channel 16 --modality e2e-tgt 
--submit no --padding_mode block --app "--predict_xstart True --training_mode e2e 
--vocab_size 821  --e2e_train ../datasets/e2e_data " --notes xstart_e2e"""


def main():
	args = create_argparser().parse_args()  # 用本文件定义的create_argparser函数
	set_seed(args.seed)  # 设置随机数种子
	dist_util.setup_dist()  # 多卡运行，DEBUG **
	logger.configure()

	logger.log("creating model and diffusion...")
	# 创建模型
	model, diffusion = create_model_and_diffusion(
		**args_to_dict(args, model_and_diffusion_defaults().keys())
	)  # 传入的参数是字典（关键字参数, 默认参数字典）
	print(f"types of model and diffusion are {type(model)} and {type(diffusion)}, respectively")
	model.to(dist_util.dev())  # DEBUG **

	pytorch_total_params = sum(p.numel() for p in model.parameters())

	logger.log(f'the parameter count is {pytorch_total_params}')  # 打印模型参数量
	schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)  # "schedule_sampler": "uniform"

	# 保存训练参数 training_args.json
	logger.log(f'saving the hyper-parameters to {args.checkpoint_path}/training_args.json')
	with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
		json.dump(args.__dict__, f, indent=2)
		f.close()

	# 创建wandb记事本，类似于SummaryWriter
	wandb.init(
		project=os.getenv("WANDB_PROJECT", "diffusion_lm"),
		name=args.checkpoint_path,
	)
	wandb.config.update(args.__dict__, allow_val_change=True)

	if args.experiment_mode == 'conditional_gen':
		assert args.modality in ['e2e']
		assert args.padding_mode == 'pad'

	logger.log("creating data loader...")
	# 未执行if分支代码
	if args.modality == 'image':  # args.modality == "e2e-tgt"
		data = load_data(
			data_dir=args.data_dir,
			batch_size=args.batch_size,  # 64
			image_size=args.image_size,  # useless param for language model
			class_cond=args.class_cond,  # False
		)
		data_valid = None

	else:  # 执行的是这部分代码
		print('load data', '*' * 50)
		if args.modality == 'roc-aug' or args.modality == 'commonGen-aug':  # modality == "e2e-tgt"
			tokenizer = load_tokenizer(args.modality, args.experiment,
									   'predictability/diffusion_models_v7/diff_roc_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart')
			rev_tokenizer = {v: k for k, v in tokenizer.items()}
			print(len(rev_tokenizer), 'loading from tokenizer. ')
		elif args.use_bert_tokenizer == 'yes':  # use_bert_tokenizer == "no"，默认为no
			rev_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
		else:  # 执行的是这个分支
			rev_tokenizer = None
		print(f"the type of rev_tokenizer is {type(rev_tokenizer)}")

		if args.experiment == "random1":  # args.experiment == "random"
			args.experiment = "random"
			print('loading from the vocabs here.')
			assert args.in_channel == 64
			assert args.modality == "roc"
			model22 = torch.nn.Embedding(args.vocab_size, args.in_channel)
			model22_weight = torch.load(
				'predictability/diffusion_models_v7/diff_roc-aug_pad_rand64_'
				'transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart_e2e/'
				'ema_0.9999_200000.pt', map_location='cpu'
			)['word_embedding.weight']
			model22.weight = model22_weight
			model22.weight.requires_grad = False
		else:  # 执行的是这个分支
			model22 = None
		print(f"the type of model22 is {type(model22)}")

		# 生成训练集的dataloader
		data = load_data_text(
			data_dir=args.data_dir,  # data_dir是无用参数
			batch_size=args.batch_size,  # 64
			image_size=args.image_size,  # 8, image_size ** 2 == seqlen
			class_cond=args.class_cond,  # class_cond也是无用参数
			data_args=args,
			task_mode=args.modality,  # e2e-tgt
			padding_mode=args.padding_mode,  # block, pad
			load_vocab=rev_tokenizer,  # 默认为None
			model=model22,  # 默认为None
		)  # dataloader生成器, split默认为train
		next(data)  # 这一步意义不明，可能是忘了打印或者不需要第一批数据
		model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
										args.checkpoint_path, extra_args=args)  # model_name_or_path是无用参数
		# model2是torch.embedding，tokenizer是字典
		print(f"types of model2 and tokenizer are {type(model2)} and {type(tokenizer)}, respectively")
		if args.modality == 'book' or args.use_bert_tokenizer == 'yes':
			rev_tokenizer = tokenizer  # BERT tokenizer BPE.
		else:  # 执行这个分支
			rev_tokenizer = {v: k for k, v in tokenizer.items()}
		print(f"rev_tokenizer updated, its type is {type(rev_tokenizer)} now!")

		data_valid = load_data_text(
			data_dir=args.data_dir,
			batch_size=args.batch_size,
			image_size=args.image_size,
			class_cond=args.class_cond,
			data_args=args,
			task_mode=args.modality,
			padding_mode=args.padding_mode,  # block, pad
			split='valid',
			load_vocab=rev_tokenizer,
			model=model2,
		)  # 划分验证集

	# dist.barrier()
	# import time
	# while not os.path.exists(os.path.join(args.checkpoint_path, 'vocab.json')):
	#     time.sleep(1)
	# 定义一个将{单词}映射为{词向量}的函数，直接对diffusion的mapping函数进行替换
	def get_mapping_func(args, diffusion, data):
		model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
										args.checkpoint_path, extra_args=args)  # model2是torch.embedding
		model3 = get_weights(model2, args)  # 将model2的weight参数锁死，禁止更新。model3也是torch.embedding
		print(model3, model3.weight.requires_grad)
		mapping_func = partial(compute_logp, args, model3.cuda())
		diffusion.mapping_func = mapping_func
		return mapping_func

	get_mapping_func(args, diffusion, data)

	logger.log("training...")
	TrainLoop(
		model=model,  # <class 'improved_diffusion.transformer_model2.TransformerNetModel2'>
		diffusion=diffusion,  # <class 'improved_diffusion.respace.SpacedDiffusion'>
		data=data,
		batch_size=args.batch_size,
		microbatch=args.microbatch,
		lr=args.lr,
		ema_rate=args.ema_rate,
		log_interval=args.log_interval,
		save_interval=args.save_interval,
		resume_checkpoint=args.resume_checkpoint,
		use_fp16=args.use_fp16,
		fp16_scale_growth=args.fp16_scale_growth,
		schedule_sampler=schedule_sampler,
		weight_decay=args.weight_decay,
		lr_anneal_steps=args.lr_anneal_steps,
		checkpoint_path=args.checkpoint_path,
		gradient_clipping=args.gradient_clipping,
		eval_data=data_valid,
		eval_interval=args.eval_interval
	).run_loop()


def create_argparser():
	defaults = dict(
		data_dir="",
		schedule_sampler="uniform",
		lr=1e-4,
		weight_decay=0.0,
		lr_anneal_steps=0,
		batch_size=1,
		microbatch=-1,  # -1 disables microbatches
		ema_rate="0.9999",  # comma-separated list of EMA values
		log_interval=50,
		save_interval=50000,
		resume_checkpoint="",
		use_fp16=False,
		fp16_scale_growth=1e-3,
		seed=101,
		gradient_clipping=-1.0,
		eval_interval=2000,
		checkpoint_path='diff_models'
	)
	text_defaults = dict(
		modality='text',
		dataset_name='wikitext',
		dataset_config_name='wikitext-2-raw-v1',
		config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml',
		model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
		experiment='gpt2_pre_compress',
		model_arch='conv-unet',
		roc_train='diffusion_lm/ROCstory',  # 'diffusion_lm/ROCstory/ROCstory17.csv',
		wiki_train='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
		e2e_train='e2e_data',
		yelp_train='diffusion_lm/yelpnlg-resources/yelpnlg-corpus',
		commonGen_train='diffusion_lm/common-gen/commongen_data',
		emb_scale_factor=1.0,
		noise_level=0.0,
		cache_mode='no',
		use_bert_tokenizer='no',
		padding_mode='block',
		preprocessing_num_workers=1
	)
	defaults.update(model_and_diffusion_defaults())
	defaults.update(text_defaults)
	# 把配置更新了一下
	parser = argparse.ArgumentParser()
	add_dict_to_argparser(parser, defaults)
	return parser


if __name__ == "__main__":
	main()
