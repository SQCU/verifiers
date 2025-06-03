# train.py
# a generic or even hypothetical script to be elaborated thru the protocol suggested in readme.md

import verifiers as vf
parser = vf.XMLParser(['thinkies', 'answer']) # <think>...</think>\n<answer>...</answer>

"""
rubric = vf.Rubric(
	your_custom_reward_func, # def func(prompt, completion, answer, **kwargs)
	parser.get_format_reward_func(),
weights=[1.0, 0.2])
"""

#soft_math_rubric.py
#basically: slices from end of last '<answer>' to start of last '</answer>'.
#then parses (a entire external library checking for mathematical notation equivalency)
#and returns a reward of 0 for a parsing error, 0.01 for a wrong answer, 1.01 for a right answer.
rubric = vf.SoftMathRubric( 
    funcs = [],  #class-syntactic reward functions, see source
    weights = [],    #add_reward_func from root class implicitly populates weights list, see source
    parser = parser
)

"""
vf_env = vf.SingleTurnEnv(
	dataset=..., # hf Dataset with 'question' + 'answer' columns
	system_prompt=f"... Respond in the following format: {parser.get_format_str()}",    
	rubric
)
"""

MATH_PROMPT = f"""
answer the following by thinking carefully and diligently, step by step, until you reach an answer.
only your very last answer is graded, so it's very important to follow the attached format:
{parser.get_format_str()}
remember to do your best on the following task!
"""

#yeah we're shidding and farding and pooping our pands with this one
oneshot_dset = vf.load_example_dataset("One-Shot-RLVR-Datasets")

vf_env = vf.SingleTurnEnv(
	dataset=oneshot_dset, # hf Dataset with 'question' + 'answer' columns
	system_prompt=MATH_PROMPT,    
	rubric
)

model_name = "Qwen/Qwen3-0.6B-Base"
run_name = "somath-" + model_name.split("/")[-1].lower()

#uhh calibrated for 4xGPU node. good luck!
training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=20,
    num_train_epochs=1,
    temperature=0.8,
    min_p=0.01,
    max_steps=1000,
    bf16=True,
    max_grad_norm=0.01,
    num_iterations=2,
    beta=0.002,
    epsilon=0.2,    #clipping value used in PPO algorithm!
    epsilon_high=0.28,  #dapo paper
    max_prompt_length=1024,
    max_completion_length=2048,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_generations=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=100,
    eval_accumulation_steps=1,
    eval_on_start=False,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    vllm_server_host="0.0.0.0", # replace with your inference server's host for multi-node setups
    vllm_server_port=8000,
    vllm_gpu_memory_utilization=0.9,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to=None, #stop depending on external apis
    reward_weights=vf_env.get_reward_weights(),
    loss_type="dr_grpo",
    mask_truncated_completions=True,
)

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args
)
trainer.train()