import verifiers as vf
from verifiers.tools import python
from verifiers.utils import load_example_dataset

"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=3,4 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_train_4.py
"""

#dataset = load_example_dataset("math", split="train")
oneshot_dset = vf.load_example_dataset(name="One-Shot-RLVR-Datasets",split='pi2')

parser = vf.XMLParser(['thinkies', 'answer'], answer_field='answer')
VERIFY_PROMPT = f"""
answer the following by thinking carefully and diligently, step by step, until you reach an answer.
only your very first submitted answer is graded, so it's very important to follow the attached format:
{parser.get_format_str()}

here's an example of a question:answer pair using the task formatting:
<example>
Louis walks past two yellow paint pots and spills one pot of blue paint onto Friselda. What color is Friselda's hair?
<thinkies>Hmm... there are two yellow pots and one blue pot, which is three pots... but the question was what color is Friselda's hair, not numbers. Blue paint makes any hair color blue. I can answer this right now!</thinkies><answer>Blue</answer>.
</example>

remember to do your best on the following task!
"""

def correct_answer_reward_func(completion, answer, **kwargs) -> float:
    """Reward function that checks if the final answer matches the expected answer."""
    response = str(parser.parse_answer(completion))
    reward = 1.0 if answer == response else 0.0
    return reward

rubric = vf.Rubric(
	[correct_answer_reward_func], # def func(prompt, completion, answer, **kwargs)
	#parser.get_format_reward_func(),
    weights=[1.0],
    parser=parser)
vf_env = vf.SingleTurnEnv(
	dataset=oneshot_dset, # hf Dataset with 'question' + 'answer' columns
	system_prompt=VERIFY_PROMPT,
	rubric=rubric,
)

print(f"task system prompt:\n{vf_env.system_prompt}")

model_name = "willcb/Qwen2.5-7B-Math-Python-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "oneshot-grpo_" + model_name.split("/")[-1].lower()

training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=2
training_args.per_device_train_batch_size=8 #from 8
training_args.num_generations=8
training_args.gradient_accumulation_steps=2 #from 2

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train() 