# train.py
# a generic or even hypothetical script to be elaborated thru the protocol suggested in readme.md
from typing import List

from verifiers import RewardFunc
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

#upgrade mathrubric verifier?
class SoftMathRubric(Rubric):
    def __init__(self,
                 funcs: List[RewardFunc] = [],
                 weights: List[float] = [],
                 parser: XMLParser | None = None):
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        if not isinstance(self.parser, XMLParser):
            self.parser = XMLParser(fields=["thinkies", "answer"])
        self.add_reward_func(self.correct_answer_reward_func)
        self.add_reward_func(self.parser.get_format_reward_func(), weight=0.2)
        self.logblast = 0

    def correct_answer_reward_func(self, completion, answer, **kwargs) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        try:
            from math_verify import parse, verify as math_v_parse, math_v_verify # type: ignore
            #response = self.parser.parse_answer(completion)
            fieldselection = self.parser.parse_answer_from_completion(completion, selection_field="answer")
            if response is not None:
                return 1.01 if math_v_verify(math_v_parse(answer), math_v_parse(response)) else 0.01 #so the thing is. wrong answers are ontologically different from typeerror exceptions.
            if logblast<12:
                print(f"""None-typed parse without exception. is this because your parser is wack?
                completion:{repr(completion)[:80]}
                raw_parse:{repr(self.parser.parse(completion))[:80]}""")
                logblast +=1
            return 0.001 #you get a tiny reward for not triggering exceptions at least.
        except Exception as excy:
            #crucial debugging string below
            print(f"slow down there cowboy you're yeehawing a solution you can't be rightly held to:{excy}")
            raise excy
            return 0.0


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
rubric = SoftMathRubric( 
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
here's an example of a question:answer pair using the task formatting:
Louis walks past two yellow paint pots and spills one pot of blue paint onto Friselda. What color is Friselda's hair?
<thinkies>Hmm... there are two yellow pots and one blue pot, which is three pots... but the question was what color is Friselda's hair, not numbers. Blue paint makes any hair color blue. I can answer this right now!</thinkies><answer>Blue</answer>.
remember to do your best on the following task!
"""

#yeah we're shidding and farding and pooping our pands with this one
oneshot_dset = vf.load_example_dataset(name="One-Shot-RLVR-Datasets",split='pi2')
eval_dset = vf.load_example_dataset("math")

vf_env = vf.SingleTurnEnv(
	dataset=oneshot_dset, # hf Dataset with 'question' + 'answer' columns
    eval_dataset=eval_dset,
	system_prompt=MATH_PROMPT,    
	rubric=rubric
)

model_name = "Qwen/Qwen3-0.6B-Base"
run_name = "somath-II-pi2-" + model_name.split("/")[-1].lower()

from verifiers.trainers.grpo_config import GRPOConfig
#uhh calibrated for 4xGPU node. good luck!
#batchsize 32/32 blows up 4xA100.
#trying 16/16...
training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=20,
    num_train_epochs=1,
    temperature=0.8,
    min_p=0.02,
    max_steps=2000,
    bf16=True,
    max_grad_norm=0.001,    #set to kalomaze levels
    num_iterations=2,
    beta=0,    #suspicious
    epsilon=0.2,    #clipping value used in PPO algorithm!
    epsilon_high=0.28,  #dapo paper
    max_prompt_length=1024,
    max_completion_length=2048,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_generations=16,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    eval_strategy="no",  #eval strategy breaks in this trainer revision. check eval in root environment.py
    eval_steps=100,
    eval_accumulation_steps=1,
    eval_on_start=False,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    vllm_server_host="0.0.0.0", # replace with your inference server's host for multi-node setups
    vllm_server_port=8000,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to=None, #stop depending on external apis
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