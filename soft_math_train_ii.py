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
            #from math_verify import parse, verify # type: ignore
            #math_verify untrustworthy in multiprocessing context; signalerrors about timeouts.
            fieldselection = self.parser.parse_answer_from_completion(completion, selection_field="answer")
            #low-dependency math_python.py 'verification'.
            if fieldselection is not None:
                if answer == fieldselection:
                    return 1.01
                else:
                    return 0.01
                #return 1.01 if verify(parse(answer), parse(fieldselection)) else 0.01 #so the thing is. wrong answers are ontologically different from typeerror exceptions.
            if self.logblast<12:
                print(f"""None-typed parse without exception. is this because your parser is wack?
                completion:{repr(completion)[:80]}
                raw_parse:{repr(self.parser.parse(completion))[:80]}""")
                self.logblast +=1
            return 0.001 #you get a tiny reward for not triggering exceptions at least.
        except Exception as excy:
            #crucial debugging string below
            if self.logblast<12:
                print(f"slow down there cowboy you're yeehawing a solution you can't be rightly held to:{excy}")
                self.logblast +=1
                raise excy
            return 0.0

def format_reward_func(completion):
    """Reward function that checks if each step follows the expected format."""
    """
    msg is annoying wide dtype:
    [
        msg@idx:{'role':'rolename',
        'content':str},
    ]
    """
    asst_role = "assistant"
    model_messages = [item["content"] for item in completions if item["role"]=="assistant"]
    if not model_messages:
        return 0.0
    
    # Calculate format adherence for each message
    format_scores = []
    for msg in model_messages:
        #content = msg['content']
        content = msg
        parsed = self.parse(content)
        parsed_no_strip = self.parse(content, strip=False)
        
        # Check if the message has at least one valid field
        has_any_field = False
        fields_with_content = 0
        total_fields = 0
        
        # Keep track of which expected fields are present
        expected_field_count = len(self._fields)  # Total number of expected field sets
        present_field_sets = set()  # Which field sets have at least one alternative present
        malformed_field_sets = set()   #which field sets are totally scrungled
        
        has_correct_spacing=True
        for i, (canonical, alternatives) in enumerate(self._fields):
            field_set_present = False
            for alt in alternatives:
                if hasattr(parsed, alt) and getattr(parsed, alt) is not None:
                    has_any_field = True
                    fields_with_content += 1
                    total_fields += 1
                    field_set_present = True
                # Check if field exists in non-stripped version too (proper spacing)
                    if not (hasattr(parsed_no_strip, alt) and 
                            getattr(parsed_no_strip, alt) is not None):
                        has_correct_spacing = False
                elif content.count(f"<{alt}>") > 0 or content.count(f"</{alt}>") > 0:
                    # Tag exists but content wasn't properly parsed
                    total_fields += 1
                    field_set_malformed = True
            
            # If any alternative from this field set was present, count it
            if field_set_present:
                present_field_sets.add(i)
            if field_set_malformed:
                malformed_field_sets.add(i)
        
        total_malfies = 0
        for malfs in malformed_field_sets:
            scrungled_alts = self._fields[malf][1]
            for alt in scrungled_alts:
                total_malfies += content.count(f"<{alt}>")
                total_malfies += content.count(f"</{alt}>")

        # Calculate format score components
        format_score = 0.0
        
        # Weight the score based on different criteria
        if has_any_field:
            # Calculate the proportion of expected field sets that are present
            field_malformation_ratio = total_fields / total_fields + total_malfies
            format_score += 0.4 * field_malformation_ratio
        
        format_scores.append(format_score)
    
    # Return average format adherence
    if not format_scores:
        return 0.0
    return (sum(format_scores) / len(format_scores))

return format_reward_func

import verifiers as vf

if __name__ == "__main__":
parser = vf.XMLParser(['thinkies', 'answer']) # <think>...</think>\n<answer>...</answer>


#soft_math_rubric.py
#basically: slices from end of last '<answer>' to start of last '</answer>'.
#then parses (a entire external library checking for mathematical notation equivalency)
#and returns a reward of 0 for a parsing error, 0.01 for a wrong answer, 1.01 for a right answer.
rubric = SoftMathRubric( 
    funcs = [],  #class-syntactic reward functions, see source
    weights = [],    #add_reward_func from root class implicitly populates weights list, see source
    parser = parser
)


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

#from verifiers.trainers.grpo_config import GRPOConfig
#uhh calibrated for 4xGPU node. good luck!
#batchsize 32/32 blows up 4xA100.
#trying 16/16...
training_args=vf.trainers.GRPOConfig(
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
    beta=0.001,    #suspicious
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