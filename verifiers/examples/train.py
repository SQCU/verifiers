# train.py
# a generic or even hypothetical script to be elaborated thru the protocol suggested in readme.md

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=vf.grpo_defaults(run_name="...")
)
trainer.train()