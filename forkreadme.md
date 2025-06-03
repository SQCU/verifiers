#forkreadme.md

### Upstream merge notes!
tracking `fe8bd1d63862444f5bf9bcf338acc50a14ced6e9`

many features are funky and different!

crucially you call vllm inside of this project through `vf-vllm`, which blah blah blah `/verifiers/inference/vllm_server.py->cli_main():`. make sense? yeah, exactly. what this means for us is that the shell scripts we use should all be rewritten. 


### Fork notes!
there are a collection of shell scripts about the root directory! they probably do things.

math_train_qw3_0.6b.py has very peculiar results that demand further study;

particularly, it is revealed that tool-calling is far more easily learned through the suggested training template... than use of the answer output formats needed for math-correctness-score-based verification! this is a surprising or perhaps even stunning result, and suggests further development of continuous, soft, overlapping, or curriculum verifiers in further experiments.

### End fork notes!