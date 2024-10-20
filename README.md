# HumanEval Java ja

This code is a Japanese translation of the evaluation data with only Java evaluation sites extracted.

original: https://github.com/deepseek-ai/DeepSeek-Coder/tree/main/Evaluation

## example
- model: [grapevine-AI/gemma-2-2b-jpn-it-gguf](https://huggingface.co/grapevine-AI/gemma-2-2b-jpn-it-gguf/resolve/main/gemma-2-2B-jpn-it-IQ4_XS.gguf)
- server: [llama.cpp](https://github.com/ggerganov/llama.cpp)

```bash
$ python human_eval_java.py  --show
generating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [21:29<00:00,  8.21s/it]
compile and run: 157it [04:07,  1.57s/it]
pass@1: 0.24203821656050956
HumanEval_23_strlen passed
HumanEval_89_encrypt failed: wrong answer
HumanEval_95_check_dict_case failed: wrong answer
HumanEval_85_add failed: wrong answer
HumanEval_140_fix_spaces failed: wrong answer
HumanEval_63_fibfib failed: wrong answer
HumanEval_151_double_the_difference failed: wrong answer
HumanEval_22_filter_integers passed
:
HumanEval_9_rolling_max failed: Exception in thread "main" java.lang.IndexOutOfBoundsException: Index 0 out of bounds for length 0
        at java.base/jdk.internal.util.Preconditions.outOfBounds(Preconditions.java:64)
        at java.base/jdk.internal.util.Preconditions.outOfBoundsCheckIndex(Preconditions.java:70)
        at java.base/jdk.internal.util.Preconditions.checkIndex(Preconditions.java:266)
        at java.base/java.util.Objects.checkIndex(Objects.java:361)
        at java.base/java.util.ArrayList.get(ArrayList.java:427)
        at Problem.rollingMax(Problem.java:14)
        at Problem.main(Problem.java:24)
:
```