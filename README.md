QLora fine tuned LLM for better performance on maths word problems
-Used “microsoft/orca-math-word-problems-200k” dataset from HuggingFace after preprocessing it as per model’s required input prompt
-Used “databricks/dolly-v2-3b” model in 4 bit quantization using HuggingFace
-PEFT fine tuned the model using Quantized-LORA technique
