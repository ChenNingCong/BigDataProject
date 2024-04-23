from transformers import AutoTokenizer
"""
AutoTokenizer will use a fast tokenizer (implemented in Rust) by default,
while BertTokenizer is slow.
"""
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")