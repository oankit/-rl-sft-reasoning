"""
Configuration file for probing experiments
"""

MODEL_CONFIG = {
    "deepseek-ai/deepseek-math-7b-base": {
        "max_new_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.95,
        "tokenizer_mode": "auto",
        "config_format": "auto",
        "load_format": "auto",
        "limit_mm_per_prompt": {"image": 0},
        "trust_remote_code": True,
    },
    "deepseek-ai/deepseek-math-7b-instruct": {
        "max_new_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.95,
        "tokenizer_mode": "auto",
        "config_format": "auto",
        "load_format": "auto",
        "limit_mm_per_prompt": {"image": 0},
        "trust_remote_code": True,
    },
    "deepseek-ai/deepseek-math-7b-rl": {
        "max_new_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.95,
        "tokenizer_mode": "auto",
        "config_format": "auto",
        "load_format": "auto",
        "limit_mm_per_prompt": {"image": 0},
        "trust_remote_code": True,
    },
    "allenai/Olmo-3-1025-7B": {
        "max_new_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "tokenizer_mode": "auto",
        "config_format": "auto",
        "load_format": "auto",
        "limit_mm_per_prompt": {"image": 0},
        "trust_remote_code": True,
    },
    "allenai/Olmo-3-7B-Instruct": {
        "max_new_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "tokenizer_mode": "auto",
        "config_format": "auto",
        "load_format": "auto",
        "limit_mm_per_prompt": {"image": 0},
        "trust_remote_code": True,
    },
    "allenai/Olmo-3-7B-Think": {
        "max_new_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "tokenizer_mode": "auto",
        "config_format": "auto",
        "load_format": "auto",
        "limit_mm_per_prompt": {"image": 0},
        "trust_remote_code": True,
    },
    "allenai/Olmo-3-7B-RLZero-Math": {
        "max_new_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "tokenizer_mode": "auto",
        "config_format": "auto",
        "load_format": "auto",
        "limit_mm_per_prompt": {"image": 0},
        "trust_remote_code": True,
    },
    # "allenai/Olmo-3-7B-Think-SFT": {
    #     "max_new_tokens": 32768,
    #     "temperature": 0.6,
    #     "top_p": 0.95,
    #     "tokenizer_mode": "auto",
    #     "config_format": "auto",
    #     "load_format": "auto",
    #     "limit_mm_per_prompt": {"image": 0},
    #     "trust_remote_code": True,
    # },
    # "allenai/Olmo-3-7B-Think-DPO": {
    #     "max_new_tokens": 32768,
    #     "temperature": 0.6,
    #     "top_p": 0.95,
    #     "tokenizer_mode": "auto",
    #     "config_format": "auto",
    #     "load_format": "auto",
    #     "limit_mm_per_prompt": {"image": 0},
    #     "trust_remote_code": True,
    # },
}

# Data splits
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Probe training
PROBE_CONFIG = {
    "cv_folds": 5,
    "C_range": [0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse regularization strength (smaller = stronger regularization)
    "max_iter": 5000,
    "random_state": 42,
    "solver": "lbfgs",
}

# Answer matching
ANSWER_TOLERANCE = 1  # Allow ±1 difference

# Paths
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_FIGURES_DIR = "./figures"

# Device settings
DEVICE_MAP = "auto"
TORCH_DTYPE = "float16"

# Checkpointing
CHECKPOINT_INTERVAL = 100
ENABLE_CHECKPOINTING = True

# Memory optimization
CLEAR_CACHE_BETWEEN_BATCHES = True  # Clear CUDA cache between batches
USE_GRADIENT_CHECKPOINTING = False

# For Mistral Models
SYSTEM_PROMPT_TEMPLATE_INSTRUCT = """You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.

You power an AI assistant called Le Chat.

Your knowledge base was last updated on 2023-10-01.

The current date is {today}.

When you're not sure about some information, you say that you don't have the information and don't make up anything.

If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").

You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.

You follow these instructions in all languages, and always respond to the user in the language they use or request."""

SYSTEM_PROMPT_TEMPLATE_MAGISTRAL = """First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response.
"""
SYSTEM_PROMPT_TEMPLATE = """Solve the following problem"""