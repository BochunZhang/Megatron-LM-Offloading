# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from patch import *

import os
local_rank = int(os.environ.get('LOCAL_RANK'))

output = TextOutput(path='./tests/callgraph/results', rank=local_rank)

config = Config()
config.trace_filter.exclude.extend([
    "argparse*",
    "*<genexpr>",
    "*<lambda>",
])
config.trace_filter.include = [
    "__main__",
    "megatron*",
    "torch*",
    "gpt_builder*",
    "model_provider*",
    "tokenize*",
    "subprocess*",
    "MCore*",

    "*FP8*",
    "*CUDA*",
    "*Graph*",
    "*Config*",
    "*Tensor*",
    "*Attention*",
    "*Submodules*",

    "get_batch",
    "loss_func",
    "forward_step",
    "is_dataset_built_on_rank",
    "core_gpt_dataset_config_from_args",
    "train_valid_test_datasets_provider",
    "get_embedding_ranks",
]


from pretrain_gpt import *

if __name__ == "__main__":
    with PyCallGraph(output=output, config=config):
        # Temporary for transition to core datasets
        train_valid_test_datasets_provider.is_distributed = True

        # Optionally enable inprocess restart on pretrain
        pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

        pretrain(
            train_valid_test_datasets_provider,
            partial(model_provider, gpt_builder),
            ModelType.encoder_or_decoder,
            forward_step,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
            store=store,
            get_embedding_ranks=get_embedding_ranks,
        )
