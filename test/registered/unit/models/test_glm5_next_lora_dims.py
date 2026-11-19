"""GLM-5.3-Flash LoRA buffer dims: KDA and DSA layers size o_proj differently."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.lora.utils import (
    ATTN_TP_LORA_MODULE_NAMES,
    REPLICATED_LINEAR_LORA_NAMES,
)
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration as Model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# zai-org/GLM-5.3-Flash text config: 45 layers, every 4th layer (3, 7, ...) is DSA
KDA_LAYERS = [i for i in range(45) if i % 4 != 3]
CONFIG = Glm5NextTextConfig(
    hidden_size=4096,
    num_attention_heads=64,
    v_head_dim=256,
    intermediate_size=12288,
    moe_intermediate_size=2048,
    n_shared_experts=1,
    first_k_dense_replace=3,
    linear_attn_config={
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "kda_layers": KDA_LAYERS,
        "full_attn_layers": [i for i in range(45) if i % 4 == 3],
    },
)


def dims(module_name, layer_idx):
    return Model.get_hidden_dim(SimpleNamespace(config=CONFIG), module_name, layer_idx)


class TestGlm5NextLoRADims(CustomTestCase):
    def test_o_proj_per_layer_type(self):
        self.assertEqual(dims("o_proj", 0), (64 * 128, 4096))  # KDA
        self.assertEqual(dims("o_proj", 3), (64 * 256, 4096))  # DSA

    def test_mlp_down_proj(self):
        self.assertEqual(dims("down_proj", 0), (12288, 4096))  # dense layer
        self.assertEqual(dims("down_proj", 4), (2048, 4096))  # shared expert

    def test_kda_projection_targets(self):
        self.assertEqual(dims("qkv_proj", 0), (4096, 3 * 8192))
        self.assertEqual(dims("b_proj", 0), (4096, 64))
        self.assertEqual(dims("f_b_proj", 0), (128, 8192))
        self.assertTrue({"f_a_proj", "g_a_proj"} <= set(REPLICATED_LINEAR_LORA_NAMES))
        self.assertTrue({"b_proj", "f_b_proj", "g_b_proj"} <= ATTN_TP_LORA_MODULE_NAMES)


if __name__ == "__main__":
    unittest.main()
