import torch
import torch.nn as nn
import gymnasium  # 完整导入 gymnasium

# from gymnasium import spaces # 单独导入 spaces 以便使用 spaces.Tuple 等 (或者直接用 gymnasium.spaces)
import numpy as np

# 假设你的 hybrid_distribution.py 文件可以被导入
# 确保这个文件与测试脚本在同一目录，或者在 PYTHONPATH 中
from hybrid_distribution import make_hybrid_proba_distribution, HybridDistribution

# 从 SB3 导入 LOG_STD 常量，以便在测试中断言中使用相同的值
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
)  # 用于类型检查

# --- 0. 定义测试用的参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 定义一个混合动作空间: Tuple(Discrete(3), Box(shape=(2,)))
discrete_size = 3
continuous_shape = (2,)
# 使用 gymnasium.spaces.Tuple
action_space = gymnasium.spaces.Tuple(
    (
        gymnasium.spaces.Discrete(discrete_size),
        gymnasium.spaces.Box(low=-1, high=1, shape=continuous_shape, dtype=np.float32),
    )
)
print(f"Test Action Space: {action_space}")

# 模拟 Actor 输出的 latent_pi 的维度
latent_dim = 64
# 模拟一批 latent_pi (例如 batch_size=4)
batch_size = 4
example_latent_pi = torch.randn(batch_size, latent_dim).to(DEVICE)

# --- 1. 测试 make_hybrid_proba_distribution ---
print("\n--- 1. Testing make_hybrid_proba_distribution ---")
hybrid_dist_instance = None  # 初始化以备后续检查
try:
    hybrid_dist_instance = make_hybrid_proba_distribution(action_space)
    assert isinstance(hybrid_dist_instance, HybridDistribution), (
        "make_hybrid_proba_distribution did not return a HybridDistribution instance."
    )
    assert hybrid_dist_instance.discrete_dim == discrete_size, (
        f"Discrete dim mismatch: Expected {discrete_size}, Got {hybrid_dist_instance.discrete_dim}"
    )
    assert hybrid_dist_instance.continuous_dim == np.prod(continuous_shape).astype(
        int
    ), (
        f"Continuous dim mismatch: Expected {np.prod(continuous_shape).astype(int)}, Got {hybrid_dist_instance.continuous_dim}"
    )
    print("PASSED: make_hybrid_proba_distribution")
except Exception as e:
    print(f"FAILED: make_hybrid_proba_distribution - {e}")


# --- 2. 测试 HybridDistribution.proba_distribution_net ---
print("\n--- 2. Testing HybridDistribution.proba_distribution_net ---")
discrete_logits_net, continuous_mean_net, continuous_log_std_net = (
    None,
    None,
    None,
)  # 初始化
if hybrid_dist_instance is not None:
    try:
        log_std_init_val = -2.5
        (
            discrete_logits_net_test,
            continuous_mean_net_test,
            continuous_log_std_net_test,
        ) = hybrid_dist_instance.proba_distribution_net(
            latent_dim=latent_dim, log_std_init=log_std_init_val
        )
        discrete_logits_net, continuous_mean_net, continuous_log_std_net = (
            discrete_logits_net_test,
            continuous_mean_net_test,
            continuous_log_std_net_test,
        )

        assert isinstance(discrete_logits_net, nn.Module), (
            "discrete_logits_net is not an nn.Module."
        )
        assert isinstance(continuous_mean_net, nn.Module), (
            "continuous_mean_net is not an nn.Module."
        )
        assert isinstance(continuous_log_std_net, nn.Module), (
            "continuous_log_std_net is not an nn.Module."
        )

        discrete_logits_net.to(DEVICE)
        continuous_mean_net.to(DEVICE)
        continuous_log_std_net.to(DEVICE)

        test_logits = discrete_logits_net(example_latent_pi)
        test_mean = continuous_mean_net(example_latent_pi)
        test_log_std = continuous_log_std_net(example_latent_pi)

        assert test_logits.shape == (batch_size, discrete_size), (
            f"Discrete logits shape mismatch: Expected {(batch_size, discrete_size)}, Got {test_logits.shape}"
        )
        assert test_mean.shape == (batch_size, np.prod(continuous_shape).astype(int)), (
            f"Continuous mean shape mismatch: Expected {(batch_size, np.prod(continuous_shape).astype(int))}, Got {test_mean.shape}"
        )
        assert test_log_std.shape == (
            batch_size,
            np.prod(continuous_shape).astype(int),
        ), (
            f"Continuous log_std shape mismatch: Expected {(batch_size, np.prod(continuous_shape).astype(int))}, Got {test_log_std.shape}"
        )

        if (
            hasattr(continuous_log_std_net, "bias")
            and continuous_log_std_net.bias is not None
        ):
            expected_bias = torch.full_like(
                continuous_log_std_net.bias, log_std_init_val
            )
            assert torch.allclose(continuous_log_std_net.bias, expected_bias), (
                f"continuous_log_std_net bias not initialized correctly. Expected all {log_std_init_val}, Got {continuous_log_std_net.bias.data}"
            )
        print("PASSED: HybridDistribution.proba_distribution_net")

    except Exception as e:
        print(f"FAILED: HybridDistribution.proba_distribution_net - {e}")
else:
    print("SKIPPED: proba_distribution_net tests (hybrid_dist_instance is None).")


# --- 3. 测试 HybridDistribution.proba_distribution (设置参数) ---
print("\n--- 3. Testing HybridDistribution.proba_distribution ---")
params_set_successfully = False
sim_discrete_logits, sim_continuous_mean, sim_continuous_log_std = None, None, None
if (
    hybrid_dist_instance
    and discrete_logits_net
    and continuous_mean_net
    and continuous_log_std_net
):
    try:
        sim_discrete_logits = discrete_logits_net(example_latent_pi)
        sim_continuous_mean = continuous_mean_net(example_latent_pi)
        sim_continuous_log_std = continuous_log_std_net(example_latent_pi)

        # DEBUG: Print received logits inside proba_distribution
        print("Calling hybrid_dist_instance.proba_distribution with:")
        print(
            f"  sim_discrete_logits (grad_fn: {sim_discrete_logits.grad_fn}):\n{sim_discrete_logits.detach().cpu().numpy()}"
        )

        hybrid_dist_instance.proba_distribution(
            sim_discrete_logits, sim_continuous_mean, sim_continuous_log_std
        )

        assert hybrid_dist_instance._cur_cat_dist is not None, (
            "_cur_cat_dist was not set."
        )
        assert isinstance(
            hybrid_dist_instance._cur_cat_dist, torch.distributions.Categorical
        ), "_cur_cat_dist is not a torch.distributions.Categorical."
        assert hybrid_dist_instance._cur_squashed_gauss_dist is not None, (
            "_cur_squashed_gauss_dist was not set."
        )
        assert isinstance(
            hybrid_dist_instance._cur_squashed_gauss_dist,
            SquashedDiagGaussianDistribution,
        ), "_cur_squashed_gauss_dist is not a SquashedDiagGaussianDistribution."

        # 验证离散部分：比较概率
        probs_from_sim_logits = torch.softmax(sim_discrete_logits, dim=-1)
        probs_from_internal_dist = hybrid_dist_instance._cur_cat_dist.probs
        assert torch.allclose(
            probs_from_internal_dist, probs_from_sim_logits, atol=1e-7
        ), (
            "Probabilities from internal Categorical distribution do not match probabilities from input sim_discrete_logits."
        )
        print("Discrete part probabilities match.")

        # 验证连续部分
        # _cur_squashed_gauss_dist is the SB3 SquashedDiagGaussianDistribution instance
        # its .distribution attribute is the underlying torch.distributions.Normal
        underlying_normal_dist = (
            hybrid_dist_instance._cur_squashed_gauss_dist.distribution
        )
        assert isinstance(underlying_normal_dist, torch.distributions.Normal), (
            "Internal continuous distribution is not torch.distributions.Normal"
        )

        internal_gaussian_mean = underlying_normal_dist.loc
        internal_gaussian_std = underlying_normal_dist.scale

        expected_std = torch.exp(
            torch.clamp(sim_continuous_log_std, LOG_STD_MIN, LOG_STD_MAX)
        )

        assert torch.allclose(internal_gaussian_mean, sim_continuous_mean, atol=1e-7), (
            "Continuous mean in underlying Gaussian distribution does not match input."
        )
        assert torch.allclose(internal_gaussian_std, expected_std, atol=1e-7), (
            "Continuous std in underlying Gaussian distribution does not match clamped and exponentiated log_std."
        )
        print("  Continuous part parameters match.")

        print("PASSED: HybridDistribution.proba_distribution")
        params_set_successfully = True
    except Exception as e:
        print(f"FAILED: HybridDistribution.proba_distribution - {e}")
        import traceback

        traceback.print_exc()  # Print full traceback for the error
else:
    print("SKIPPED: proba_distribution tests (prerequisites not met).")


# --- 4. 测试采样: sample() 和 mode() ---
print("\n--- 4. Testing HybridDistribution.sample() and mode() ---")
disc_sample, cont_sample, disc_mode, cont_mode = None, None, None, None
sampling_tests_passed = False
if hybrid_dist_instance and params_set_successfully:
    try:
        disc_sample, cont_sample = hybrid_dist_instance.sample()
        assert disc_sample.shape == (batch_size,), (
            f"Discrete sample shape mismatch: Expected {(batch_size,)}, Got {disc_sample.shape}"
        )
        assert disc_sample.dtype == torch.int64, (
            f"Discrete sample dtype mismatch: Expected torch.int64, Got {disc_sample.dtype}"
        )
        assert ((disc_sample >= 0) & (disc_sample < discrete_size)).all(), (
            "Discrete sample out of bounds."
        )

        assert cont_sample.shape == (
            batch_size,
            np.prod(continuous_shape).astype(int),
        ), f"Continuous sample shape mismatch."
        assert ((cont_sample >= -1.0001) & (cont_sample <= 1.0001)).all(), (
            "Continuous sample (squashed) out of [-1, 1] bounds."
        )
        print("  sample() basic checks PASSED.")

        disc_mode, cont_mode = hybrid_dist_instance.mode()
        assert disc_mode.shape == (batch_size,), f"Discrete mode shape mismatch."
        assert disc_mode.dtype == torch.int64, f"Discrete mode dtype mismatch."
        assert ((disc_mode >= 0) & (disc_mode < discrete_size)).all(), (
            "Discrete mode out of bounds."
        )

        assert cont_mode.shape == (batch_size, np.prod(continuous_shape).astype(int)), (
            f"Continuous mode shape mismatch."
        )
        assert ((cont_mode >= -1.0001) & (cont_mode <= 1.0001)).all(), (
            "Continuous mode (squashed) out of [-1, 1] bounds."
        )

        if sim_continuous_mean is not None:
            expected_cont_mode = torch.tanh(sim_continuous_mean)
            assert torch.allclose(cont_mode, expected_cont_mode, atol=1e-6), (
                f"Continuous mode does not match tanh(mean). Got {cont_mode.detach().cpu().numpy()}, expected {expected_cont_mode.detach().cpu().numpy()}"
            )
        print("  mode() basic checks PASSED.")
        print("PASSED: HybridDistribution.sample() and mode()")
        sampling_tests_passed = True
    except Exception as e:
        print(f"FAILED: Sampling/mode tests - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: sample/mode tests (prerequisites not met).")


# --- 5. 测试 log_prob() ---
print("\n--- 5. Testing HybridDistribution.log_prob() ---")
log_prob_tests_passed = False
if hybrid_dist_instance and params_set_successfully and sampling_tests_passed:
    try:
        # Ensure samples are on the correct device if they were generated on CPU/GPU and network is on another
        actions_for_log_prob = (disc_sample.to(DEVICE), cont_sample.to(DEVICE))
        log_p = hybrid_dist_instance.log_prob(actions_for_log_prob)

        assert log_p.shape == (batch_size,), (
            f"log_prob shape mismatch: Expected {(batch_size,)}, Got {log_p.shape}"
        )
        assert not torch.isnan(log_p).any(), "log_prob contains NaN."
        assert not torch.isinf(log_p).any(), "log_prob contains Inf."
        print("  log_prob() basic checks PASSED.")

        mode_actions_for_log_prob = (disc_mode.to(DEVICE), cont_mode.to(DEVICE))
        log_p_mode = hybrid_dist_instance.log_prob(mode_actions_for_log_prob)
        print(f"  Log prob of mode actions (mean): {log_p_mode.mean().item()}")
        print("PASSED: HybridDistribution.log_prob()")
        log_prob_tests_passed = True
    except Exception as e:
        print(f"FAILED: log_prob() test - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: log_prob() tests (prerequisites not met).")

# --- 6. 测试 entropy() ---
print("\n--- 6. Testing HybridDistribution.entropy() ---")
if hybrid_dist_instance and params_set_successfully:
    try:
        entropy_val = hybrid_dist_instance.entropy()
        assert entropy_val is None, f"Entropy should be None, but got {entropy_val}"
        print("PASSED: entropy() (returned None as expected).")
    except Exception as e:
        print(f"FAILED: entropy() test - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: entropy() tests (prerequisites not met).")


# --- 7. 测试 actions_from_params() 和 log_prob_from_params() ---
print("\n--- 7. Testing actions_from_params() and log_prob_from_params() ---")
if (
    hybrid_dist_instance
    and discrete_logits_net
    and continuous_mean_net
    and continuous_log_std_net
    and sim_discrete_logits is not None
    and sim_continuous_mean is not None
    and sim_continuous_log_std is not None
):
    try:
        actions_nd_disc, actions_nd_cont = hybrid_dist_instance.actions_from_params(
            sim_discrete_logits,
            sim_continuous_mean,
            sim_continuous_log_std,
            deterministic=False,
        )
        assert actions_nd_disc.shape == (batch_size,)
        assert actions_nd_cont.shape == (
            batch_size,
            np.prod(continuous_shape).astype(int),
        )
        print("  actions_from_params (non-deterministic) basic checks PASSED.")

        actions_d_disc, actions_d_cont = hybrid_dist_instance.actions_from_params(
            sim_discrete_logits,
            sim_continuous_mean,
            sim_continuous_log_std,
            deterministic=True,
        )
        if disc_mode is not None and cont_mode is not None:
            assert torch.allclose(actions_d_disc.float(), disc_mode.float()), (
                "Deterministic discrete action mismatch with mode."
            )
            assert torch.allclose(actions_d_cont, cont_mode, atol=1e-6), (
                "Deterministic continuous action mismatch with mode."
            )
        print("  actions_from_params (deterministic) basic checks PASSED.")

        (sampled_actions_tuple, log_p_from_params) = (
            hybrid_dist_instance.log_prob_from_params(
                sim_discrete_logits, sim_continuous_mean, sim_continuous_log_std
            )
        )
        assert (
            isinstance(sampled_actions_tuple, tuple) and len(sampled_actions_tuple) == 2
        ), "log_prob_from_params did not return a tuple of 2 actions."
        assert sampled_actions_tuple[0].shape == (batch_size,)
        assert sampled_actions_tuple[1].shape == (
            batch_size,
            np.prod(continuous_shape).astype(int),
        )
        assert log_p_from_params.shape == (batch_size,), (
            f"log_prob_from_params log_prob shape mismatch. Expected {(batch_size,)}, Got {log_p_from_params.shape}"
        )

        recalculated_log_p = hybrid_dist_instance.log_prob(sampled_actions_tuple)
        assert torch.allclose(log_p_from_params, recalculated_log_p, atol=1e-6), (
            "log_prob from log_prob_from_params does not match re-calculated log_prob."
        )
        print("  log_prob_from_params basic checks PASSED.")
        print("PASSED: actions_from_params() and log_prob_from_params()")

    except Exception as e:
        print(f"FAILED: actions_from_params() or log_prob_from_params() test - {e}")
        import traceback

        traceback.print_exc()
else:
    print(
        "SKIPPED: actions_from_params/log_prob_from_params tests (prerequisites not met)."
    )

print("\n--- HybridDistribution Testing Complete ---")
