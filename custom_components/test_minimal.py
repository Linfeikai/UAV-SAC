import torch
from stable_baselines3.common.distributions import (
    CategoricalDistribution as SB3CategoricalDistribution,
)
import torch.distributions as torch_dist

print("\n--- Minimal Categorical Test ---")

# 1. Create sim_discrete_logits (same way as in your main test)
# Assuming discrete_logits_net and example_latent_pi are still defined and on DEVICE
if "discrete_logits_net" in locals() and discrete_logits_net is not None:
    sim_discrete_logits_minimal_test = discrete_logits_net(example_latent_pi)
    print("sim_discrete_logits_minimal_test:", sim_discrete_logits_minimal_test)
    print(
        "sim_discrete_logits_minimal_test grad_fn:",
        sim_discrete_logits_minimal_test.grad_fn,
    )

    # 2. Test with SB3 CategoricalDistribution
    sb3_cat_dist = SB3CategoricalDistribution(
        action_dim=discrete_size
    )  # discrete_size from your main test
    sb3_cat_dist.proba_distribution(sim_discrete_logits_minimal_test)
    sb3_internal_logits = sb3_cat_dist.distribution.logits
    print(
        "Logits from SB3 CategoricalDistribution's internal torch.dist.Categorical:",
        sb3_internal_logits,
    )
    print(
        "Logits grad_fn from SB3 CategoricalDistribution:", sb3_internal_logits.grad_fn
    )
    assert torch.allclose(sb3_internal_logits, sim_discrete_logits_minimal_test), (
        "SB3 CatDist logits mismatch!"
    )
    print("Minimal test with SB3 CategoricalDistribution PASSED (if no assert error).")

    # 3. Test directly with torch.distributions.Categorical
    torch_cat_direct = torch_dist.Categorical(logits=sim_discrete_logits_minimal_test)
    torch_direct_logits = torch_cat_direct.logits
    print("Logits from direct torch.distributions.Categorical:", torch_direct_logits)
    print(
        "Logits grad_fn from direct torch.distributions.Categorical:",
        torch_direct_logits.grad_fn,
    )
    assert torch.allclose(torch_direct_logits, sim_discrete_logits_minimal_test), (
        "Direct torch CatDist logits mismatch!"
    )
    print(
        "Minimal test with direct torch.distributions.Categorical PASSED (if no assert error)."
    )

else:
    print("Skipping minimal categorical test as discrete_logits_net is not available.")
