import wandb
import random
import time

print("Attempting to connect to Weights & Biases...")

try:
    # 1. Initialize a new wandb run
    # This is the primary step that tests the connection and authentication.
    # A project name is specified to organize your runs.
    run = wandb.init(
        project="connectivity-test-project",
        notes="A simple script to verify wandb connectivity."
    )

    print("✅ Successfully initialized wandb run.")
    print(f"🚀 Your run is live at: {run.url}")

    # 2. Log some dummy data to the run
    # This confirms that data can be successfully transmitted.
    print("\nLogging a few steps of dummy data...")
    for step in range(10):
        # Simulate metrics
        loss = random.uniform(0.1, 0.9)
        accuracy = 1 - loss + random.uniform(-0.1, 0.1)
        
        # Log metrics to wandb
        wandb.log({"loss": loss, "accuracy": accuracy})
        
        print(f"  - Step {step}: Logged loss and accuracy.")
        time.sleep(0.5) # A small delay to simulate a real process

    print("✅ Successfully logged data.")

finally:
    # 3. Finish the run
    # This is a crucial step to ensure the run is properly closed and all data is saved.
    # The 'finally' block ensures this happens even if an error occurs above.
    if 'run' in locals() and run is not wandb.run:
        run.finish()
        print("\n✅ Successfully finished and synced the run.")
        print("\n🎉 Your wandb connectivity test was successful!")
    else:
        # This part handles cases where wandb.init() failed
        print("\n❌ The wandb run could not be initialized.")
        print("Please check your setup by following the steps below.")