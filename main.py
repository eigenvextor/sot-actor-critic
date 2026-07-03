import torch
import os
from model import ActorCriticTracker
from train import train_offline_ddpg
from data import get_otb_dataloaders

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"executing on device: {device}")

    OTB_ROOT_DIR = "OTB2015"  
    TOTAL_ITERATIONS = 500000
    CHECKPOINT_ITERATIONS = 25000
    BATCH_SIZE = 64

    print("initializing ACT model")
    model = ActorCriticTracker().to(device)

    print("loading OTB dataset")
    if not os.path.exists(OTB_ROOT_DIR):
        raise FileNotFoundError(f"cannot find OTB dataset at {OTB_ROOT_DIR}")
        
    train_loader, test_loader = get_otb_dataloaders(
        otb_root_dir=OTB_ROOT_DIR, 
        split_ratio=1
    )

    try:
        train_offline_ddpg(
            model=model,
            train_loader=train_loader,
            device=device,
            total_iterations=TOTAL_ITERATIONS,
            checkpoint_iterations=CHECKPOINT_ITERATIONS,
            batch_size=BATCH_SIZE
        )
    except KeyboardInterrupt:
        print("\ntraining manually interrupted. saving emergency checkpoint")
        torch.save(model.state_dict(), "checkpoints/emergency_stop.pth")
        print("emergency checkpoint saved")