import torch
import torch.multiprocessing as mp
import os
from ddp import setup, cleanup, reduce_value, gather_value, rank_barrier, spawn_nproc, get_rank

# Dummy args with port
class Args:
    port = 12355  # Change if this port is in use

def demo_test(rank, args, cfg, dataset):
    setup(rank, args=args)
    print(f"[Rank {rank}] Initialized")

    # Test reduce_value
    val = torch.tensor([rank + 1.0], device=rank)
    reduced = reduce_value(val.clone(), average=True)
    print(f"[Rank {rank}] reduce_value (average=True): {reduced.item()}")

    # Test gather_value
    gathered = gather_value(val.clone())
    print(f"[Rank {rank}] gather_value: {gathered.tolist()}")

    # Test barrier
    print(f"[Rank {rank}] Before barrier")
    rank_barrier()
    print(f"[Rank {rank}] After barrier")

    cleanup()
    print(f"[Rank {rank}] Cleaned up")

if __name__ == "__main__":
    args = Args()
    cfg = None  # Pass anything your main function expects
    dataset = None  # Same here
    spawn_nproc(demo_test, args, cfg, dataset)
