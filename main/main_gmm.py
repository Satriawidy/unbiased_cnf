import os
import csv
import torch
import argparse
import numpy as np
from datetime import datetime
from theory import create_gmm_ndim, create_gmm_normal
from nn import MLPGMM, MLPGMMUnbias
from utils import join_paths
from train import train_step
from eval import eval_step

if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 #single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch_device = 'cpu'
    float_dtype = np.float32 #double
    torch.set_default_tensor_type(torch.DoubleTensor)
print(f"TORCH DEVICE: {torch_device}")

rng = torch.Generator(device = torch_device).manual_seed(42)

def main(args):
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        details = f"GMM_{args.network}_n{args.n}"
        run_name = args.run_name or (
            f"CNF_{details}_bs{args.bs}_lr{args.lr}_dt{args.dt}_eps{args.eps}_steps{args.steps}"
            f"_integrator_{args.integrator}_noise{args.num_noise}_{timestamp}"
        )
        output_dir = join_paths(args.main_dir, f"results/cnf_gmm/{run_name}")
    if output_dir and args.mode == "train":
        os.makedirs(output_dir, exist_ok=True)

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is not installed. Install dependencies or disable --use-wandb.") from exc
        wandb_name = args.run_name if args.run_name else os.path.basename(output_dir)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args),
            entity="lqft-flow",
        )

    if args.network == "mlpgmm":
        if args.integrator == "unbiasv2":
            model = MLPGMMUnbias(args.n, args.mlpgmm_hidden)
        else:
            model = MLPGMM(args.n, args.mlpgmm_hidden, args.num_noise)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max_scheduler, eta_min=args.lr_min)

    integrator = args.integrator
    if integrator == "unbiasv2" or integrator == "unbiasv1":
        integrator = "unbias"
    

    csv_path = join_paths(args.main_dir, f"results/cnf_result_gmm.csv")
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if args.mode == "train":
        prior = create_gmm_ndim(args.n, rng)
        action = create_gmm_normal(torch.zeros((1, args.n)))
        times = torch.arange(1.0, -args.dt*0.9, -args.dt)

        history = {
            'loss' : [],
            'logp' : [],
            'logq' : [],
            'ess' : []
        }

        for i in range(args.steps):
            train_step(model, action, prior, optimizer, history, times, integrator, args.eps,
                       args.bs, args.num_noise, args.num_checkpoint)

            if wandb_run is not None:
                wandb_data = {'loss': history['loss'][-1],
                            'logp': history['logp'][-1].mean(),
                            'logq': history['logq'][-1].mean(),
                            'ess': history['ess'][-1],
                            'lr': optimizer.param_groups[0]["lr"]}
                wandb_run.log(wandb_data, step=i + 1)
            
            if i < args.t_max_scheduler:
                scheduler.step()
        
        torch.save(model.state_dict(), f"{output_dir}/state.pt")

        results = eval_step(model, action, prior, times, integrator, "gmm", args.eps,
                       args.bs, args.num_noise, args.num_bootstrap)

        row = {
                "timestamp_eval": timestamp,
                "dt_eval": args.dt,
                "eps_eval": args.eps,
                "num_noise_eval": args.num_noise,
                "integrator_eval": args.integrator,
                "network": args.network,
                "n": args.n,
                "dt": args.dt,
                "eps": args.eps,
                "bs": args.bs,
                "integrator": args.integrator,
                "num_noise": args.num_noise,
                "hidden": args.mlpgmm_hidden,
                "num_boots": args.num_bootstrap,
                "logp_avg": results[0],
                "logp_err": results[1],
                "loss_avg": results[2],
                "loss_err": results[3],
                "part_avg": results[4],
                "part_err": results[5],
                "free_avg": results[6],
                "free_err": results[7],
                "ess_avg": results[8],
                "ess_err": results[9]
            }
    
    
    elif args.mode == "eval":
        prior = create_gmm_normal(torch.zeros((1, args.n)))
        action = create_gmm_ndim(args.n, rng)
        times = torch.arange(0.0, 1.0 + args.dt*0.9, args.dt)

        model.load_state_dict(torch.load(args.eval_path, weights_only=True))

        results = eval_step(model, action, prior, times, integrator, "gmm", args.eps,
                       args.bs, args.num_noise, args.num_bootstrap)
        
        csv_path = join_paths(args.main_dir, f"results/cnf/result.csv")
        write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        row = {
                "timestamp_eval": args.timestamp_eval,
                "dt_eval": args.dt_eval,
                "eps_eval": args.eps_eval,
                "num_noise_eval": args.num_noise_eval,
                "integrator_eval": args.integrator_eval,
                "network": args.network,
                "n": args.n,
                "dt": args.dt,
                "eps": args.eps,
                "bs": args.bs,
                "integrator": args.integrator,
                "num_noise": args.num_noise,
                "hidden": args.mlpgmm_hidden,
                "num_boots": args.num_bootstrap,
                "logp_avg": results[0],
                "logp_err": results[1],
                "loss_avg": results[2],
                "loss_err": results[3],
                "part_avg": results[4],
                "part_err": results[5],
                "free_avg": results[6],
                "free_err": results[7],
                "ess_avg": results[8],
                "ess_err": results[9]
            }

    fieldnames = [
            "timestamp_eval",
            "dt_eval",
            "eps_eval",
            "num_noise_eval",
            "integrator_eval",
            "network",
            "n",
            "dt",
            "eps",
            "bs",
            "integrator",
            "num_noise",
            "num_boots",
            "hidden",
            "logp_avg",
            "logp_err",
            "loss_avg",
            "loss_err",
            "part_avg",
            "part_err",
            "free_avg",
            "free_err",
            "ess_avg",
            "ess_err"
        ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_parser():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Train/eval a CNF method on certain theory"
    )
    parser.add_argument(
        "--main-dir",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Project/output root directory.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument("--run-name", type=str, default="", help="Optional label for this run.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])

    parser.add_argument("--eval-path", type=str, default="None", help="Path for model to be evaluated.")
    parser.add_argument("--num-bootstrap", type=int, default=2000, help="Number of bootstrap resample for evaluation.")
    parser.add_argument("--dt-eval", type=float, default=0.1, help="Integration time step of the evaluated model.")
    parser.add_argument("--eps-eval", type=float, default=0.25, help="Noise strength epsilon for Unbiased CNF of the evaluated model.")
    parser.add_argument("--num-noise-eval", type=int, default=10, help="Number of noise estimator for hutchinson/fp of the evaluated model.")
    parser.add_argument("--timestamp-eval", type=int, default=None, help="Timestamp to differentiate evaluated model.")
    parser.add_argument("--integrator-eval", type=str, default=None, help="Integrator of the evaluated model.")

    parser.add_argument("--n", type=int, default=10, help="Number of dimension for GMM.")

    parser.add_argument("--network", 
                        type=str, 
                        default="mlpgmm",
                        choices=["mlpgmm"])
    
    parser.add_argument("--mlpgmm-hidden",
                        type=int,
                        nargs="+",
                        default=[64, 64],
                        help="Hidden sizes for mlpgmm.",
    )
    
    parser.add_argument("--integrator", 
                        type=str, 
                        default="unbiasv2", 
                        choices=["unbiasv1", "unbiasv2", "hutch", "exact", "fp"])
    parser.add_argument("--bs", type=int, default=256, help="Batch size.")
    parser.add_argument("--dt", type=float, default=0.1, help="Integration time step.")
    parser.add_argument("--eps", type=float, default=0.25, help="Noise strength epsilon for Unbiased CNF.")
    parser.add_argument("--num-noise", type=int, default=10, help="Number of noise estimator for hutchinson/fp.")
    parser.add_argument("--num-checkpoint", type=int, default=20, help="Number of checkpoints for the gradient.")

    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate at the beginning.")
    parser.add_argument("--lr-min", type=float, default=5e-4, help="Learning rate at the end of cosine anneal scheduler.")
    parser.add_argument("--t-max-scheduler", type=int, default=400, help="Number of steps to anneal from initial to minimal lr.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps.")

    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cnf-explore", help="wandb project name")

    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)