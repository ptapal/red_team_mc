import argparse
import warnings


from helpers import load_sessions
from config import DEFAULT_N_ITER, DEFAULT_C_UCB, DEFAULT_OUTPUT_DIR, SEED
from attacks import run_redteam

warnings.filterwarnings("ignore")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Within-session EEG red-team via UCB / MCTS")
    parser.add_argument("--n_iter",     type=int,   default=DEFAULT_N_ITER,
                        help="UCB iterations per day (default: 1000)")
    parser.add_argument("--c_ucb",      type=float, default=DEFAULT_C_UCB,
                        help="UCB exploration constant (default: sqrt(2))")
    parser.add_argument("--output_dir", type=str,   default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed",       type=int,   default=SEED)
    args = parser.parse_args()

    participants = ["S" + str(i) for i in range(1, 16)]
    days         = ["Day_" + str(i) for i in [1, 7, 80, 200]]

    print("Loading sessions …")
    sessions = load_sessions(participants, days)
    print(f"Found {len(sessions)} valid participant x day sessions.")

    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx          = {p: i for i, p in enumerate(all_participants)}
    print(f"Subjects ({len(all_participants)}): {all_participants}")

    run_redteam(
        sessions   = sessions,
        cls2idx    = cls2idx,
        n_iter     = args.n_iter,
        c_ucb      = args.c_ucb,
        output_dir = args.output_dir,
        seed       = args.seed,
    )