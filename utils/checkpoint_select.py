import argparse
import os
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None,
                        help='results directory to analysis.')
    return parser.parse_args()


def get_all_best_checkpoint(directory):
    trainer_states = json.load(open(os.path.join(directory, 'trainer_state.json')))
    eval_check_key = 'eval_runtime'
    omit_keys = ["epoch", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "step"]

    steps = []
    metrics_timeline = {}
    for record_item in trainer_states["log_history"]:
        # eval results_item
        if eval_check_key in record_item:
            steps.append(record_item["step"])
            # if key is set in metrics_timeline
            if metrics_timeline:
                for key in record_item:
                    if key not in omit_keys:
                        metrics_timeline[key].append(record_item[key])
            # the key is not set yet, metrics timeline is empty
            else:
                for key in record_item:
                    if key not in omit_keys:
                        metrics_timeline[key] = [record_item[key]]
    
    checkpoint_results = {}
    for key in metrics_timeline:    
        if 'loss' in key:
            best_idx = int(np.argmin(metrics_timeline[key]))
        else:
            best_idx = int(np.argmax(metrics_timeline[key]))
        
        stop_iter_num = len(steps) - 1 - best_idx
        best_step = steps[best_idx]
        best_ckpt_dir = os.path.join(directory, f'checkpoint-{best_step}')
        checkpoint_results[key] = {
            "best_step": best_step,
            "best_iter": best_idx,
            "stop_iter_after_best": stop_iter_num,
            "best_checkpoint": best_ckpt_dir
        }

    json.dump(checkpoint_results, open(os.path.join(directory, "best_checkpoints.json"), 'w'), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_args()
    get_all_best_checkpoint(args.dir)
