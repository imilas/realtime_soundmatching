#!/bin/bash
# Example: run 10 trials for every synth/method combo sequentially.
# Edit --trials to change trial count.

source .venv/bin/activate

python paper_experiments/run_paper.py --synth bandpass_noise --method GD           --trials 10 --budget 200
python paper_experiments/run_paper.py --synth bandpass_noise --method HillClimber  --trials 10 --budget 200
python paper_experiments/run_paper.py --synth bandpass_noise --method RandomSearch --trials 10 --budget 200
python paper_experiments/run_paper.py --synth bandpass_noise --method CMA-ES       --trials 10 --budget 200
python paper_experiments/run_paper.py --synth bandpass_noise --method BO           --trials 10 --budget 200
python paper_experiments/run_paper.py --synth bandpass_noise --method QL           --trials 10 --budget 200

python paper_experiments/run_paper.py --synth am_noise --method GD           --trials 10 --budget 200
python paper_experiments/run_paper.py --synth am_noise --method HillClimber  --trials 10 --budget 200
python paper_experiments/run_paper.py --synth am_noise --method RandomSearch --trials 10 --budget 200
python paper_experiments/run_paper.py --synth am_noise --method CMA-ES       --trials 10 --budget 200
python paper_experiments/run_paper.py --synth am_noise --method BO           --trials 10 --budget 200
python paper_experiments/run_paper.py --synth am_noise --method QL           --trials 10 --budget 200

python paper_experiments/run_paper.py --synth add_sinesaw --method GD           --trials 10 --budget 200
python paper_experiments/run_paper.py --synth add_sinesaw --method HillClimber  --trials 10 --budget 200
python paper_experiments/run_paper.py --synth add_sinesaw --method RandomSearch --trials 10 --budget 200
python paper_experiments/run_paper.py --synth add_sinesaw --method CMA-ES       --trials 10 --budget 200
python paper_experiments/run_paper.py --synth add_sinesaw --method BO           --trials 10 --budget 200
python paper_experiments/run_paper.py --synth add_sinesaw --method QL           --trials 10 --budget 200
