set -a && . ./.env && set +a
python run_experiment.py --config configs/triplet_bidir.json
