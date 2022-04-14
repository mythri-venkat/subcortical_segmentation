import torch

from unetseg3d.datasets.utils import get_class
from unetseg3d.unet3d.config import load_config
from unetseg3d.unet3d.utils import get_logger
import optuna
from optuna.storages import RetryFailedTrialCallback
import os
import sys

logger = get_logger('TrainingSetup')


def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # create trainer
    default_trainer_builder_class = 'UNet3DTrainerBuilder'
    trainer_builder_class = config['trainer'].get('builder', default_trainer_builder_class)+'Opt'
    trainer_builder = get_class(trainer_builder_class, modules=['unetseg3d.unet3d.trainer'])
    
    trainer = trainer_builder(config)
    storage = optuna.storages.RDBStorage(
        "sqlite:///example.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=2),
    )
    dirname = os.path.dirname(sys.argv[2]).split("/")[-1]
    split_name = config['loaders']['train']['file_paths'][0].split("split_")[-1].replace(".pkl","")
    study = optuna.create_study(
        storage=storage, study_name="roionly_"+dirname+'_'+split_name, direction="maximize", load_if_exists=True
    )
    study.optimize(trainer.build, n_trials=None, timeout=86400)

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # The line of the resumed trial's intermediate values begins with the restarted epoch.
    optuna.visualization.plot_intermediate_values(study).show()

if __name__ == '__main__':
    main()
