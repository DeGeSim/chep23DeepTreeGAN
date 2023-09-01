"""Main module."""
import importlib
import os
import sys
from pathlib import Path

# Add the project to the path, -> `import fgsim.x`
sys.path.append(os.path.dirname(os.path.realpath(".")))

def main():
    import fgsim.config
    (
        fgsim.config.conf,
        fgsim.config.hyperparameters,
    ) = fgsim.config.parse_arg_conf()
    from fgsim.config import conf
    from fgsim.monitoring.logger import init_logger, logger

    init_logger()
    logger.info(
        f"tag: {conf.tag} hash: {conf.hash} loader_hash: {conf.loader_hash}"
    )
    logger.info(f"Running command {conf.command}")

    if conf.command == "train":
        from fgsim.commands.training import training_procedure

        training_procedure()

    elif conf.command == "test":
        from fgsim.commands.testing import test_procedure

        test_procedure()


    else:
        raise Exception

if __name__ == "__main__":
    main()
