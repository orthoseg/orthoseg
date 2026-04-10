import logging
import os
from pathlib import Path

from orthoseg.lib import cleanup

from orthoseg.helpers import config_helper as conf
from orthoseg.util import log_util

logger = logging.getLogger(__name__)


def run_clean_project_dir(projects_dir: Path, simulate: bool = False):
    # Exclude directories where name starts with '_'
    projects = [
        subdir for subdir in os.listdir(projects_dir)
        if os.path.isdir(projects_dir/subdir) and not subdir.startswith("_")
    ]
    for project in projects:
        config_path = projects_dir / project / f"{project}.ini"
        if config_path.exists():
            conf.read_orthoseg_config(config_path=config_path)
            global logger
            logger = log_util.main_log_init(
                log_dir=conf.dirs.getpath("log_dir"),
                log_basefilename=run_clean_project_dir.__name__
            )
            cleanup.clean_project_dir(
                model_dir=conf.dirs.getpath("model_dir"),
                model_versions_to_retain=conf.cleanup.getint("model_versions_to_retain"),
                training_dir=conf.dirs.getpath("training_dir"),
                training_versions_to_retain=conf.cleanup.getint(
                    "training_versions_to_retain"
                ),
                output_vector_dir=conf.dirs.getpath("output_vector_dir"),
                prediction_versions_to_retain=conf.cleanup.getint(
                    "prediction_versions_to_retain"
                ),
                simulate=simulate,
            )
        else:
            # logger.info(f"Config_path ({config_path}) doesn't exist.")
            print(f"Config_path ({config_path}) doesn't exist.")


# If the script is ran directly...
if __name__ == "__main__":
    run_clean_project_dir(projects_dir=Path(r"X:\Monitoring\OrthoSeg"), simulate=True)
