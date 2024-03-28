import logging
import os
from pathlib import Path

from orthoseg.lib import cleanup

from orthoseg.helpers import config_helper as conf

logger = logging.getLogger(__name__)

# TODO: deze geeft een probleem bij get_aidetectioninfo
# X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\DK-2019\prc_DK2019.gpkg

def get_projects(projects_dir: Path):
    # Remove directories where name starts with '_'
    projects = [
        subdir for subdir in os.listdir(projects_dir)
        if os.path.isdir(projects_dir/subdir) and not subdir.startswith("_")
    ]
    # Remove certain directories
    projects_to_remove = {"bin", "project_template", "download"}
    projects_to_clean = [ele for ele in projects if ele not in projects_to_remove]
    
    return projects_to_clean


def cleanup_old_data(projects_dir: Path, versions_to_retain: dict[str, int], simulate: bool = False):
    projects = get_projects(projects_dir=projects_dir)
    for project in projects:
        config_path = projects_dir / project / f"{project}.ini"
        conf.read_orthoseg_config(config_path)
        if config_path.exists():
            subdirs = [
                subdir for subdir in os.listdir(projects_dir/project)
                if os.path.isdir(projects_dir/project/subdir)
            ]
            for subdir in subdirs:
                match subdir:
                    case 'models':
                        cleanup.clean_models(
                            config_path=config_path,
                            path=conf.dirs.getpath("model_dir"),
                            versions_to_retain=versions_to_retain['models'],
                            simulate=simulate,
                        )
                    case 'output_vector':
                        cleanup.clean_training_data_directories(
                            config_path=config_path,
                            path=conf.dirs.getpath("training_dir"),
                            versions_to_retain=versions_to_retain['training_dirs'],
                            simulate=simulate,
                        )
                    case 'training':
                        cleanup.clean_predictions(
                            config_path=config_path,
                            path=conf.dirs.getpath("output_vector_dir"),
                            versions_to_retain=versions_to_retain['predictions'],
                            simulate=simulate,
                        )
        else:
            logger.info(f"Config_path ({config_path}) doesn't exist.")
            print(f"Config_path ({config_path}) doesn't exist.")
                
        

# If the script is ran directly...
if __name__ == "__main__":
    # cleanup()
    projects_dir = Path(r"X:\Monitoring\OrthoSeg")
    versions_to_retain = {
        "models": 3,
        "training_dirs": 3,
        "predictions": 3
    }
    cleanup_old_data(projects_dir=projects_dir, versions_to_retain=versions_to_retain, simulate=True)
