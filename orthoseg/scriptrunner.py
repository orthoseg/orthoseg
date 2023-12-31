"""
Run the scripts in a directory.
"""
import argparse
import configparser
from pathlib import Path
import subprocess
import time

from orthoseg.util import log_util
from orthoseg.util import config_util

runner_config = None


def main():
    # Interprete arguments
    parser = argparse.ArgumentParser(add_help=False)

    # Optional arguments
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "-d", "--script_dir", help="Directory containing the scripts to run."
    )
    optional.add_argument(
        "-w",
        "--watch",
        action="store_true",
        default=False,
        help="Watch the directory forever for files getting in it.",
    )
    help = "Path to a config file with parameters that need to overrule the defaults."
    optional.add_argument(
        "-c",
        "--config",
        help=help,
    )

    # Add back help
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    args = parser.parse_args()

    # Init stuff
    script_dir = Path(args.script_dir)
    if not script_dir.exists():
        raise Exception(f"script dir {script_dir} does not exist")

    # Load the scriptrunner config
    conf = load_scriptrunner_config(args.config, script_dir)

    # Init logging
    log_util.clean_log_dir(
        log_dir=conf["dirs"].getpath("log_dir"),
        nb_logfiles_tokeep=conf["logging"].getint("nb_logfiles_tokeep"),
    )
    logger = log_util.init_logging_dictConfig(
        logconfig_dict=conf["logging"].getdict("logconfig"),
        log_basedir=conf["dirs"].getpath("script_dir"),
        loggername=__name__,
    )

    # Init working dirs
    done_dir = conf["dirs"].getpath("done_dir")
    error_dir = conf["dirs"].getpath("error_dir")

    # Loop over scripts to be ran
    wait_message_printed = False
    while True:
        # List the scripts in the dir
        script_paths = []
        script_patterns = conf["general"].getlist("script_patterns")
        for script_pattern in script_patterns:
            script_paths.extend(list(script_dir.glob(script_pattern)))

        # If no scripts found, sleep or stop...
        if len(script_paths) == 0:
            if args.watch is False:
                logger.info(f"No scripts found (anymore) in {script_dir}, so stop")
                break
            else:
                if wait_message_printed is False:
                    logger.info(
                        f"No scripts to run in {script_dir}, so watch script dir..."
                    )
                    wait_message_printed = True
                time.sleep(10)
                continue

        # Get next script alphabetically
        script_path = sorted(script_paths)[0]

        try:
            # Run the script and print output in realtime
            wait_message_printed = False
            logger.info(f"Run script {script_path}")
            cmd = [script_path]

            process = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
            # ,creationflags=subprocess.CREATE_NO_WINDOW)

            still_running = True
            while still_running:
                if process.stdout is not None:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        still_running = False
                    if output:
                        logger.info(output.strip())

            # If error code != 0, an error occured
            rc = process.poll()
            if rc != 0:
                # Script gave an error, so move to error dir
                logger.error(f"Script {script_path} gave error return code: {rc}")
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / script_path.name
                if error_path.exists():
                    error_path.unlink()
                script_path.rename(target=error_path)
            else:
                # Move the script to the done dir
                done_dir.mkdir(parents=True, exist_ok=True)
                done_path = done_dir / script_path.name
                if done_path.exists():
                    done_path.unlink()
                script_path.rename(target=done_path)

        except Exception:
            logger.exception(f"Error running script {script_path}")

            # If the script still exists, move it to error dir
            if script_path.exists():
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / script_path.name
                if error_path.exists():
                    error_path.unlink()
                script_path.rename(target=error_path)


def load_scriptrunner_config(
    config_path: str, script_dir: Path
) -> configparser.ConfigParser:
    # Load defaults first
    scriptrunner_py_dir = Path(__file__).resolve().parent
    config_paths = [scriptrunner_py_dir / "scriptrunner_defaults.ini"]

    # If a config path is specified, this config should overrule the defaults
    if config_path is not None:
        config_paths.append(Path(config_path))

    # Load!
    scriptrunner_config = config_util.read_config_ext(config_paths)

    # If a script_dir is specified, it overrides the config file(s)
    if script_dir is not None:
        scriptrunner_config["dirs"]["script_dir"] = script_dir.as_posix()

    return scriptrunner_config


if __name__ == "__main__":
    main()
