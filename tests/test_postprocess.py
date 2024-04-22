import pytest

# import orthoseg
# from orthoseg.helpers import config_helper as conf
# import geofileops as gfo
import os


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt",
    reason="crashes on github CI on windows",
)
@pytest.mark.order(after="test_4_predict")
@pytest.mark.parametrize("keep_original_file", [False, True])
@pytest.mark.parametrize("keep_intermediary_files", [False, True])
def test_5_postprocess(
    keep_original_file: str,
    keep_intermediary_files: str,
):
    pass
    # # Load project config to init some vars.
    # config_path = footballfields_dir / "footballfields_BEFL-2019_test.ini"
    # overrules = []
    # overrules.append(f"postprocess.keep_original_file={keep_original_file}")
    # overrules.append(f"postprocess.keep_intermediary_files={keep_intermediary_files}")
    # conf.read_orthoseg_config(config_path, overrules=overrules)

    # # # Cleanup result if it isn't empty yet
    # result_vector_dir = conf.dirs.getpath("output_vector_dir")
    # result_path = result_vector_dir / "footballfields_01_201_BEFL-2019.gpkg"
    # result_diss_path = (
    #     result_vector_dir / "footballfields_01_201_BEFL-2019_dissolve.gpkg"
    # )
    # # if result_diss_path.exists():
    # #     gfo.remove(result_diss_path)

    # # Run task to postprocess
    # orthoseg.postprocess(config_path=config_path, config_overrules=overrules)

    # # Check results
    # if not keep_original_file and not keep_intermediary_files:
    #     assert result_path.exists()
    #     result_gdf = gfo.read_file(result_path)
    #     if os.name == "nt":
    #         assert len(result_gdf) == 204
    #     else:
    #         assert len(result_gdf) == 204
