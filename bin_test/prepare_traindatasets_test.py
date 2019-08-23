

def main():
    
    None
    '''
    # WMS server we can use to get the image data
    WMS_SERVER_URL = "http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?"
    WMS_SERVER_LAYER = "ofw"
    
    # General initialisations
    segment_subject = "horsetracks"  # "greenhouses"
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"
    project_dir = os.path.join(base_dir, segment_subject)
    train_dir = os.path.join(project_dir, "training")
    input_labels_dir = os.path.join(project_dir, "input_labels")
    force = False
    
    # Initialisation of the logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)
    # Prepare train dataset
    input_labels_filename = f"{segment_subject}_trainlabels.shp"
    input_labels_filepath = os.path.join(input_labels_dir, 
                                         input_labels_filename)
    dataset_basedir = os.path.join(train_dir, "train")
    prepare_traindatasets(
            input_vector_label_filepath=input_labels_filepath,
            wms_server_url=WMS_SERVER_URL,
            wms_server_layer=WMS_SERVER_LAYER,
            output_basedir=dataset_basedir,
            force=force)

    # Prepare validation dataset
    input_labels_filename = f"{segment_subject}_validationlabels.shp"
    input_labels_filepath = os.path.join(input_labels_dir, 
                                         input_labels_filename)
    dataset_basedir = os.path.join(train_dir, "validation")
    prepare_traindatasets(
            input_vector_label_filepath=input_labels_filepath,
            wms_server_url=WMS_SERVER_URL,
            wms_server_layer=WMS_SERVER_LAYER,
            output_basedir=dataset_basedir,
            force=force)

    # Prepare test dataset
    input_labels_filename = f"{segment_subject}_testlabels.shp"
    input_labels_filepath = os.path.join(input_labels_dir, 
                                         input_labels_filename)
    dataset_basedir = os.path.join(train_dir, "test")
    prepare_traindatasets(
            input_vector_label_filepath=input_labels_filepath,
            wms_server_url=WMS_SERVER_URL,
            wms_server_layer=WMS_SERVER_LAYER,
            output_basedir=dataset_basedir,
            force=force)
    '''
    
if __name__ == "__main__":
    main()