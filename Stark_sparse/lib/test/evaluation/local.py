from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/workspace/tracking_datasets/got10k_lmdb'
    settings.got10k_path = '/workspace/tracking_datasets/got10k/'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/workspace/tracking_datasets/lasot_lmdb'
    settings.lasot_path = '/workspace/tracking_datasets/lasot/LaSOTBenchmark'
    settings.network_path = '/workspace/tracking_datasets/stark/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/workspace/tracking_datasets/nfs'
    settings.otb_path = '/workspace/tracking_datasets/OTB2015'
    settings.prj_dir = '/workspace/stark'
    settings.result_plot_path = '/workspace/tracking_datasets/stark/test/result_plots'
    settings.results_path = '/workspace/tracking_datasets/stark/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/workspace/tracking_datasets/stark'
    settings.segmentation_path = '/workspace/tracking_datasets/stark/test/segmentation_results'
    settings.tc128_path = '/workspace/tracking_datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/workspace/tracking_datasets/trackingnet'
    settings.uav_path = '/workspace/tracking_datasets/UAV123'
    settings.vot_path = '/workspace/tracking_datasets/VOT2019'
    settings.youtubevos_dir = ''

    return settings

