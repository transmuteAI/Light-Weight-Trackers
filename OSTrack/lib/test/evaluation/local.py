from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/workspace/tracking_datasets/ostrack/got10k_lmdb'
    settings.got10k_path = '/workspace/tracking_datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/workspace/tracking_datasets/ostrack/itb'
    settings.lasot_extension_subset_path_path = '/workspace/tracking_datasets/ostrack/lasot_extension_subset'
    settings.lasot_lmdb_path = '/workspace/tracking_datasets/ostrack/lasot_lmdb'
    settings.lasot_path = '/workspace/tracking_datasets/lasot/LaSOTBenchmark/'
    settings.network_path = '/workspace/tracking_datasets/ostrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/workspace/tracking_datasets/ostrack/nfs'
    settings.otb_path = '/workspace/tracking_datasets/otb/'
    settings.prj_dir = '/workspace/OSTrack'
    settings.result_plot_path = '/workspace/tracking_datasets/ostrack/test/result_plots'
    settings.results_path = '/workspace/tracking_datasets/ostrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/workspace/tracking_datasets/ostrack'
    settings.segmentation_path = '/workspace/tracking_datasets/ostrack/test/segmentation_results'
    settings.tc128_path = '/workspace/tracking_datasets/ostrack/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/workspace/tracking_datasets/ostrack/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/workspace/tracking_datasets/trackingnet'
    settings.uav_path = '/workspace/tracking_datasets/ostrack/uav'
    settings.vot18_path = '/workspace/tracking_datasets/ostrack/vot2018'
    settings.vot22_path = '/workspace/tracking_datasets/ostrack/vot2022'
    settings.vot_path = '/workspace/tracking_datasets/ostrack/VOT2019'
    settings.youtubevos_dir = ''

    return settings

