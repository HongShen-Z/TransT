from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '/home/test/zhs/datasets/GOT-10k/'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/mnt/e/datasets/SOT/LaSOT/'
    settings.network_path = '/home/test/zhs/projects/TransT/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/home/test/zhs/projects/TransT/pytracking/result_plots/'
    settings.results_path = '/home/test/zhs/projects/TransT/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/test/zhs/projects/TransT/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

