from python_code.plotters.plotter_config import get_config, PlotType
from python_code.plotters.plotter_methods import compute_for_method, RunParams
from python_code.plotters.plotter_utils import plot_by_values, plot_by_values2

if __name__ == '__main__':
    run_over = False  # whether to run over previous results
    trial_num = 1  # number of trials per point estimate, used to reduce noise by averaging results of multiple runs
    run_params_obj = RunParams(run_over=run_over,
                               trial_num=trial_num)
    label_name = PlotType.BY_BLOCK # Always_VS_Random_By_SNR, Always_VS_Random_By_Block

    print(label_name.name)
    params_dicts, methods_list, values, xlabel, ylabel, plot_type = get_config(label_name.name)
    all_curves = []

    for method in methods_list:
        print(method)
        for params_dict in params_dicts:
            print(params_dict)
            compute_for_method(all_curves, method, params_dict, run_params_obj)

    if plot_type == 'plot_by_blocks' or plot_type == 'plot_by_snrs':
        plot_by_values(all_curves, values, xlabel, ylabel, plot_type)
    else:
        plot_by_values2(all_curves, values, xlabel, ylabel, plot_type)
