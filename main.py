import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import panel as pn
import param
import pickle as pkl
import shelve
import bisect

from bokeh.models import Span, Label, Title, Range1d
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure, show
import datetime as dtt
from  datetime import datetime

from legendmeta import LegendMetadata
from legendmeta.catalog import Props

path = "/data1/users/marshall/prod-ref/v06.00/"
pn.extension()

def gen_run_dict(path):
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    
    par_file = os.path.join(prod_config["paths"]["par"], 'key_resolve.jsonl')
    run_dict = {}
    with open(par_file, 'r') as file:
        for json_str in file:
            run_dict[json.loads(json_str)["apply"][0].split("-")[2]] = {"experiment":json.loads(json_str)["apply"][0].split("-")[0],
                                                                        "period":json.loads(json_str)["apply"][0].split("-")[1],
                                                                        "timestamp":json.loads(json_str)["valid_from"]}
    out_dict={}
    for run in run_dict:
        if os.path.isfile(os.path.join(prod_config["paths"]["par_hit"], f"cal/p02/{run}/{run_dict[run]['experiment']}-{run_dict[run]['period']}-{run}-cal-{run_dict[run]['timestamp']}-par_hit.json")):
            out_dict[run] = run_dict[run]    
    return out_dict


def sorter(path, timestamp, key="String"):
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    
    cfg_file = prod_config["paths"]["chan_map"]

    configs = LegendMetadata(path = cfg_file)
    
    chmap = configs.channelmaps.on(timestamp).map("daq.fcid")
    
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    software_config_path = prod_config["paths"]["config"]
    software_config_db = LegendMetadata(path = software_config_path)
    software_config = software_config_db.on(timestamp, system="cal").hardware_configuration.channel_map
    
    def sort_string(chan):
        string = chmap[chan]['location']['string']
        position =  chmap[chan]['location']['position']
        return int(f'{string:02}{position:02}')

    def sort_cc4(chan):
        cc4 = chmap[chan]['electronics']['cc4']["id"]
        channel =  chmap[chan]['electronics']['cc4']["channel"]
        return int(f'{ord(cc4[0]):02}{cc4[1]}{channel:02}')

    def sort_hv(chan):
        hv = chmap[chan]['voltage']['card']["id"]
        channel =  chmap[chan]['voltage']["channel"]
        return int(f'{hv:02}{channel:02}')

    def sort_daq(chan):
        crate = chmap[chan]['daq']['crate']
        card = chmap[chan]['daq']['card']['id']
        channel =  chmap[chan]['daq']["channel"]
        return int(f'{crate:02}{card:02}{channel:02}')
    
    if key == "String":
        channels = sorted(channels, key = sort_string)
    elif key == "CC4":
        channels = sorted(channels, key = sort_cc4)
    elif key == "HV":
        channels = sorted(channels, key = sort_hv)
    elif key == "DAQ":
        channels = sorted(channels, key = sort_daq)

    out_dict={}
    for channel in channels:
        detector = chmap[channel]["name"]
        if key == "String":
            curr_entry = f"String:{int(chmap[channel]['location']['string']):02}"
        elif key == "CC4":
            curr_entry = f"CC4:{chmap[channel]['electronics']['cc4']['id']}"
        elif key == "HV":
            curr_entry = f"HV:{chmap[channel]['voltage']['card']['id']:02}"
        elif key == "DAQ":
            curr_entry = f"DAQ:Cr:{chmap[channel]['daq']['crate']:02},Ch:{chmap[channel]['daq']['card']['id']:02}"
        if curr_entry in out_dict:
            out_dict[curr_entry].append(channel)
        else:
            out_dict[curr_entry]=[channel]
    return out_dict, software_config, chmap

class detailed_cal_params(param.Parameterized):
    
    path = "/data1/users/marshall/prod-ref/v06.00"
    
    run_dict = gen_run_dict(path)
    
    all_runs = list(run_dict)

    run = param.Selector(objects = all_runs, default = all_runs[-1])
    
    prod_config = os.path.join(path, "config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    
    
    file = os.path.join(prod_config["paths"]["plt"],
                        f'hit/cal/{run_dict[all_runs[-1]]["period"]}/{all_runs[-1]}/{run_dict[all_runs[-1]]["experiment"]}-{run_dict[all_runs[-1]]["period"]}-{all_runs[-1]}-cal-{run_dict[all_runs[-1]]["timestamp"]}-plt_hit')
    
    with shelve.open(file, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
        channels = list(shelf.keys())
        
    strings_dict, chan_dict, channel_map = sorter(path,
                                                    run_dict[all_runs[-1]]["timestamp"],
                                                    "String")
    channel_list = []
    for channel in channels:
        channel_list.append(f"{channel}: {channel_map[int(channel[2:])]['name']}")
    channel = param.Selector(objects = channel_list)
    
    cal_plots = ['cal_stability',
                 'peak_fits',
                 'cal_fit',
                 'fwhm_fit',
                 'spectrum_plot',
                 'survival_frac',
                 "spectrum",
                 "logged_spectrum"]

    
    aoe_plots = ['dt_deps',
                 'compt_bands_nocorr',
                 'band_fits',
                 'mean_fit',
                 'sigma_fit',
                 'compt_bands_corr',
                 'surv_fracs',
                 'PSD_spectrum',
                 'psd_sf']
    
    tau_plots =["slope", "waveforms"]
    
    optimisation_plots = ["trap_kernel", "zac_kernel", "cusp_kernel", "trap_acq", "zac_acq", "cusp_acq"]
    
    _options = {'cuspEmax_ctc': cal_plots , 'zacEmax_ctc': cal_plots,
          'trapEmax_ctc': cal_plots , 'trapTmax': cal_plots,
          "A/E": aoe_plots, "Tau": tau_plots, "Optimisation": optimisation_plots}
    
    parameter = param.ObjectSelector(default = "cuspEmax_ctc", objects = list(_options))
    
    plot_type = param.ObjectSelector(default = "cal_stability", objects= cal_plots)
    
    @param.depends('run', watch=True)
    def update_plot_dict(self):
        self.file = os.path.join(self.prod_config["paths"]["plt"],
                              f'hit/cal/{self.run_dict[self.run]["period"]}/{self.run}/{self.run_dict[self.run]["experiment"]}-{self.run_dict[self.run]["period"]}-{self.run}-cal-{self.run_dict[self.run]["timestamp"]}-plt_hit')  
    
        with shelve.open(self.file, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
            channels = list(shelf.keys())
            
        self.strings_dict, self.chan_dict, self.channel_map = sorter(self.path,
                                                    self.run_dict[self.run]["timestamp"],
                                                    "String")
        channel_list = []
        for channel in channels:
            channel_list.append(f"{channel}: {self.channel_map[int(channel[2:])]['name']}")
        
        self.param["channel"].objects = channel_list
        self.channel = channel_list[0]
    
    @param.depends('parameter', watch=True)
    def update_plot_type(self):
        plots = self._options[self.parameter]
        self.param["plot_type"].objects = plots
        self.plot_type = plots[0]
    
    @param.depends("run", "channel", "parameter", "plot_type")
    def view(self):
        with shelve.open(self.file, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
                plot_dict = shelf[self.channel[:5]]
        with shelve.open(self.file.replace("hit","dsp"), 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
               dsp_dict = shelf[self.channel[:5]]
        if self.parameter == "A/E":
            fig = plot_dict[self.plot_type]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
        elif self.parameter == "Tau":
            fig = dsp_dict[self.plot_type]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
        elif self.parameter == "Optimisation":
            fig = dsp_dict[f"{self.plot_type.split('_')[0]}_optimisation"][f"{self.plot_type.split('_')[1]}_space"]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
        else:
            if self.plot_type == "spectrum" or self.plot_type == "logged_spectrum":
                fig = plt.figure()
                
                plt.step((plot_dict[self.parameter]["spectrum"]["bins"][1:]+\
                          plot_dict[self.parameter]["spectrum"]["bins"][:-1])/2, 
                          plot_dict[self.parameter]["spectrum"]["counts"], 
                         where='post')
                plt.xlabel("Energy (keV)")
                plt.ylabel("Counts")
                if self.plot_type =="logged_spectrum":
                    plt.yscale("log")
            else:
                fig = plot_dict[self.parameter][self.plot_type]
                dummy = plt.figure()
                new_manager = dummy.canvas.manager
                new_manager.canvas.figure = fig
                fig.set_canvas(new_manager.canvas)

        return fig


def plot_energy_resolutions(run, run_dict, path, key="String"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.fcid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [chmap[int(field[2:])]["name"] for field in soft_dict if soft_dict[field]["software_status"]=="Off"]
    
    file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict["period"]}/{run}')
    path = os.path.join(file_path, 
                            f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_hit_results.json')
    
    with open(path, 'r') as r:
        all_res = json.load(r)
        
    default = {'cuspEmax_ctc_cal': {'Qbb_fwhm': np.nan, 
                                                  'Qbb_fwhm_err': np.nan, 
                                                  '2.6_fwhm': np.nan, 
                                                  '2.6_fwhm_err': np.nan, 
                                                  'm0': np.nan, 
                                                  'm1': np.nan}, 
                             'zacEmax_ctc_cal': {'Qbb_fwhm': np.nan, 
                                                 'Qbb_fwhm_err': np.nan, 
                                                 '2.6_fwhm': np.nan, 
                                                 '2.6_fwhm_err': np.nan, 
                                                 'm0': np.nan, 
                                                 'm1': np.nan}, 
                             'trapEmax_ctc_cal': {'Qbb_fwhm': np.nan, 
                                                  'Qbb_fwhm_err': np.nan, 
                                                  '2.6_fwhm': np.nan, 
                                                  '2.6_fwhm_err': np.nan, 
                                                  'm0': np.nan, 
                                                  'm1': np.nan}}
    res = {}
    for stri in strings:
        res[stri]=default
        for channel in strings[stri]:
            detector = channel_map[channel]["name"]
            try:
                res[detector] = all_res[f"ch{channel:03}"]["ecal"]
            except:
                res[detector] = default
    
    
            
                
    fig = plt.figure() #
    plt.errorbar(list(res), [res[det]["cuspEmax_ctc_cal"]["Qbb_fwhm"] for det in res],
                yerr=[res[det]["cuspEmax_ctc_cal"]["Qbb_fwhm_err"] for det in res], 
                 marker='o',linestyle = ' ', color='deepskyblue', 
                 label = f'Cusp Average: {np.nanmean([res[det]["cuspEmax_ctc_cal"]["Qbb_fwhm"] for det in res]):.2f}keV')
    plt.errorbar(list(res), [res[det]["zacEmax_ctc_cal"]["Qbb_fwhm"] for det in res],
                yerr=[res[det]["zacEmax_ctc_cal"]["Qbb_fwhm_err"] for det in res], 
                 marker='o',linestyle = ' ', color='green',
                label = f'Zac Average: {np.nanmean([res[det]["zacEmax_ctc_cal"]["Qbb_fwhm"] for det in res]):.2f}keV')
    plt.errorbar(list(res), [res[det]["trapEmax_ctc_cal"]["Qbb_fwhm"] for det in res],
                yerr=[res[det]["trapEmax_ctc_cal"]["Qbb_fwhm_err"] for det in res], marker='o',linestyle = ' ', color='orangered',
                label = f'Trap Average: {np.nanmean([res[det]["trapEmax_ctc_cal"]["Qbb_fwhm"] for det in res]):.2f}keV')
    for stri in strings:
        loc=np.where(np.array(list(res))==stri)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("blue")
        plt.axvline(stri, color='black')
    plt.tick_params(axis='x', labelrotation=90)
    
    for off_det in off_dets:
        loc=np.where(np.array(list(res))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")

    plt.yticks(np.arange(0,11,1))
    plt.xlabel('Detector')
    plt.ylabel('FWHM (keV)')
    plt.grid(linestyle='dashed', linewidth=0.5,which="both", axis='both')
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} Energy Resolutions")
    plt.legend(loc='upper right')
    plt.ylim([1,5])
    plt.tight_layout()
    plt.close()
    return fig

def plot_no_fitted_energy_peaks(run, run_dict, path, key="String"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"])
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.fcid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [field for field in soft_dict if soft_dict[field]["software_status"]=="Off"]
    
    file_path = os.path.join(prod_config["paths"]["par_hit"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_hit_results.json')
    
    res = {}
    with open(file_path, 'r') as r:
        res = json.load(r)

    peaks = [583.191,
        727.330,
        860.564,
        1592.53,
        1620.50,
        2103.53,
        2614.50]
    grid = np.zeros((len(peaks), len(channels)))
    for i,channel in enumerate(channels):
        idxs = []
        try:
            fitted_peaks = res[f"ch{channel:03}"]["ecal"]["cuspEmax_ctc_cal"]["fitted_peaks"]
            if not isinstance(fitted_peaks,list):
                fitted_peaks = [fitted_peaks]
            for j,peak in enumerate(peaks):
                if peak in fitted_peaks:
                    idxs.append(j)
            if len(idxs)>0:
                grid[np.array(idxs),i]=1
                
        except:
            if chmap[channel] in off_dets:
                grid[:,i]=1
            pass
        
    fig=plt.figure()
    plt.imshow(grid, cmap = "brg")
    plt.ylabel("peaks")
    plt.xlabel("channel")

    yticks, ylabels = plt.yticks()
    plt.yticks(ticks = yticks[1:-1], labels = [f"{peak:.1f}" for peak in peaks])

    plt.xticks(ticks = np.arange(0,len(channels),1), labels = [f"{chmap[channel]['name']}" for channel in channels], rotation = 90)
    for off_det in off_dets:
        loc = np.where(np.array(channels)==int(off_det[2:]))[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} Energy Fits")
    plt.tight_layout()
    plt.show()
    return fig

def plot_no_fitted_aoe_slices(run, run_dict, path, key="String"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.fcid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [chmap[int(field[2:])]["name"] for field in soft_dict if soft_dict[field]["software_status"]=="Off"]
    
    
    file_path = os.path.join(prod_config["paths"]["par_hit"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_hit_results.json')
    
    res = {}
    with open(file_path, 'r') as r:
        res = json.load(r)

    nfits = {}
    for stri in strings:
        res[stri]=np.nan
        for channel in strings[stri]:
            detector = channel_map[channel]["name"]
            try:
                nfits[detector] =res[f"ch{channel:03}"]["aoe"]["correction_fit_results"]["n_of_valid_fits"]
            except:
                nfits[detector] =np.nan
        
    fig=plt.figure()
    plt.scatter(list(nfits), [nfits[channel] for channel in nfits])
    plt.tick_params(axis='x', labelrotation=90)
    for off_det in off_dets:
        loc = np.where(np.array(list(nfits))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.xlabel('Channel')
    plt.ylabel('# of A/E fits')
    plt.grid(linestyle='dashed', linewidth=0.5,which="both", axis='both')
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} A/E fits")
    plt.tight_layout()
    plt.close()
    return fig

def get_aoe_results(run, run_dict, path, key="String"):

    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.fcid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [chmap[int(field[2:])]["name"] for field in soft_dict if soft_dict[field]["software_status"]=="Off"]
    
    file_path = os.path.join(prod_config["paths"]["par_hit"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_hit_results.json')
    
    
    with open(file_path, 'r') as r:
        all_res = json.load(r)
    
    default = {'A/E_Energy_param': 'cuspEmax', 
                                 'Cal_energy_param': 'cuspEmax_ctc', 
                                 'dt_param': 'dt_eff', 
                                 'rt_correction': False, 
                                 'Mean_pars': [np.nan, np.nan], 
                                 'Sigma_pars': [np.nan, np.nan], 
                                 'Low_cut': np.nan, 'High_cut': np.nan, 
                                 'Low_side_sfs': {
                                     '1592.5': {
                                         'sf': np.nan, 
                                         'sf_err': np.nan}, 
                                     '1620.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '2039': {'sf': np.nan, 
                                              'sf_err': np.nan}, 
                                     '2103.53': {'sf': np.nan, 
                                                 'sf_err': np.nan}, 
                                     '2614.5': {'sf': np.nan, 
                                                'sf_err': np.nan}}, 
                                 '2_side_sfs': {
                                     '1592.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '1620.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '2039': {'sf': np.nan, 
                                              'sf_err': np.nan}, 
                                     '2103.53': {'sf': np.nan, 
                                                 'sf_err': np.nan}, 
                                     '2614.5': {'sf': np.nan, 
                                                'sf_err': np.nan}}}
            
        
    aoe_res = {}
    for stri in strings:
        aoe_res[stri]=default
        for channel in strings[stri]:
            detector = channel_map[channel]["name"]

            try:  
                aoe_res[detector] =all_res[f"ch{channel:03}"]["aoe"]
            except:
                aoe_res[detector] = default

            if len(list(aoe_res[detector])) ==10:
                aoe_res[detector].update({
                                 'Low_side_sfs': {
                                     '1592.5': {
                                         'sf': np.nan, 
                                         'sf_err': np.nan}, 
                                     '1620.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '2039': {'sf': np.nan, 
                                              'sf_err': np.nan}, 
                                     '2103.53': {'sf': np.nan, 
                                                 'sf_err': np.nan}, 
                                     '2614.5': {'sf': np.nan, 
                                                'sf_err': np.nan}}, 
                                 '2_side_sfs': {
                                     '1592.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '1620.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '2039': {'sf': np.nan, 
                                              'sf_err': np.nan}, 
                                     '2103.53': {'sf': np.nan, 
                                                 'sf_err': np.nan}, 
                                     '2614.5': {'sf': np.nan, 
                                                'sf_err': np.nan}}})  

            elif len(list(aoe_res[detector])) <10:
                aoe_res[detector] = default
                
    fig = plt.figure()
    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["1592.5"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["1592.5"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ', 
                 label = 'Tl DEP')

    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["1620.5"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["1620.5"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ',  
                 label = f'Bi FEP')
    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["2039"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["2039"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ',  
                 label = r'CC @ $Q_{\beta \beta}$')
    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["2103.53"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["2103.53"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ',  
                 label = f'Tl SEP')
    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["2614.5"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["2614.5"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ',  
                 label = f'Tl FEP')

    for stri in strings:
        loc=np.where(np.array(list(aoe_res))==stri)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("blue")
        plt.axvline(stri, color='black')
    plt.tick_params(axis='x', labelrotation=90)
    for off_det in off_dets:
        loc = np.where(np.array(list(aoe_res))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.yticks(np.arange(0,110,10))
    plt.xlabel('Detector')
    plt.ylabel('Survival fraction')
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} A/E Survival Fractions")
    plt.legend(loc='upper right')
    plt.ylim([0,100])
    plt.tight_layout()
    #plt.savefig("/data1/users/marshall/prod-ref/optim_test/aoe.png")
    plt.close()

    
    return fig


def plot_pz_consts(run, run_dict, path, key="String"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.fcid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [chmap[int(field[2:])]["name"] for field in soft_dict if soft_dict[field]["software_status"]=="Off"]
    
    cal_dict_path = os.path.join(prod_config["paths"]["par_dsp"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_dsp.json')
    
    

    with open(cal_dict_path,'r') as r:
        cal_dict = json.load(r)
    
    taus={}

    for stri in strings:
        taus[stri]=np.nan
        for channel in strings[stri]:
            det = channel_map[channel]["name"]
            try:
                taus[det] = float(cal_dict[f"ch{channel:03}"]["pz"]["tau"][:-3])/1000
            except:
                taus[det] =np.nan
    
    fig = plt.figure()
    plt.errorbar(list(taus),[taus[det] for det in taus] ,yerr=10,
                 marker='o', color='deepskyblue', linestyle = '')
    for stri in strings:
        loc=np.where(np.array(list(taus))==stri)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("blue")
        plt.axvline(stri, color='black')
    plt.tick_params(axis='x', labelrotation=90)
    for off_det in off_dets:
        loc = np.where(np.array(list(taus))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.xlabel('Detector')
    plt.ylabel(f'Pz constant ($\mu s$)')
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} Pole Zero Constants")
    plt.tight_layout()
    plt.close()
    return fig

def plot_alpha(run, run_dict, path, key="String"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.fcid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [chmap[int(field[2:])]["name"] for field in soft_dict if soft_dict[field]["software_status"]=="Off"]
    
    
    cal_dict_path = os.path.join(prod_config["paths"]["par_dsp"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_dsp.json')
    
    with open(cal_dict_path,'r') as r:
        cal_dict = json.load(r)
    
    trap_alpha={}
    cusp_alpha={}
    zac_alpha={}

    
    for stri in strings:
        trap_alpha[stri]=np.nan
        cusp_alpha[stri]=np.nan
        zac_alpha[stri]=np.nan
        for channel in strings[stri]:
            det = channel_map[channel]["name"]
            try:
                trap_alpha[det]=(float(cal_dict[f"ch{channel:03}"]["ctc_params"]["trapEmax_ctc"]["parameters"]["a"]))
                cusp_alpha[det]=(float(cal_dict[f"ch{channel:03}"]["ctc_params"]["cuspEmax_ctc"]["parameters"]["a"]))
                zac_alpha[det]=(float(cal_dict[f"ch{channel:03}"]["ctc_params"]["zacEmax_ctc"]["parameters"]["a"]))
            except:
                trap_alpha[det]=np.nan
                cusp_alpha[det]=np.nan
                zac_alpha[det]=np.nan

    fig = plt.figure()
    plt.scatter(list(trap_alpha), [trap_alpha[det] for det in trap_alpha],
                 marker='o', color='deepskyblue', label='Trap')
    plt.scatter(list(cusp_alpha), [cusp_alpha[det] for det in cusp_alpha],
                 marker='o', color='orangered', label='Cusp')
    plt.scatter(list(zac_alpha), [zac_alpha[det] for det in zac_alpha],
                 marker='o', color='green', label='Zac')
    for stri in strings:
        loc=np.where(np.array(list(trap_alpha))==stri)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("blue")
        plt.axvline(stri, color='black')
    plt.tick_params(axis='x', labelrotation=90)
    for off_det in off_dets:
        loc = np.where(np.array(list(trap_alpha))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.xlabel('Detector')
    plt.ylabel(f'Alpha Value (1/ns)')
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} Charge Trapping Constants")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.close()
    return fig

def plot_bls(plot_dict,chan_dict, channels, 
             string, key="String"):

    p = figure(width=700, height=600, y_axis_type="log")
    p.title.text = string
    p.title.align = "center"
    p.title.text_font_size = "15px"
    colours = Category20[len(channels)]
    with shelve.open(plot_dict, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
        for i,channel in enumerate(channels):
            try:

                plot_dict_chan = shelf[f"ch{channel:03}"]

                p.step(plot_dict_chan["baseline"]["bins"], 
                         plot_dict_chan["baseline"]["bl_array"],
                           legend_label=f'ch{channel:03}: {chan_dict[channel]["name"]}', 
                          mode="after", line_width=2, line_color = colours[i])
            except:
                pass

            plot_dict_chan=None
        
    p.add_layout(Title(text="bl\_mean", align="center"), "below")
    p.add_layout(Title(text="Counts", align="center"), "left")
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    return p
    
def plot_fep_stability_channels2d(plot_dict, chan_dict, channels, yrange, string, 
                                  key="String", energy_param = "cuspEmax_ctc"):
    
    times = None
    p = figure(width=700, height=600, y_axis_type="log", x_axis_type='datetime')
    p.title.text = string
    p.title.align = "center"
    p.title.text_font_size = "15px"
    colours = Category20[len(channels)]
    with shelve.open(plot_dict, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
        for i,channel in enumerate(channels):
            try:

                plot_dict_chan = shelf[f"ch{channel:03}"]
                p.line([datetime.fromtimestamp(time) for time in plot_dict_chan[energy_param]["mean_stability"]["time"]], 
                         plot_dict_chan[energy_param]["mean_stability"]["energy"], 
                         legend_label=f'ch{channel:03}: {chan_dict[channel]["name"]}', 
                          line_width=2, line_color = colours[i])
                if times is None:
                    times = [datetime.fromtimestamp(t) for t in plot_dict_chan[energy_param]["mean_stability"]["time"]]      
            except:
                pass

    p.y_range = Range1d(yrange[0], yrange[1])
    p.add_layout(Title(text=f"Time (UTC), starting: {times[0].strftime('%d/%m/%Y %H:%M:%S')}", align="center"), "below")
    p.add_layout(Title(text="Energy (keV)", align="center"), "left")
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    return p

def plot_energy_spectra(plot_dict, chan_dict, channels, string,  
                        key="String", energy_param = "cuspEmax_ctc"):
    
    p = figure(width=700, height=600, y_axis_type="log")
    p.title.text = string
    p.title.align = "center"
    p.title.text_font_size = "15px"
    colours = Category20[len(channels)]
    
    with shelve.open(plot_dict, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
        for i,channel in enumerate(channels):
            try:

                plot_dict_chan = shelf[f"ch{channel:03}"]
                p.step((plot_dict_chan[energy_param]["spectrum"]["bins"][1:]+\
                          plot_dict_chan[energy_param]["spectrum"]["bins"][:-1])/2, 
                         plot_dict_chan[energy_param]["spectrum"]["counts"], 
                         legend_label=f'ch{channel:03}: {chan_dict[channel]["name"]}', 
                          mode="after", line_width=2, line_color = colours[i])
            except:
                pass
    
    p.add_layout(Title(text=f"Energy (keV)", align="center"), "below")
    p.add_layout(Title(text="Counts", align="center"), "left")
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    
    return p
    
class cal_summary(param.Parameterized):
    
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (16, 6)
    plt.rcParams['figure.dpi'] = 100
    
    path = "/data1/users/marshall/prod-ref/v06.00"
    cached_plots ={}
    
    run_dict = gen_run_dict(path)
    
    all_runs = list(run_dict)
    
    run = param.Selector(objects = all_runs, default = all_runs[-1])
    
    prod_config = os.path.join(path, "config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    
    plot_dict = os.path.join(prod_config["paths"]["plt"],
                        f'hit/cal/{run_dict[all_runs[-1]]["period"]}/{all_runs[-1]}/{run_dict[all_runs[-1]]["experiment"]}-{run_dict[all_runs[-1]]["period"]}-{all_runs[-1]}-cal-{run_dict[all_runs[-1]]["timestamp"]}-plt_hit')
    
    
    plot_types = ["Energy", "A/E", "Tau", "Alpha", "Baseline", 
                  "Stability", "Spectra", "Valid_Energy_fits", "Valid_A/E_fits"]
    plot_type = param.ObjectSelector(default = "Energy", objects= plot_types)
    
    plot_types_dict = {"Energy": plot_energy_resolutions, "A/E":get_aoe_results, 
                       "Tau":plot_pz_consts, "Alpha": plot_alpha, 
                       "Valid_Energy_fits": plot_no_fitted_energy_peaks, 
                       "Valid_A/E_fits": plot_no_fitted_aoe_slices,
                      "Baseline": plot_bls, "Spectra": plot_energy_spectra}
    
    sort_by_options = ["String", "CC4", "DAQ", "HV"]
    sort_by = param.ObjectSelector(default = "String", objects= sort_by_options)
        
    strings_dict, chan_dict, channel_map = sorter(path,
                                                    run_dict[all_runs[-1]]["timestamp"],
                                                    sort_by_options[0])
    string = param.ObjectSelector(default = list(strings_dict)[0], objects = list(strings_dict))
    
    @param.depends('run', watch=True)
    def update_plot_dict(self):
        self.cached_plots=None
        self.plot_dict = os.path.join(self.prod_config["paths"]["plt"],
                        f'hit/cal/{self.run_dict[self.run]["period"]}/{self.run}/{self.run_dict[self.run]["experiment"]}-{self.run_dict[self.run]["period"]}-{self.run}-cal-{self.run_dict[self.run]["timestamp"]}-plt_hit')
        self.update_strings()
        
    @param.depends("sort_by", watch=True)
    def update_strings(self):
        self.strings_dict, self.chan_dict, self.channel_map = sorter(self.path, 
                                                                    self.run_dict[self.run]["timestamp"],
                                                                    key=self.sort_by)

        self.param["string"].objects = list(self.strings_dict)
        self.string = f"{list(self.strings_dict)[0]}"


    @param.depends("run", "sort_by", "plot_type", "string")
    def view(self):
        figure=None
        if self.plot_type in ["Energy", "A/E", "Tau", "Alpha", "Valid_Energy_fits", "Valid_A/E_fits"]:
            figure = self.plot_types_dict[self.plot_type](self.run, 
                                             self.run_dict[self.run], 
                                             self.path, key=self.sort_by)
            
            
        elif self.plot_type in ["Baseline", "Spectra"]:
            try:
                figure = self.cached_plots[self.plot_type][self.sort_by][self.string]
            except KeyError:
                figure = self.plot_types_dict[self.plot_type](self.plot_dict, self.channel_map, 
                                   self.strings_dict[self.string],
                                   self.string, key=self.sort_by)
                if self.plot_type in self.cached_plots:
                    if self.sort_by in self.cached_plots[self.plot_type]:
                        self.cached_plots[self.plot_type][self.sort_by][self.string]=figure
                    else:
                        self.cached_plots[self.plot_type][self.sort_by]={self.string:figure}
                else:
                    self.cached_plots[self.plot_type]={self.sort_by:{self.string:figure}}
            
        elif self.plot_type == "Stability":
            try:
                figure = self.cached_plots[self.plot_type][self.string]
            except KeyError:
                figure = plot_fep_stability_channels2d(self.plot_dict, self.channel_map, 
                                                       self.strings_dict[self.string],
                                                       [2612,2616], self.string,
                                                        key=self.sort_by)
                if self.plot_type in self.cached_plots:
                    if self.sort_by in self.cached_plots[self.plot_type]:
                        self.cached_plots[self.plot_type][self.sort_by][self.string]=figure
                    else:
                        self.cached_plots[self.plot_type][self.sort_by]={self.string:figure}
                else:
                    self.cached_plots[self.plot_type]={self.sort_by:{self.string:figure}}
        else:
            figure = plt.figure()
            plt.close()
        
        return figure

        
def plot_energy(path, run_dict, det, plot, colour):
    
    
    cals= []
    times = []
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict[run]["period"]}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            cals.append(hit_pars_dict[f"ch{channel:03}"]["operations"]["cuspEmax_ctc_cal"]["parameters"]["a"]*20000 +\
                       hit_pars_dict[f"ch{channel:03}"]["operations"]["cuspEmax_ctc_cal"]["parameters"]["b"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    if len(cals)>0:
        cals =np.array(cals)
        plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    100*(cals-cals[0])/cals[0],
               legend_label=det, mode="after", line_width=2, line_color = colour)

        plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    100*(cals-cals[0])/cals[0],
                legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_energy_res(path, run_dict, det, plot, colour):
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    reses= []
    times = []
    for run in run_dict:
        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict[run]["period"]}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit_results.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            reses.append(hit_pars_dict[f"ch{channel:03}"]["ecal"]["cuspEmax_ctc_cal"]["Qbb_fwhm"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass

    if len(reses)>0:
        plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    (reses),
               legend_label=det, mode="after", line_width=2, line_color = colour)

        plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    (reses),
                legend_label=det, fill_color="white", size=8, color = colour)

    return plot


def plot_aoe_mean(path, run_dict, det, plot, colour):

    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    cals= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict[run]["period"]}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            cals.append(hit_pars_dict[f"ch{channel:03}"]["operations"]["AoE_Corrected"]["parameters"]["a"]*20000 +\
                       hit_pars_dict[f"ch{channel:03}"]["operations"]["AoE_Corrected"]["parameters"]["b"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    cals=np.array(cals)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(cals-cals[0])/cals[0],
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(cals-cals[0])/cals[0],
            legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_aoe_sig(path, run_dict, det, plot, colour):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    cals= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict[run]["period"]}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            cals.append(hit_pars_dict[f"ch{channel:03}"]["operations"]["AoE_Classifier"]["parameters"]["c"]*20000 +\
                       hit_pars_dict[f"ch{channel:03}"]["operations"]["AoE_Classifier"]["parameters"]["d"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    cals=np.array(cals)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(cals-cals[0])/cals[0],
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(cals-cals[0])/cals[0],
            legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_tau(path, run_dict, det, plot, colour):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    values= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        dsp_pars_file_path = os.path.join(prod_config["paths"]["par_dsp"],f'cal/{run_dict[run]["period"]}/{run}')
        dsp_pars_path = os.path.join(dsp_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_dsp.json')

        with open(dsp_pars_path,"r")as r:
            dsp_pars_path = json.load(r)
        try:
            values.append(float(dsp_pars_path[f"ch{channel:03}"]["pz"]["tau"][:-3]))
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    values=np.array(values)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(values-values[0])/values[0],
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(values-values[0])/values[0],
            legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_ctc_const(path, run_dict, det, plot, colour):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    values= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        dsp_pars_file_path = os.path.join(prod_config["paths"]["par_dsp"],f'cal/{run_dict[run]["period"]}/{run}')
        dsp_pars_path = os.path.join(dsp_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_dsp.json')

        with open(dsp_pars_path,"r")as r:
            dsp_pars_path = json.load(r)
        try:
            values.append(dsp_pars_path[f"ch{channel:03}"]["ctc_params"]["cuspEmax_ctc"]["parameters"]["a"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    values=np.array(values)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (values-values[0]),
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (values-values[0]),
            legend_label=det, fill_color="white", size=8, color = colour)


    return plot

def plot_tracking(run_dict, path, plot_func, string, key="String"):    

    strings_dict, soft_dict, chmap = sorter(path, run_dict[list(run_dict)[0]]["timestamp"], key=key)
    string_dets={}
    for stri in strings_dict:
        dets =[]
        for chan in strings_dict[stri]:
            dets.append(chmap[chan]["name"])
        string_dets[stri] = dets
        
    p = figure(width=700, height=600, x_axis_type="datetime")
    p.title.text = f"String No: {string}"
    p.title.align = "center"
    p.title.text_font_size = "15px"

    colours = Category10[10]
    
    for i, det in enumerate(string_dets[string]):
        try:
            plot_func(path, run_dict, det, p, colours[i])
        except:
            pass

    
    for run in run_dict:
        sp = Span(location=datetime.strptime(run_dict[run]["timestamp"], '%Y%m%dT%H%M%SZ'),
                  dimension='height',
                   line_color='black', line_width=1.5)
        p.add_layout(sp)

        label = Label(x=datetime.strptime(run_dict[run]["timestamp"], '%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 200), y=0, 
                     text=run)

        p.add_layout(label)


    p.add_layout(Title(text="Time", align="center"), "below")
    
    if plot_func == plot_energy:
        p.add_layout(Title(text="% Shift in keV", align="center"), "left")
    elif plot_func == plot_energy_res:
        p.add_layout(Title(text="FWHM at Qbb", align="center"), "left")
    elif plot_func == plot_aoe_mean:
        p.add_layout(Title(text="% Shift of A/E mean", align="center"), "left")
    elif plot_func == plot_aoe_sig:
        p.add_layout(Title(text="% Shift of A/E sigma", align="center"), "left")
    elif plot_func == plot_tau:
        p.add_layout(Title(text="% Shift PZ const", align="center"), "left")
    elif plot_func == plot_ctc_const:
        p.add_layout(Title(text="Shift CT constant", align="center"), "left")
        
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    
    return p
    
    
class cal_tracking(param.Parameterized):

    self.path = "/data1/users/marshall/prod-ref/v06.00"
    
    run_dict = gen_run_dict(path)
    
    periods = {}
    for run in run_dict: 
        if run_dict[run]['period'] not in periods:
            periods[run_dict[run]['period']] = [run]
        else:
            periods[run_dict[run]['period']].append(run)
            
    start_period = periods[list(periods)[0]]
    date_range = param.DateRange(default=(datetime.strptime(run_dict[sorted(start_period)[0]]["timestamp"],'%Y%m%dT%H%M%SZ')-dtt.timedelta(minutes = 100), 
                             datetime.strptime(run_dict[sorted(start_period)[-1]]["timestamp"],'%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 110)), 
                            bounds=(datetime.strptime(run_dict[sorted(start_period)[0]]["timestamp"],'%Y%m%dT%H%M%SZ')-dtt.timedelta(minutes = 100), 
                             datetime.strptime(run_dict[sorted(start_period)[-1]]["timestamp"],'%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 110)))
    
    
    plot_types = ["Energy", "Energy Res", "A/E Mean", "A/E Sigma", "Tau", "Alpha"]
    plot_type = param.ObjectSelector(default = "Energy", objects= plot_types)
    
    
    plot_dict = {"Energy": plot_energy,"Energy Res": plot_energy_res, "A/E Mean": plot_aoe_mean,
                "A/E Sigma": plot_aoe_sig, "Tau": plot_tau,  "Alpha": plot_ctc_const}
    
    sort_by_options = ["String", "CC4", "DAQ", "HV"]
    sort_by = param.ObjectSelector(default = "String", objects= sort_by_options)
    
    strings_dict, chan_dict, channel_map = sorter(self.path,
                                                    run_dict[sorted(start_period)[0]]["timestamp"],
                                                    sort_by_options[0])
    strings =  [f"{stri}" for stri in strings_dict]
    string = param.ObjectSelector(default = strings[0], objects = strings)
    
    
    
    @param.depends("date_range", watch=True)
    def _get_run_dict(self):
        valid_from = [datetime.timestamp(datetime.strptime(self.run_dict[entry]["timestamp"], '%Y%m%dT%H%M%SZ')) for entry in self.run_dict]
        pos1 = bisect.bisect_right(valid_from, datetime.timestamp(self.date_range[0]))
        pos2 = bisect.bisect_left(valid_from, datetime.timestamp(self.date_range[-1]))
        if pos1<0:
            pos1=0
        if pos2>=len(self.run_dict):
            pos2= len(self.run_dict)
        valid_idxs = np.arange(pos1,pos2,1)
        valid_keys = np.array(list(self.run_dict))[valid_idxs]
        out_dict = {key:self.run_dict[key] for key in valid_keys}
        return out_dict
    
    @param.depends("sort_by", watch=True)
    def update_strings(self):
        self.strings_dict, self.chan_dict, self.channel_map = sorter(self.path, 
                                                                    self.run_dict[list(self._get_run_dict())[0]]["timestamp"],
                                                                    key=self.sort_by)

        self.param["string"].objects = list(self.strings_dict)
        self.string = f"{list(self.strings_dict)[0]}"
        

    @param.depends("date_range", "plot_type", "string", "sort_by")
    def view(self):

        figure = plot_tracking(self._get_run_dict(), 
                               self.path, 
                               self.plot_dict[self.plot_type], self.string, key=self.sort_by)
        return figure
