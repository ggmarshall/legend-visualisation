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

import datetime as dtt
from  datetime import datetime

from legendmeta import LegendMetadata
from legendmeta.catalog import Props

from util import *
from summary_plots import *
from tracking_plots import *
from detailed_plots import *

class run_monitor(param.Parameterized):
    run = param.Selector(objects = ['r000'], default = 'r000')
    def __init__(self, path):
        super().__init__()
        self.path=path
        prod_config = os.path.join(self.path, "config.json")
        self.prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
        
        self.run_dict = gen_run_dict(self.path)
        all_runs = list(self.run_dict)
        
        self.param["run"].objects = all_runs
        self.run = all_runs[-1]

class detailed_cal_params(run_monitor):
    
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

    channel = param.Selector(objects = ["ch000"])
    parameter = param.ObjectSelector(default = "cuspEmax_ctc", objects = list(_options))
    plot_type = param.ObjectSelector(default = "cal_stability", objects= cal_plots)
    
    
    
    def __init__(self, path):
        super().__init__(path=path)
    
        self.update_plot_dict()
        self.update_plot_type()

    
    @param.depends('run', watch=True)
    def update_plot_dict(self):
        self.file = os.path.join(self.prod_config["paths"]["plt"],
                              f'hit/cal/{self.run_dict[self.run]["period"]}/{self.run}',
                            f'{self.run_dict[self.run]["experiment"]}-{self.run_dict[self.run]["period"]}-{self.run}-cal-{self.run_dict[self.run]["timestamp"]}-plt_hit')  
    
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


class cal_summary(run_monitor):
    
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (16, 6)
    plt.rcParams['figure.dpi'] = 100
    
    plot_types_dict = {"Energy_at_Qbb": plot_energy_resolutions_Qbb, 
                        "Energy_at_2.6MeV": plot_energy_resolutions_2614,
                        "A/E":get_aoe_results, 
                       "Tau":plot_pz_consts, "Alpha": plot_alpha, 
                       "Valid_Energy_fits": plot_no_fitted_energy_peaks, 
                       "Valid_A/E_fits": plot_no_fitted_aoe_slices,
                      "Baseline": plot_bls, "Spectra": plot_energy_spectra}
    
    plot_type = param.ObjectSelector(default = list(plot_types_dict)[0], objects= list(plot_types_dict))
    sort_by = param.ObjectSelector(default = list(sort_dict)[0], objects= list(sort_dict))
    string = param.ObjectSelector(default = "String:00", objects = ["String:00"])
    
    def __init__(self, path):
        super().__init__(path=path)
    
        self.update_plot_dict()
    
        self.plot_dict = None
        self.update_plot_dict()
    
    
    @param.depends('run', watch=True)
    def update_plot_dict(self):
        self.cached_plots={}
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
        if self.plot_type in ["Energy_at_Qbb", "Energy_at_2.6MeV", "A/E", "Tau", "Alpha", "Valid_Energy_fits", "Valid_A/E_fits"]:
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
    
    
class cal_tracking(param.Parameterized):
    
    plot_dict = {"Energy": plot_energy,"Energy_Res_Qbb": plot_energy_res_Qbb, "Energy_Res_2.6": plot_energy_res_2614,
                "A/E Mean": plot_aoe_mean, "A/E Sigma": plot_aoe_sig, "Tau": plot_tau,  "Alpha": plot_ctc_const}
    
    date_range = param.DateRange(default = (datetime.now()-dtt.timedelta(minutes = 10),
                                        datetime.now()+dtt.timedelta(minutes = 10)) , 
                                 bounds=(datetime.now()-dtt.timedelta(minutes = 110),
                                        datetime.now()+dtt.timedelta(minutes = 110)))
    sort_by = param.ObjectSelector(default = list(sort_dict)[0], objects= list(sort_dict))
    plot_type = param.ObjectSelector(default = list(plot_dict)[0], objects= list(plot_dict))
    string = param.ObjectSelector(default=0, objects=[0])
    
    def __init__(self, path):
        super().__init__()
        self.path=path
        
        self.run_dict = gen_run_dict(self.path)

        self.periods = {}
        for run in self.run_dict: 
            if self.run_dict[run]['period'] not in self.periods:
                self.periods[self.run_dict[run]['period']] = [run]
            else:
                self.periods[self.run_dict[run]['period']].append(run)

        start_period = self.periods[list(self.periods)[0]]        
        
        self.param["date_range"].bounds = (datetime.strptime(self.run_dict[sorted(start_period)[0]]["timestamp"],'%Y%m%dT%H%M%SZ')-dtt.timedelta(minutes = 100), 
                                 datetime.strptime(self.run_dict[sorted(start_period)[-1]]["timestamp"],'%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 110))
        self.date_range = (datetime.strptime(self.run_dict[sorted(start_period)[0]]["timestamp"],'%Y%m%dT%H%M%SZ')-dtt.timedelta(minutes = 100), 
                            datetime.strptime(self.run_dict[sorted(start_period)[-1]]["timestamp"],'%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 110))
        
        self.update_strings()
    
    
    
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

if __name__ == "__main__":
    path = "/data1/users/marshall/prod-ref/v06.00/"
    pn.extension()
    pn.extension("plotly")

    pars = detailed_cal_params(path=path)
    buttons = pn.Param(pars.param, widgets={
        'run': {'widget_type': pn.widgets.Select, 'width':150},
        'channel': {'widget_type': pn.widgets.Select, 'width':150},
        'parameter': {'widget_type': pn.widgets.RadioButtonGroup,  'button_type':'primary', 
                    'orientation':"vertical", 'width':150},
        'plot_type': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success', 
                    'orientation':"vertical", 'width':150}}
    )

    cal = cal_summary(path=path)
    cal_buttons = pn.Param(cal.param, widgets={
        'run': {'widget_type': pn.widgets.Select, 'width':150},
        'sort_by': {'widget_type': pn.widgets.RadioButtonGroup,  'button_type':'primary', 
                    'orientation':"vertical", 'width':150},
        'plot_type': {'widget_type': pn.widgets.RadioButtonGroup,  'button_type':'primary', 
                    'orientation':"vertical", 'width':150},
        'string': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success', 
                    'orientation':"vertical", 'width':150}}
    )


    ct = cal_tracking(path=path)
    ct_buttons = pn.Param(ct.param, widgets={
        'date_range': {'widget_type': pn.widgets.DateRangeSlider, 'width':150},
        'plot_type': {'widget_type': pn.widgets.RadioButtonGroup,  'button_type':'primary', 
                    'orientation':"vertical", 'width':150},
        'sort_by': {'widget_type': pn.widgets.RadioButtonGroup,  'button_type':'primary', 
                    'orientation':"vertical", 'width':150},
        'string': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success', 
                    'orientation':"vertical", 'width':150}}
    )


    pn.Tabs(("Cal Summary", pn.Column("# Summary Cal Plots", 
          pn.Row(
                 cal_buttons, 
              cal.view)
         ))
         , 
       ("Cal Tracking" , pn.Column("# Cal Tracking", 
          pn.Row(
                 ct_buttons, 
              ct.view)
         )),
    
        ("Detailed_cal_plots",pn.Column("# Detailed Calibration Plots", 
          pn.Row(
                 buttons, 
              pars.view)
         )
        )
       )