#%%
from data_prep.exp import experiment
import numpy as np
from scipy.signal import find_peaks,peak_prominences,peak_widths
import matplotlib.pyplot as plt
from typing import Union
import pandas as pd
from data_prep.utils import save_pickle_file,open_pickle_file,plot_mea_map,fix_roc,find_best_intensity_auc_xrmse,find_best_threshold_response
import shutil
import os
from mea_reservoir.meaRC import meaRC
#from profiler_utility import profile,print_stats
from time import time
from scipy.interpolate import interp1d
from data_prep import metric
import math
from sklearn.metrics import mean_squared_error
import copy
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve,auc,RocCurveDisplay
import seaborn as sns

#%%


#%%
class analyzer():
    '''
    analyzer is a class containing all the methods used to analyze experiment data (both simulation or experimental recording).
    Must be initialized with an object of type experiment.
    '''

    def __init__(self, exp: experiment):

        if type(exp) != experiment:
            raise TypeError('An object of type experiment must be passed to analyzer class. Abort.')
        else:
            self.exp = exp

    def _index_to_xdata(self,xdata, indices):
        "interpolate the values from signal.peak_widths to xdata"
        ind = np.arange(len(xdata))
        f = interp1d(ind,xdata)
        return f(indices)

    def __basic_network_burst_detection_alg(self,data:dict,time_th:Union[int,float],ch_th:int = 2):

        '''
        Basic version of network burst detection algorithm. 
        A network burst is identified when there are a number of channel >= ch_th that are bursting together (difference between burst time <= time_th )

        Input:
            - data: burst data dictionary
            - time_th: time threshold 
            - ch_th: threshold on the channel number necessary to consider a network burst
        '''
        burst_start = data['burst_start'].to_numpy(copy=True)
        
        NB_results_list=[]
        NB_length_list = []

        idx = 0 # index of the first burst

        while idx < len(burst_start)-1:

            i = idx 
            j = idx + 1 # index of the next burst

            while  j < len(burst_start) and burst_start[j]-burst_start[i] <= time_th:

                j += 1 
                i += 1

            partecipating_elec=data['channel_id'].iloc[idx:j].nunique(dropna=True) # we measure how many unique electrodes are partecipating to the network burst
            

            if partecipating_elec >= ch_th: # if there are more consecutive burst than n, it means we have a network burst

                    nb_length = data['burst_end'].iloc[i] - burst_start[idx] # nb length

                    NB_results_list.append(burst_start[idx]) # the time in millisecond of the first spike of the first burst in each network bursts is append
                    NB_length_list.append(nb_length) # length expressed in ms       

            idx += (j-idx)

        return NB_results_list,NB_length_list




    

    #@jit
    def __basic_burst_detection_alg(self,data:np.ndarray,isith:Union[int,float],n:int=3)-> np.ndarray:
        '''
        Basic version of burst detection algorithm which uses ISI threshold to identify bursts, defined as n spikes separated less than ISIth.

        Input:
            - data: spikes data of one channel (n_spikes) of the channel data matrix (n_ch x n_spikes)
        '''

       
        data = data[~np.isnan(data)]*(1/self.retrieve_param_exp('sampling_fr')*1000) # remove nan and obtain the spike time in millisecond
       
        tmp_ch_burst_list = [] 

        idx = 0

        while idx < len(data)-1:

            i = idx
            j = idx + 1

            while  j < len(data) and data[j]-data[i] <= isith:

                j += 1
                i += 1

            if (j-idx) >= n: # if there are more consecutive spikes than n, it means we have a burst

                burst = data[idx:j]

                tmp_ch_burst_list.append([burst[0],burst[-1],len(burst),burst[-1]-burst[0]])  # for each burst we save the time of the first and last spikes in the burst, the total number of spikes and the burst duration idx+=j

            idx += (j-idx)

        return tmp_ch_burst_list

    def __calculate_merged_IBI(self,burst_data:dict):
        

        tmp = sorted([burst[0] for ch in burst_data for burst in burst_data[ch]]) # for each channel, burst start are retrieved and sorted temporally, meaning that spikes are not divided anymore based on channel

        merged_IBI = np.diff(np.array(tmp)) # the Inter-burst interval is calculated considering all channel in a single histogram

        merged_IBI = np.log10(merged_IBI[merged_IBI!=0]) # log IBI is calculated

        return merged_IBI
    
    def __calculate_IBIth(self,IBI:np.ndarray):

        def plot_ibi_bar(count,bins,ibith=None):
            plt.stairs(count,np.power(10,bins),facecolor=[0.,.02,.02,.44], edgecolor=[0., 0., 0., 0.98], linewidth=2,fill=True)
            ax = plt.gca()
            plt.xlabel('IBI [ms]')
            ax.set_xscale('log')
            
            plt.ylabel('Density')
            if ibith!=None:
                
                ax.scatter(ibith,count[list(bins).index(np.round(np.log10(ibith),1))],color='red',marker='*',linewidth=1.5,s=80,label='Estimated Int. Time')
            #plt.title('Cumulative log-ISI distribution')
            plt.legend(fontsize = 12)
            plt.savefig(os.path.join(self.exp.results_path,'cumulative_ibi_histogram.pdf'),bbox_inches='tight')
            plt.close()

        
        plt.close()
        #Histogram of logIBI
        histo_bins=np.round(np.arange(min(IBI),max(IBI),0.1),1) # the bins have fixed width of 0.1 
       
        histogram=plt.hist(IBI,bins=histo_bins,density=True,facecolor=[0.,.02,.02,.44], edgecolor=[0., 0., 0., 0.98], linewidth=2)
        plt.xlabel('log-IBI [ms]')
        plt.ylabel('Density')
        #plt.title('Cumulative log-IBI distribution')
        plt.savefig(os.path.join(self.exp.results_path,'cumulative_log_ibi_histogram.pdf'),bbox_inches='tight')
        plt.close()

        


        binned_values=np.array(histogram[0])

        peaks,_ = find_peaks(binned_values) # peaks in IBI histogram

        #possible_peaks = peaks[(histo_bins[peaks]>= 0) & (histo_bins[peaks]<=1)] # only peaks in the range 0-1 (1-10 ms) are valid

        peaks_prominence = peak_prominences(binned_values,peaks)[0]
        peaks_width = peak_widths(binned_values,peaks)[0]

        found_flag = False

        if len(peaks) == 0: # if no peaks are identified, a standard IBI threshold (IBIth) value of 5 ms is used
            print('IBITh: No peaks identified. Standard value of 5 ms is used.')
            return 5 # IBI threshold expressed in millisecond
        
            
        else: # if there is one or more peaks

            peaks_significance = peaks_width*peaks_prominence 

            indexes = list(range(len(peaks_significance)))
            indexes.sort(key = peaks_significance.__getitem__,reverse=True) # we sort the peaks by significance 

            sorted_peak = list(map(peaks.__getitem__,indexes)) # the peaks indexes are sorted by significance

            #first_peak_idx = np.argmax(peaks_significance) # the most significant peak
            
            for peak in sorted_peak: # among all possible peaks

                peak_x_value = histo_bins[peak] # x value of the peak 

                if peak_x_value > 1: # if peak is above 10 ms
                    continue

                elif peak_x_value < np.log10(2): # if peak is less than 2 ms
                    found_flag = True
                    plot_ibi_bar(histogram[0],histo_bins,2)
                    return 2 # IBIth expressed in ms

                elif ((peak_x_value>= np.log10(2)) and (peak_x_value<1)):
                    found_flag = True
                    ibith = round(10**(peak_x_value),1)
                    plot_ibi_bar(histogram[0],histo_bins,ibith)
                    return  ibith # IBIth in millisecond

            if found_flag == False:
                
                print('No IBI threshold could be found. Returning None.')
                plot_ibi_bar(histogram[0],histo_bins,None)
                return None

    def __calculate_merged_ISI(self,ch_data: np.ndarray) -> np.ndarray:
        '''
        The function calculate the inter-spike interval (ISI) considering all channels.

        Input:
            - data: channel data matrix (n_ch x n_spikes)
        '''

        merged_ISI = [] # the Inter-spike interval distribution is calculate considering all channel in a single histogram
        for ch in range(ch_data.shape[0]):
            tmp = ch_data[ch,:][~np.isnan(ch_data[ch,:])]*(1/self.retrieve_param_exp('sampling_fr')*1000) # remove nan and obtain the spikes time in millisecond
            if len(tmp)>2: # if there are at least two spikes
                merged_ISI.extend(np.diff(tmp))
            else:
                continue
        merged_ISI = np.array(merged_ISI)
        merged_ISI=np.log10(merged_ISI[merged_ISI!=0]) # eventual zero values are removed and the logarithm is calculated obtaining log ISI
        
        return merged_ISI

    def __calculate_ISIth(self,ISI:np.ndarray)-> float:

        '''
        The function calculates the ISI threshold (ISIth) from 1D log ISI array
        '''

        def plot_isi(counts,bins,isith=None):
            tmp_hist = plt.stairs(counts,np.power(10,bins),facecolor=[0.,.02,.02,.44], edgecolor=[0., 0., 0., 0.98], linewidth=2,fill=True)
            ax = plt.gca()
            plt.xlabel('ISI [ms]')
            ax.set_xscale('log')
            
            plt.ylabel('Density')
            if isith!=None:
                ax.scatter(isith,counts[list(bins).index(np.round(np.log10(isith),1))],color='red',marker='*',linewidth=1.5,s=80,label='ISIth')
            #plt.title('Cumulative log-ISI distribution')
            plt.legend(fontsize = 12)
            plt.savefig(os.path.join(self.exp.results_path,'cumulative_isi_histogram.pdf'),bbox_inches='tight')
            plt.close()
        

        #Histogram of logISI 
        plt.close()
        try:
            min_isi = min(ISI)

        except:
            min_isi = -2
            histo_bins=np.round(np.arange(min_isi,max(ISI),0.1),1) # the bins have fixed width of 0.1 

        else:
            histo_bins=np.round(np.arange(min(ISI),max(ISI),0.1),1) # the bins have fixed width of 0.1 
        
        
        
        histogram=plt.hist(ISI,bins=histo_bins,density=True,facecolor=[0.,.02,.02,.44], edgecolor=[0., 0., 0., 0.98], linewidth=2)
        plt.xlabel('log-ISI [ms]')
        plt.ylabel('Density')
        plt.savefig(os.path.join(self.exp.results_path,'cumulative_log_isi_histogram.pdf'),bbox_inches='tight')
        plt.close()
        
        
    
        binned_values=np.array(histogram[0])

        peaks,_ = find_peaks(binned_values) # peaks in ISI histogram

        peaks_prominence = peak_prominences(binned_values,peaks)
        peaks_width = peak_widths(binned_values,peaks,rel_height=0.5,prominence_data=peaks_prominence)[0]
        peaks_width = (peaks_width/2)*0.1 #TODO must be changed with _index_to_xdata function as peak_widths calculate the width on binned values from the histogram, the peak_width value must be translate to x value
        peaks_prominence = peaks_prominence[0]

        if len(peaks) == 0: # if no peaks are identified, a standard ISI threshold (ISIth) value of 100 ms is used
            print('No peaks identified. Standard value of 100 ms is used.')
            return 100 # ISI threshold expressed in millisecond
        
        elif len(peaks) ==1: # if only one peak is identified

            plot_isi(histogram[0],histo_bins,10^(histo_bins[peaks] + peaks_width/2))
    
            return 10^(histo_bins[peaks] + peaks_width/2) # ISIth in ms
            
        else: # at least two peaks have been identified

            peaks_significance = peaks_width*peaks_prominence 

            #first peak
            first_peak_idx = np.argmax(peaks_significance) # the most significant peak
            first_peak_x = histo_bins[peaks[first_peak_idx]] # x value of the logISI histogram of the most significant peak
            first_peak_width = peaks_width[first_peak_idx]

            if first_peak_x < 1.5:
                #second peak
                second_peak_idx = np.argmax(peaks_significance[histo_bins[peaks]>1.5]) + np.sum(~(histo_bins[peaks]>1.5))
                second_peak_x = histo_bins[peaks[second_peak_idx]]  # x value of the logISI histogram of the second most significant peak 
                second_peak_width = peaks_width[second_peak_idx]
            else:
                #second peak
                second_peak_idx = np.argmax(peaks_significance[histo_bins[peaks]<1.5]) 
                second_peak_x = histo_bins[peaks[second_peak_idx]]  # x value of the logISI histogram of the second most significant peak 
                second_peak_width = peaks_width[second_peak_idx]
            
    
            if first_peak_x > second_peak_x:
                isith_idx = ((first_peak_x-(first_peak_width/2))+(second_peak_x+(second_peak_width/2)))/2 # this index points to the x value in the histogram representing a valley between the two most significant peaks

            elif first_peak_x < second_peak_x:
                isith_idx = ((first_peak_x+(first_peak_width/2))+(second_peak_x-(second_peak_width/2)))/2

            isith_idx = np.round(isith_idx,1)

            if isith_idx >= 2: # if the index is greater then 2, which correspond to 10^2 = 100 ms
                 plot_isi(histogram[0],histo_bins,10^(2))
                 return 100 # ms
            else:
                plot_isi(histogram[0],histo_bins,round(10**(isith_idx),2))
                return  round(10**(isith_idx),2) # ISIth in millisecond


    def __windowing(self,spikes_data_ms,start_time:Union[int,float],int_time:int,batch_time:int):
        '''
        The function creates one window (matrix of dimension n_ch x m, with m = batch_time/int_time)
        '''
         
        batch_time = round((batch_time//int_time +1 )*int_time,1)

        window_start = start_time # start of the window in millisecon
        window_end = start_time + batch_time # end of the window in ms
        
        m = int(batch_time/int_time)

        bin_window = np.zeros((spikes_data_ms.shape[0],m)) # binarize spike matrix 

        row_idx,col_idx = np.where(((spikes_data_ms>=window_start) & (spikes_data_ms<= window_end) & (~np.isnan(spikes_data_ms)))) # retrieve the position where spikes satisfy the condition (basically are inside the bin_window)

        for idx in range(len(row_idx)):

            r_i = row_idx[idx]
            c_i = col_idx[idx]

            val = spikes_data_ms[r_i,c_i] # retrieve spike time in ms

            val = np.round(val - window_start,1) # center the spike time based on the window start

            col_idx_window =int( val // int_time ) # retrieve the correct column idx for the window

            if col_idx_window == m:
                col_idx_window =col_idx_window-1 # if one spike happen to be exactly at the end of thw window, the spike is counted in the last column of the window

            if bin_window[r_i,col_idx_window] == 0:

                bin_window[r_i,col_idx_window]=1
            else:
                bin_window[r_i,col_idx_window]+=1

        return bin_window




    def retrieve_data_exp(self,name:str):
        '''
        The function retrieves the desired data from the data dictionary of the experiment under analysis
        '''
        return self.exp.get_data(name)
    
    def retrieve_param_exp(self,name:str):
        '''
        The function retrieves the desired param from the params dictionary of the experiment under analysis
        '''
        return self.exp.get_param(name)
    
    def set_data_exp(self,name:str,data):
        '''
        The function add data to the data dictionary of the experiment
        '''
        self.exp.set_data(name,data)

    def set_param_exp(self,name:str,param):
        '''
        The function add param to the params dictionary of the experiment 
        '''
        self.exp.set_param(name,param)


    def plot_single_population_isi(self,idx:int,filt=False):
        
        if filt == False:
            background = self.retrieve_data_exp('background')[0][idx]
            channel_list = self.retrieve_data_exp('background')[1]

            stimulation = self.retrieve_data_exp('stimulation')
        else:
            background = self.retrieve_data_exp('filt_background')[0][idx]
            channel_list = self.retrieve_data_exp('filt_background')[1]

            stimulation = self.retrieve_data_exp('filt_stimulation')

        tmp_stim = []
        for stim in stimulation:
            tmp = stimulation[stim][1][idx]
            tmp = tmp[~np.isnan(tmp)]
            tmp_stim.append(tmp)

        ch_data = np.concatenate((background,*tmp_stim))

            
            
        ISI = np.diff(np.sort(ch_data[~np.isnan(ch_data)]*(1/self.retrieve_param_exp('sampling_fr')*1000)))
        log_ISI = np.log10(ISI[ISI!=0])
        channel_idx = channel_list[idx]

        #Histogram of logISI 
        plt.close()
        try:
            min_isi = min(log_ISI)

        except:
            min_isi = -2
            histo_bins=np.round(np.arange(min_isi,max(log_ISI),0.1),1) # the bins have fixed width of 0.1 

        else:
            histo_bins=np.round(np.arange(min(log_ISI),max(log_ISI),0.1),1) # the bins have fixed width of 0.1 
        
        histogram=plt.hist(log_ISI,bins=histo_bins,density=True,facecolor=[0.,.02,.02,.44], edgecolor=[0., 0., 0., 0.98], linewidth=2)
        plt.close()

        tmp_hist = plt.stairs(histogram[0],np.power(10,histo_bins),facecolor=[0.,.02,.02,.44], edgecolor=[0., 0., 0., 0.98], linewidth=2,fill=True)
        ax = plt.gca()
        plt.xlabel('ISI [ms]')
        ax.set_xscale('log')
            
        plt.ylabel('Density')
        
        plt.title(f'Channel {channel_idx}')
        plt.savefig(os.path.join(self.exp.results_path,f'channel_{channel_idx}_isi_histogram.pdf'),bbox_inches='tight')
        plt.close()

    def raster_plot(self,start,stop,filt = False):

        if filt == False:
            background = self.retrieve_data_exp('background')[0]
            channel_list = self.retrieve_data_exp('background')[1]

            stimulation = self.retrieve_data_exp('stimulation')
        else:
            background = self.retrieve_data_exp('filt_background')[0]
            channel_list = self.retrieve_data_exp('filt_background')[1]

            stimulation = self.retrieve_data_exp('filt_stimulation')

        tmp_stim = []
        for stim in stimulation:
            tmp = stimulation[stim][1]
            tmp_stim.append(tmp)

        data = np.concatenate((background,*tmp_stim),axis=1)

        tmp_raster = []
        for ch in range(data.shape[0]):
            tmp = data[ch]
            tmp = tmp[~np.isnan(tmp)]*0.01
            tmp = tmp[(tmp>start) & (tmp<stop)]
            tmp_raster.append(tmp)

        fig,ax = plt.subplots()

        ins_ax = ax.inset_axes([-0.03,0.1,1,1])
        ins_ax.eventplot(tmp_raster,linewidths=1,colors='gray')

        ax.spines[['top','bottom','left','right']].set_visible(False)
        ax.tick_params(axis='both',which='both',labelbottom=False,labelleft=False,bottom=False,left=False,right=False)
        ins_ax.spines[['top','bottom','right']].set_visible(False)

        x_tick_pos = ins_ax.get_xticks()[0:2]

        ins_ax_x_lim = ins_ax.get_xlim()

        

        ins_ax.tick_params(axis='both',which='both',labelbottom=False,bottom=False,right=False)
        ins_ax.set_ylabel('# channel')
        ins_ax.set_ylim(-5)
        ins_ax.axhline(-5,0.05,((x_tick_pos[1]-x_tick_pos[0])/(ins_ax_x_lim[1]-ins_ax_x_lim[0]))+0.05,color='black')
        ax.text(0.05,0.05,f'{(x_tick_pos[1]-x_tick_pos[0])/1000} s',size=15)
        
        plt.savefig(os.path.join(self.exp.results_path,f'raster_plot.pdf'),bbox_inches='tight')
        plt.close()

    
    def calculate_isith(self,data:np.ndarray,just_plot=False):
        '''
        Wrapper function for ISI threshold calculation from spiking data.
        The ISIth is added in the params dictionary of the experiment.

        Input:
            - data: matrix (n_ch x m) containing the spike timestamp (m) for all channel (n_ch)
        '''

        merged_isi = self.__calculate_merged_ISI(data) # the ISI considering the data of all the channel is calculated
        isith = self.__calculate_ISIth(merged_isi) # ISI threshold expressed in millisecond

        if just_plot == False:
            self.set_param_exp('isith',isith) # the ISI th is added as parameter in the params dictionary of exp istance



    def calculate_burst(self,data:np.ndarray,ch_list:list,ABR_th:Union[float,int] = 0):
        '''
        Wrapper function for burst detection algorithm.
        The data regarding burst are saved in the data dictionary under the key 'burst' of the experiment under analysis.

        Input:
            - data: matrix (n_ch x n_spikes) containing the spike timestamp (n_spikes) for all channel (n_ch)
            - ch_list: list containing the analyzed channel in order of analysis
            - ABR_th: threshold value for removing channel with an average burst rate (burst/min) < ABR_th (default: 0)
        '''

        burst_data = {} # burst data dictionary
        duration_in_min = self.exp.get_param('duration_in_sec')/60 # recording duration expressed in minutes

        for ch in range(data.shape[0]): # for each channel in the matrix the bursts are calculated

                tmp_burst_data = self.__basic_burst_detection_alg(data[ch],self.exp.get_param('isith'))

                ABR = len(tmp_burst_data)/duration_in_min # average burst rate (burst/min)

                if ABR >= ABR_th: # only channel with an ABR >= ABR_th are further considered

                    burst_data[ch_list[ch]]=tmp_burst_data  # the key value of the dictionary is the actual channel number


        self.set_data_exp('burst',burst_data)

    def calculate_network_burst(self,data:dict):
        '''
        Wrapper function for network burst detection algorithm.
        The data regarding network burst are saved in the data dictionary under the key 'network_burst' of the experiment under analysis.

        Input:
            - data: burst data dictionary
        '''

        tmp = [(ch,burst[0],burst[1],burst[3]) for ch in data for burst in data[ch]] # for each channel, a tuple (channel, burst start, burst end, burst duration) is cretated for each burst 

        NB_df=pd.DataFrame(tmp,columns=['channel_id','burst_start','burst_end','burst_duration']) # burst start, end, and duration are expressed in millisecond
        NB_df.sort_values(by='burst_start',ignore_index=True,inplace=True)

        average_burst_length = NB_df['burst_duration'].mean()
        
        max_time_between_burst = round(average_burst_length/2,2)

        nb_data,nb_length = self.__basic_network_burst_detection_alg(NB_df,max_time_between_burst) # nb_data is a list containing the time in millisecond of the first spike of the first burst composing each detected network burst

        self.set_data_exp('network_burst',[nb_data,nb_length])
        


    def calculate_ibith(self,data:dict,just_plot:bool=False):
        '''
        Wrapper function for IBI histogram calculation from burst data of all electrodes. 
        From the IBI histogram, the integration time is calculated and added to the params dictionary under the key 'int_time' of the experiment under analysis
        
        Input:
            - data: burst data dictionary

        '''

        merged_IBI = self.__calculate_merged_IBI(data)
        ibith = self.__calculate_IBIth(merged_IBI)
        if just_plot == False:
            self.set_param_exp('ibith',ibith)
            self.set_param_exp('int_time',round(ibith,1))

    def filter_spikes_data(self,data_name:str,afr_th: Union[int,float]):
        '''
        The function filters channels having a average firing rate (AFR) < threshold (afr_th). data can be both background or stimulation data, which is specified by the data_name variable.
        The filtered data are added to the data dictionary of the experiment with the same name + filt_ at the beginning (example: background -> filt_background)
        '''

        spikes,elec_list = self.retrieve_data_exp(data_name)

        afr_mask = (np.count_nonzero(~np.isnan(spikes),axis=1)/self.exp.get_param('duration_in_sec'))>=afr_th # AFR (spikes/s) mask, where True indicates channel satisfying the condition

        filt_spikes = spikes[afr_mask,:] # select only channel with AFR >= afr_th

        filt_elec_list = np.array(elec_list)[afr_mask] # list of filtered electrode

        self.set_data_exp(f'filt_{data_name}',[filt_spikes,filt_elec_list.tolist()])

    def filter_stim_data(self,
                         stimulation_data,
                         filtered_elec_idx):
        filt_stim_data = {}
        
        filtered_elec_idx=np.array(filtered_elec_idx)

        for stim in stimulation_data:

            stim_protocol = stimulation_data[stim][0]

            stim_spikes = stimulation_data[stim][1]

            elec = np.array(stimulation_data[stim][2])
            
            to_filt = np.isin(elec,filtered_elec_idx)

            stim_spikes_filt = stim_spikes[to_filt]
            filt_elec = elec[to_filt].astype(int)
            if stim in filt_elec:
                filt_elec = filt_elec.tolist()
                filt_stim_data[stim]=[stim_protocol,stim_spikes_filt,filt_elec]
        
        self.set_data_exp('filt_stimulation',filt_stim_data)

    
    def create_training_data(self,tr_split:float=0.85,
                             data_name:str = 'background', # either 'background' or 'filt_background'
                             extra_steps:Union[int,float]=0 # extra ms to add at the end of each network burst
                             ):
        '''
        The function creates training and validation dataset starting from background activity data

        Input:
            - tr_split: indicates the percentage of data used for training and validation (default: 0.85 )
        '''

        #self.__check_data_before_training() #checks that all necessary data and parameters to create training/validation dataset are present

        nb_data,nb_length = self.retrieve_data_exp('network_burst') # list containing spike time of the first spike of the first burst for each network burst

         
        back,_ = self.retrieve_data_exp(data_name) # background activity data

        int_time = self.retrieve_param_exp('int_time') 
        
        back = np.round(back * (1/self.retrieve_param_exp('sampling_fr')*1000),2) # spikes in millisecond
        
        all_2d_windows = []

        for num,nb in enumerate(nb_data):
            nb_start = nb
            window_2d = self.__windowing(back,nb_start,int_time,nb_length[num]+extra_steps) # it creates a 2D matrix (n_ch x m), where m = batch_time/int_time
            
            all_2d_windows.append(window_2d)

        temp = list(zip(all_2d_windows,nb_length))
        self.exp.rng.shuffle(temp) # the data are shuffled as each Network burst is considered indipendent from all the others

        
        training_idx = round(len(all_2d_windows)*tr_split) # the first training_idx window will be used for training

        max_value_data = max([np.max(nb) for nb in all_2d_windows]) # maximum value 

        training,validation = all_2d_windows[0:training_idx],all_2d_windows[training_idx:] # training and validation data


        self.set_data_exp('training',training) # training and validation are saved in the data dictionary
        self.set_data_exp('validation',validation)
        self.set_param_exp('norm_factor_back',max_value_data) # the maximum value in the matrix is stored as can be used to normalize the entire matrix by its highest value


    
    #@profile
    def characterize_meaRC(self,
                       training:list, # training data
                       validation:list, # validation data
                       alphas:Union[None,list,np.ndarray] = None, # array of alphas value to test. if nothing is passed [0,1] with step of 0.1 will be tested
                       ms:Union[None,list,np.ndarray] = None, #array of m values to test. if nothing is passed [1,2,5,10,20] will be tested
                       rng:int=1, # an int which will be used as seed and to create a np.random number generator which will be used truoghout the code . If nothing is passed 1 will be used as seed to ensure consistency.
                       n_jobs:int= 1, # number of workers to use. -1 to use all available workers
                       repeat:int=10, # how many repetition for each parametrization
                       save_data: bool = False, #if set to true the meaRC model dictionary will be saved as a pickle file
                       **kwargs):
        '''
        The function perform optimization of the meaRC model by performing a gridsearch for alpha and m.

        Input:
            - training: normalized training data
            - validation: normalized validation data
            - alphas (optional): array of alphas value to test. if nothing is passed [0,1] with step of 0.1 will be tested
            - ms (optional): array of m values to test. if nothing is passed [1,2,5,10,20] will be tested
        '''

        mearc_dicto = {} # dictionary containing the results of each tested meaRC model parametrization
        
        # handling ms and alphas
        if isinstance(alphas,type(None)):
            alphas = np.arange(0,1.1,0.1)
        if isinstance(ms,type(None)):
            ms = [1,2,5,10,20]

        # check data
        if not isinstance(training,list):
            raise ValueError (f' The training data must be a np.ndarray') #TODO add the exact dimension of the training data
        if not isinstance(validation,list):
            raise ValueError (f' The validation data must be a np.ndarray') #TODO add the exact dimension of the validation data

            

        for m in ms: # for each m value different alpha values will be tested

            mearc_dicto[m]={}

            for alpha in alphas:
                mearc_dicto[m][alpha]={}

                tmp_rng = rng
                for rep in range(repeat): #
                    

                    meaRC_model = meaRC(training,validation,m=m,alpha=alpha,rng=tmp_rng,n_jobs=n_jobs,**kwargs)
                    
                    meaRC_model.run()
                   
                    mearc_dicto[m][alpha][rep] = meaRC_model
                    
                    tmp_rng+=1 # changing seed for obtaining different initialization
        

        if save_data == True:

            save_pickle_file(mearc_dicto,os.path.join(self.exp.results_path,'characterized_mearc.pickle')) # meaRC dictionary is stored as a pickle file
            #save_yaml_file(mearc_dicto,os.path.join(self.exp.results_path,'characterized_mearc.yaml'))
        return mearc_dicto
    

    

    def _PSTH(self,
              stim_protocol, # Stimulation protocol for which PTSH should be calculated
              steps_number # how many steps in time should be considered. 
              ):

        '''
        The function calculates the PSTH from raw stimulation data of one protocol
        '''
        
        timing_df = stim_protocol[0] # pd.dataframe containing starting and ending time of each pulse of the stimulation
        data = stim_protocol[1]*(1/self.retrieve_param_exp('sampling_fr')*1000) # spike data. stimulation data are stored as sample number (so timestep) but for the analysis we convert them in time in ms


        all_2d_windows = []

        int_time = self.retrieve_param_exp('int_time')

        window_dimension = min((500//int_time)*int_time,steps_number*int_time) # maximum windows dimension allowed is 500 ms

        for pulse in range(len(timing_df)):
            start_pulse = timing_df.iloc[pulse]['start']

            window_2d = self.__windowing(data,start_pulse,int_time,window_dimension) # it creates a 2D matrix (n_ch x m), where m = batch_time/int_time
            
            all_2d_windows.append(window_2d)

        all_2d_windows = np.array(all_2d_windows) # all_2d_windows is a 3-D matrix (n_NB x n_ch x m) composed by 2-D matrix (n_ch x m) with each entry i,j being the 
                                                           # spikes count
        
        averaged_window = np.mean(all_2d_windows,axis=0) # averaged response to the stimulation calculated as average across all pulses
        averaged_window = averaged_window[:,:steps_number] # to retain only the desired number of steps

        max_value = np.max(averaged_window) # highest values inside the matrix, which is used as normalization factor
        
        return averaged_window,max_value
    
    def _PSTH_integral(self,
                       normalized_psth_window:np.ndarray, # psth window of the stimulation protocol
                       ):
        '''
        Function calculates the integral over time of the normalized PTSH for each electrode
        '''

        integral = np.trapz(normalized_psth_window,axis=1)

        return integral

    def calculate_psth(self,
                       steps_number:int = 100, # how many steps in time should be considered. default 100
                       data_name:str = 'stimulation' # name of the data to use, either 'stimulation' or 'filt_stimulation'
                       ):
        '''
        PTSH of each stimulation protocol is calculated 
        '''

        # create folder to store prediction  data
        pred_folder_path = os.path.join(str(self.exp.results_path),'stimulus_prediction')
        if not os.path.exists(pred_folder_path):
            os.makedirs(pred_folder_path)

        analyzed_stim_dict = {}

        stimulation_data = self.retrieve_data_exp(data_name) 

        for stim in stimulation_data: # for each stimulation protocol

            elec_list = stimulation_data[stim][2] # electrode list

            try: ##### PLEASE MODIFY THIS SHIT, SAME CONDITION FOR EXPERIMENT AND SIMULATION (SO SAME BETWEEN FILTER AND NOT FILTER DATA)
                elec_idx = elec_list.index(stim)
            except:
                elec_idx = elec_list.index(str(stim))

            ptsh_window,psth_norm_factor = self._PSTH(stimulation_data[stim],steps_number=steps_number)

            psth_int = self._PSTH_integral(ptsh_window/psth_norm_factor)

            analyzed_stim_dict[stim]={'psth':ptsh_window,
                                       'norm_factor_stim':psth_norm_factor,
                                       'integral':psth_int,
                                       'elec_idx':elec_idx}
            
            plt.close()
            tmp = ptsh_window/psth_norm_factor
            sns.heatmap(tmp,cmap='coolwarm',xticklabels=np.array(list(range(1,tmp.shape[1]+1,1)))*self.exp.get_param('int_time'))
            plt.title('Ground truth PSTH')
            plt.xlabel('Time (ms)')
            plt.ylabel('Channel #')
            plt.savefig(os.path.join(pred_folder_path,f'gt_psth_stim_protocol_{stim}.pdf'),bbox_inches='tight')
            plt.close()
            
        self.set_data_exp(f'processed_{data_name}',analyzed_stim_dict)
    
    
    def _calculate_prediction_error(self,
                                   ground_thruth: np.ndarray, # experimental psth
                                   average_prediction:np.ndarray, # average psth prediction
                                   channel_integral:np.ndarray,
                                   pred_integral:np.ndarray,
                                   metric_to_use:callable = metric.squared_error,
                                   plot:bool = False, # whether to plot or not the xrmse vs tau plot
                                   path_to_save:str = None,
                                   channel_list:list=None):

            ground_thruth = ground_thruth + 1e-16 # a very small values is added to avoid problem with zero division
            average_prediction = average_prediction + 1e-16

            ch_wise_error,ch_wise_tau,overall_error,overall_tau = metric.cross_metrics(ground_thruth,average_prediction,channel_integral=channel_integral,pred_integral=pred_integral,metric=metric_to_use,plot=plot,path_to_save=path_to_save,int_time=self.retrieve_param_exp('int_time'),channel_list=channel_list)

            overall_tau_in_ms = overall_tau * self.retrieve_param_exp('int_time') # overall lag in ms

            return ch_wise_error,ch_wise_tau,overall_error,overall_tau_in_ms
    


    def _roc_curve(self,y,pred,name,path_to_save,plot=True):
        plt.close()
        fpr, tpr, _ = roc_curve(y, pred)
        roc_auc = auc(fpr, tpr)

        if plot:
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                estimator_name=name)
            display.plot()
            plt.legend(fontsize=12)

            plt.savefig(path_to_save,bbox_inches='tight')
            plt.close()

        return roc_auc

    def connectivity_matrix_analysis(self,
                                    mearc_dicto:dict, # dictionary containing the meaRC model for which CM correlation should be calculated  
                                    exp_type:str = 'sim'
                                    ):
        
        '''
        The function calculates the AUC/pearson correlation coefficient in case of simulation
        '''
        # create folder to store cm data
        cm_folder_path = os.path.join(str(self.exp.results_path),'cm_analysis')
        os.makedirs(cm_folder_path)

        cm_results={}

        if exp_type == 'sim':
            
            
        
            sim_matrix = pd.read_csv(os.path.join(self.exp.exp_path,'data/connectivity_matrices/weighted_conn_matrix.csv'),sep='\t')
                        
            sim_matrix = sim_matrix.iloc[:,1:].to_numpy() # we remove the first column which contains only the id of the electrodes 
        
            sim_vector = sim_matrix.flatten()
        
            sim_vector[np.isnan(sim_vector)]=0



        for m in mearc_dicto:

            cm_results[m]={}

            for alpha in mearc_dicto[m]:
                cm_results[m][alpha] = {}
                tmp_rep = []

                for rep in mearc_dicto[m][alpha]:

                    mea_model = mearc_dicto[m][alpha][rep]
                    cm_model = mea_model.results['cm'] # retrieve the cm computed by the model

                    tmp_rep.append(cm_model)
                
                tmp_rep = np.stack(tmp_rep)

                avg_cm = np.mean(tmp_rep,axis=0) # cm average across different repetitions of the model parametrized with same alpha and m 
                std_cm = np.std(tmp_rep,axis=0)

                cm_confidence = 1 - (np.max(std_cm)/np.abs(np.max(avg_cm)))

                if exp_type == 'sim':
                    avg_cm_flatten = avg_cm.flatten()
                    avg_cm_flatten[np.isnan(avg_cm_flatten)]=0
                    
                   
                    corr_coefficient = pearsonr(sim_vector,avg_cm_flatten)[0]
                    
                    binarized_sim_vector = np.where(np.abs(sim_vector)>0,1,0)
                    
                    
                    auc = self._roc_curve(binarized_sim_vector,np.abs(avg_cm_flatten),f'm={m},{chr(945)}={alpha}',os.path.join(cm_folder_path,f'post_cm_roc_{m}_{alpha}.pdf'))
                    
                
                cm_results[m][alpha]['avg_cm']=avg_cm
                cm_results[m][alpha]['std_cm']=std_cm
                cm_results[m][alpha]['cm_confidence']=cm_confidence
                
                
                if exp_type == 'sim':
                    
                    cm_results[m][alpha]['cc'] = corr_coefficient
                    cm_results[m][alpha]['auc'] = auc
        
        
        if exp_type == 'sim':
        
            alphas = []
            ms=[]
            cc_val = []
            auc_val=[]
            conf_val= []

            
            for m in sorted(list(cm_results.keys())):
            
                for alpha in sorted(list(cm_results[m].keys())):
                    alphas.append(alpha)
                    ms.append(m)
                    cc_value = cm_results[m][alpha]['cc']
                    auc_value = cm_results[m][alpha]['auc']
                    conf_value = cm_results[m][alpha]['cm_confidence']
                
                    cc_val.append(cc_value)
                    auc_val.append(auc_value)
                    conf_val.append(conf_value)
            # saves dataframe  
            tmp_to_save = pd.DataFrame({'m':ms,'alpha':alphas,'pearson':cc_val,'AUC':auc_val,'confidence':conf_val})
            tmp_to_save.to_csv(os.path.join(cm_folder_path,'cm_analysis.csv'),sep='\t')
            
            # saves images
            #save pearson
            plt.close()
            tmp_pearson = tmp_to_save.pivot(index="m",columns='alpha',values='pearson')
            sns.heatmap(tmp_pearson,cmap='coolwarm')
            plt.xlabel(chr(945))
            plt.title('$\\rho (\\mathcal{T}_0,\\mathcal{T}_{GT})$')
            plt.savefig(os.path.join(cm_folder_path,'pearson.pdf'),bbox_inches='tight')
            plt.close()
            #save AUC
            tmp_auc = tmp_to_save.pivot(index="m",columns='alpha',values='AUC')
            sns.heatmap(tmp_auc,cmap='coolwarm')
            plt.xlabel(chr(945))
            plt.title('CM ROC AUC')
            plt.savefig(os.path.join(cm_folder_path,'AUC.pdf'),bbox_inches='tight')
            plt.close()
            # save confidence 
            tmp_conf = tmp_to_save.pivot(index="m",columns='alpha',values='confidence')
            sns.heatmap(tmp_conf,cmap='coolwarm')
            plt.xlabel(chr(945))
            plt.title('$\\Gamma_{CM}$')
            plt.savefig(os.path.join(cm_folder_path,'confidence.pdf'),bbox_inches='tight')
            plt.close()
                        

        save_pickle_file(cm_results,os.path.join(cm_folder_path,'cm_analysis_results.pickle'))
        
        return cm_results

    
    def stimulate_mearc(self,
                        mearc_dict:dict, # dictionary containing meaRC trained model(s)
                        analyzed_stim_dict:dict, # dictionary containing the processed stimulation data (PSTH data)
                        steps_number:int=100, # number of steps for which the model is stimulated
                        optimize_intensity:bool = True
                        ):
        

        predicted_stimulation = {}

        for stim in analyzed_stim_dict:

            if stim not in predicted_stimulation:
                predicted_stimulation[stim]={}
            
            n_ch = analyzed_stim_dict[stim]['psth'].shape[0] # channel number
            elec_idx = analyzed_stim_dict[stim]['elec_idx'] 
            
            for m in mearc_dict: # for each m parameter

                if m not in predicted_stimulation[stim]:
                    predicted_stimulation[stim][m]={}

                for alpha in mearc_dict[m]: # for each alpha parameter

                    if alpha not in predicted_stimulation[stim][m]:

                        predicted_stimulation[stim][m][alpha]={}


                    if optimize_intensity == True:

                        if self.exp.exp_type == 'sim':
                            intensity_to_test= np.concatenate((np.arange(0.1,1,0.1),np.arange(1,5,0.5),np.arange(5,10,1),np.arange(10,25,2.5),np.arange(25,50,5)))
                        else:
                            intensity_to_test= np.concatenate((np.arange(0.1,1,0.1),np.arange(1,5,0.5),np.arange(5,10,1),np.arange(10,25,2.5),np.arange(25,50,5)))
                        #intensity_to_test = np.round(np.arange(0.0,15.1,0.1),1).tolist()
                        #intensity_to_test = np.arange(0,6,0.5)
                    else:
                        intensity_to_test = [analyzed_stim_dict[stim]['norm_factor_stim']/self.retrieve_param_exp('norm_factor_back')]

                    for intensity in intensity_to_test:

                        if intensity not in predicted_stimulation[stim][m][alpha]:
                            predicted_stimulation[stim][m][alpha][intensity]={}
                    
                        stimulation_vector = np.zeros(n_ch); stimulation_vector[elec_idx] = intensity # creation of stimulation vector
                    
                        rep_list = [] # the psth is calculated for each realization of each model parametrization (m,alpha). The final predicted response is the average of the predicted response of each realization

                        for rep in mearc_dict[m][alpha]: # for each repetition for couple (m,alpha)

                            mearc = mearc_dict[m][alpha][rep] # retrieve the mearRC model


                            predicted_psth = mearc.stimulate(stimulation_vector,steps_number) # predicted psth

                            rep_list.append(predicted_psth)
                    
                        rep_list = np.stack(rep_list) # stack of predicted psth

                        average_prediction = np.mean(rep_list,axis=0) # average psth
                        std_prediction = np.std(rep_list,axis=0) # std psth

                    
                        predicted_stimulation[stim][m][alpha][intensity]['mean']=average_prediction
                        predicted_stimulation[stim][m][alpha][intensity]['std'] =std_prediction
            
        self.set_data_exp('predicted_psth',predicted_stimulation)
        
        save_pickle_file(predicted_stimulation,os.path.join(self.exp.results_path,'predicted_stimulation.pickle'))





    def full_characterization_stim(self,
                                predicted_stim:dict, # predicted PSTH from stimulation (output of stimulate_mearc function)
                                processed_stim:dict, # ground truth stimulation data ('processed_filt_stimulation in case of experimental recording or processed_stimulation in case of simulation)
                                gt_threshold:float=0.5,
                                overwrite:bool = False
                                ):
        
        
        exp = self.exp

        int_time = exp.get_param('int_time')

        experiment_type = exp.exp_type

        # create folder to store prediction data of all mearc models
        pred_folder_path = os.path.join(exp.results_path,'stimulus_prediction')

        if overwrite == True:

            if os.path.exists(pred_folder_path) :

                shutil.rmtree(pred_folder_path)
                os.makedirs(pred_folder_path)
            else:
                raise FileExistsError(f'{pred_folder_path} already exists. To overwrite data, set overwrite parameter to True')

        if not os.path.exists(pred_folder_path):
            os.makedirs(pred_folder_path)

    
    

        all_stims_aux_xrmse_csv = {'stim':[],'m':[],'alpha':[],'intensity':[],'xrmse':[],'tau':[],'auc':[],'cc':[]}

        stims_final_auc_xrmse_csv = {'stim':[],'m':[],'alpha':[],'intensity':[],'xrmse':[],'tau':[],'auc':[],'cc':[]}
        
        for stim in predicted_stim:

            #all_channel_pred_stim_m = os.path.join(pred_folder_path,f'prediction_stim_{stim}')
            single_stim_path = os.path.join(pred_folder_path,f'prediction_stim_{stim}')

            if not os.path.exists(single_stim_path):
                os.makedirs(single_stim_path)
            
            stim_channel_int_folder = os.path.join(single_stim_path,'channel_integral_maps')
            if not os.path.exists(stim_channel_int_folder):
                os.makedirs(stim_channel_int_folder)

            #if exp.exp_type =='exp':
            best_channel_int_prediction =None # variable where to store the channel integral of the best prediction based on auc and xrmse
            best_m_channel_int_prediction = None # same but for m
            best_alpha_channel_int_prediction = None # same but for alpha
            best_int_channel_int_prediction = None # same for intensity
            best_overall_channel_int_prediction_auc = - math.inf # used to identify the best results
            best_overall_channel_int_prediction_xrmse = math.inf
            ch_wise_error_int_prediction = None
            best_channel_int_bin_prediction = None # variable where to store the channel integral of the best prediction based on xrmse
            best_cc_int_prediction = None # variable to store cc

            tmp_pred = predicted_stim[stim]

            gt_integral = processed_stim[stim]['integral']
            #norm_factor_stim = processed_stim[stim]['norm_factor_stim']
            norm_factor_back = self.exp.get_param('norm_factor_back')
            ground_truth_stim = processed_stim[stim]['psth']/ norm_factor_back

            max_ground_truth_stim = np.max(ground_truth_stim,axis=1)
            gt_thr = gt_threshold/norm_factor_back

            binarized_gt = np.where(np.sum(np.where(ground_truth_stim>=gt_thr,1,0),axis=1)>=1,1,0)

            if (np.sum(binarized_gt)/len(binarized_gt)) == 0:
                ground_truth_stim = processed_stim[stim]['psth']/1

            
            if exp.exp_type == 'exp':
                stim_elec_list = exp.get_data('filt_stimulation')[stim][2]
                unfiltered_elec_list = exp.get_data('stimulation')[stim][2]
            else:
                stim_elec_list = exp.get_data('stimulation')[stim][2]
                unfiltered_elec_list = '' #!TODO change when filtering of simulation will be added
            
            all_m_ch_prediction_dict = {}

            all_m_dict = {}
            
            plot_mea_map(unfiltered_elec_list,stim_elec_list,gt_integral,stim,'GT integral [a.u.]',os.path.join(stim_channel_int_folder,f'stim_{stim}_gt_integral.pdf'),exp_type=experiment_type)
            plot_mea_map(unfiltered_elec_list,stim_elec_list,binarized_gt,stim,'Binarized GT',os.path.join(stim_channel_int_folder,f'stim_{stim}_binarized_gt.pdf'),bin=True,exp_type=experiment_type)
            plot_mea_map(unfiltered_elec_list,stim_elec_list,max_ground_truth_stim,stim,'GT response [a.u.]',os.path.join(stim_channel_int_folder,f'stim_{stim}_gt_amplitude.pdf'),exp_type=experiment_type)

            m_values = sorted(list(tmp_pred.keys()))

            #heatmap m vs alpha auc and xrmse for best intensity 

            heatmap_auc_xrmse = {'m':[],'alpha':[],'best_intensity':[],'auc':[],'xrmse':[],'cc':[]}
            

            for m in m_values:
                
                m_path = os.path.join(single_stim_path,f'mearc_m_{m}_prediction') # prediction folder of specific m

                all_m_ch_prediction_dict[m] ={}
                all_m_dict[m]={}

                tmp_m_pred = tmp_pred[m]

                alphas_values = sorted(list(tmp_m_pred.keys())) # retrieve all alphas value

                # for merged alpha graphs
                all_alpha_dict = {}
                all_alpha_dict_prediction ={}

                #all_channel_pred_stim_path = os.path.join(m_path,f'prediction_stim_{stim}_m_{m}_all_alpha')
                all_alpha_path = os.path.join(m_path,'all_alpha') # folder where graphs showing all alphas are saved

                os.makedirs(all_alpha_path)


                for alpha_val in alphas_values:

                    all_alpha_dict[alpha_val]={}

                    all_alpha_dict_prediction[alpha_val] ={}

                    alpha_value_specific_path = os.path.join(m_path,f'alpha_{alpha_val}') # path for a specific alpha value

                    all_m_dict[m][alpha_val]={}
                    all_m_ch_prediction_dict[m][alpha_val]={}

                    if not os.path.exists(alpha_value_specific_path):
                        os.makedirs(alpha_value_specific_path)

                    tmp_pred_alpha = tmp_m_pred[alpha_val]

                    intensity_values = sorted(list(tmp_pred_alpha.keys()))

                    overall_err_list = []
                    overall_tau_list = []
                    auc_list=[]
                    cc_list = []


                    for intensity in intensity_values:


                        predicted_psth = predicted_stim[stim][m][alpha_val][intensity]['mean']

                        pred_integral = self._PSTH_integral(predicted_psth)

                        _,_,overall_error,overall_tau = self._calculate_prediction_error(ground_truth_stim,predicted_psth,gt_integral,pred_integral)


                        max_amplitude_psth_roc = np.max(predicted_psth,axis=1)

                       


                        max_amplitude_cc = pearsonr(max_amplitude_psth_roc,max_ground_truth_stim)[0]

                        max_amplitude_psth_roc = np.round(np.where(max_amplitude_psth_roc>1,1,max_amplitude_psth_roc),2)


                        
        
                        binarized_gt_tmp = copy.deepcopy(binarized_gt)
                        binarized_gt_roc,predicted_psth_roc,_ = fix_roc(binarized_gt_tmp,max_amplitude_psth_roc,'_',gt_thr=gt_thr)


                        roc_auc = self._roc_curve(binarized_gt_roc,predicted_psth_roc,'_','_',False)

                        overall_err_list.append(overall_error)
                        overall_tau_list.append(overall_tau)
                        auc_list.append(roc_auc)
                        cc_list.append(max_amplitude_cc)


                        # save everything
                        
                        all_stims_aux_xrmse_csv['stim'].append(stim),all_stims_aux_xrmse_csv['m'].append(m),all_stims_aux_xrmse_csv['alpha'].append(alpha_val),all_stims_aux_xrmse_csv['cc'].append(max_amplitude_cc)
                        all_stims_aux_xrmse_csv['intensity'].append(intensity),all_stims_aux_xrmse_csv['xrmse'].append(overall_error),all_stims_aux_xrmse_csv['tau'].append(overall_tau),all_stims_aux_xrmse_csv['auc'].append(roc_auc)



                    all_alpha_dict[alpha_val]['xrmse']=overall_err_list
                    all_alpha_dict[alpha_val]['auc']=auc_list


                    all_m_dict[m][alpha_val]['xrmse']=overall_err_list
                    all_m_dict[m][alpha_val]['auc']=auc_list



                    #overall_error vs intensity
                    plt.close()
                    
                    plt.plot(intensity_values,overall_err_list)
                    
                    plt.xlabel('Intensity [a.u.]')
                    plt.ylabel('$\\overline{R}$')
                    
                    plt.savefig(os.path.join(alpha_value_specific_path,'xrmse_vs_intensity.pdf'),bbox_inches='tight')
                    plt.close()


                    #pearson vs intensity
                    plt.close()
                    
                    plt.plot(intensity_values,cc_list)
                    
                    plt.xlabel('Intensity [a.u.]')
                    plt.ylabel('$\\rho$')
                    
                    plt.savefig(os.path.join(alpha_value_specific_path,'pearson_vs_intensity.pdf'),bbox_inches='tight')
                    plt.close()


                    #overall_tau vs intensity
                    plt.close()
                    
                    plt.plot(intensity_values,overall_tau_list)
                    
                    plt.xlabel('Intensity [a.u.]')
                    plt.ylabel('$\\overline{\\tau}_l [ms]$')
                    
                    plt.savefig(os.path.join(alpha_value_specific_path,f'{chr(964)}_vs_intensity.pdf'),bbox_inches='tight')
                    plt.close()

                    #auc vs intensity
                    plt.close()
                    
                    plt.plot(intensity_values,auc_list)
                    
                    plt.xlabel('Intensity [a.u.]')
                    plt.ylabel(f'AUC')
                    
                    plt.savefig(os.path.join(alpha_value_specific_path,f'Response_auc_vs_intensity.pdf'),bbox_inches='tight')
                    plt.close()


                    # best intensity using AUC and xrmse

                    best_intensity = find_best_intensity_auc_xrmse(cc_list,overall_err_list,intensity_values)

                    best_intensity_psth = predicted_stim[stim][m][alpha_val][best_intensity]['mean']
                    best_intensity_psth_std = predicted_stim[stim][m][alpha_val][best_intensity]['std']
                    best_intensity_integral = self._PSTH_integral(best_intensity_psth)
                    

                    tau_path = os.path.join(alpha_value_specific_path,'channels_tau')
                    if not os.path.exists(tau_path):
                        os.makedirs(tau_path)

                    ch_wise_error,_,overall_error,overall_tau = self._calculate_prediction_error(ground_truth_stim,best_intensity_psth,gt_integral,best_intensity_integral,plot=True,path_to_save=tau_path,channel_list=stim_elec_list)                
                    

                    
                    tmp_best_intensity_path = os.path.join(pred_folder_path,'best_intensity_psth')
                    if not os.path.exists(tmp_best_intensity_path):
                        os.makedirs(tmp_best_intensity_path)


                    plt.close()
                    sns.heatmap(best_intensity_psth,cmap='coolwarm',xticklabels=np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time)
                    plt.title(f'Normalized ISR - Prediction [a.u.], Time bin = {int_time} ms')
                    plt.xlabel('Time bin [ms]')
                    plt.ylabel('Channel number')
                    plt.savefig(os.path.join(tmp_best_intensity_path,f'psth_stim_protocol_{stim}_m_{m}_{chr(945)}_{alpha_val}_intensity_{best_intensity}.pdf'),bbox_inches='tight')
                    plt.close()

                    all_alpha_dict_prediction[alpha_val]={'best_intensity_psth':best_intensity_psth,'best_intensity_psth_std':best_intensity_psth_std,'ch_wise_err':ch_wise_error,'intensity':best_intensity}


                    all_m_ch_prediction_dict[m][alpha_val]={'best_intensity_psth':best_intensity_psth,'best_intensity_psth_std':best_intensity_psth_std,'ch_wise_err':ch_wise_error,'intensity':best_intensity}


                    channel_prediction_folder = os.path.join(alpha_value_specific_path,'channel_predictions')
                    os.makedirs(channel_prediction_folder)
                    
                    for ch in range(best_intensity_psth.shape[0]):
                        
                        ch_id = stim_elec_list[ch]
                        plt.close()
                        
                        plt.errorbar(np.round(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,1),best_intensity_psth[ch,:],yerr=best_intensity_psth_std[ch,:],label = 'Prediction',color='red',marker='o',linestyle='--',linewidth=0.5)
                        plt.plot(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,ground_truth_stim[ch,:],label='Ground truth',color='blue',marker='o',linestyle='--',linewidth=0.5)
                        plt.title(f'Channel {ch_id} response ISR, Time bin = {int_time} ms')
                        plt.xlabel(f'Time [ms]')
                        plt.ylabel('ISR [a.u.]')
                        plt.legend(bbox_to_anchor=(1.25, 1),loc='center left')
                        plt.savefig(os.path.join(channel_prediction_folder,f'channel_{ch_id}_prediction.pdf'),bbox_inches='tight')
                        plt.close()

                    # response roc
                    max_amplitude_psth = np.max(best_intensity_psth,axis=1)

                    max_amplitude_psth_to_save = copy.deepcopy(max_amplitude_psth)

                    max_amplitude_cc = pearsonr(max_amplitude_psth,max_ground_truth_stim)[0]

                    max_amplitude_psth = np.round(np.where(max_amplitude_psth>1,1,max_amplitude_psth),2)

                    
                    binarized_gt_tmp_2 = copy.deepcopy(binarized_gt)

                    binarized_gt_roc_2,max_amplitude_psth_roc_2,roc_curve_path = fix_roc(binarized_gt_tmp_2,max_amplitude_psth,alpha_value_specific_path,gt_thr=gt_thr)
                    
                    auc = self._roc_curve(binarized_gt_roc_2,max_amplitude_psth_roc_2,f'm={m},{chr(945)}={alpha_val},int={best_intensity}',roc_curve_path)

                    
                    heatmap_auc_xrmse['m'].append(m),heatmap_auc_xrmse['alpha'].append(alpha_val),heatmap_auc_xrmse['best_intensity'].append(best_intensity),heatmap_auc_xrmse['auc'].append(auc),heatmap_auc_xrmse['xrmse'].append(overall_error),heatmap_auc_xrmse['cc'].append(max_amplitude_cc)

                    

                    stims_final_auc_xrmse_csv['stim'].append(stim),stims_final_auc_xrmse_csv['m'].append(m),stims_final_auc_xrmse_csv['alpha'].append(alpha_val),stims_final_auc_xrmse_csv['cc'].append(max_amplitude_cc)
                    stims_final_auc_xrmse_csv['intensity'].append(best_intensity),stims_final_auc_xrmse_csv['xrmse'].append(overall_error),stims_final_auc_xrmse_csv['auc'].append(auc),stims_final_auc_xrmse_csv['tau'].append(overall_tau)

                    
                    fpr, tpr, ths = roc_curve(binarized_gt_roc_2, max_amplitude_psth_roc_2)

                    
                    selected_ths = find_best_threshold_response(fpr,tpr,ths)
                    
                    
            
                    binarized_max_amplitude_psth = np.where(max_amplitude_psth>=selected_ths,1,0)


                    if overall_error < best_overall_channel_int_prediction_xrmse:
                        #if auc >= best_overall_channel_int_prediction_auc:
                        

                            best_overall_channel_int_prediction_auc = auc
                            best_overall_channel_int_prediction_xrmse = overall_error
                            best_channel_int_prediction =best_intensity_integral # variable where to store the channel integral of the best prediction based on xrmse
                            best_m_channel_int_prediction = m # same but for m
                            best_alpha_channel_int_prediction = alpha_val # same but for alpha
                            best_int_channel_int_prediction = best_intensity # same for intensity
                            ch_wise_error_int_prediction = ch_wise_error
                            best_channel_int_bin_prediction = binarized_max_amplitude_psth # variable where to store the channel integral of the best prediction based on xrmse
                            best_cc_int_prediction = max_amplitude_psth_to_save
            

                #save best alpha for response
                best_alpha_for_protocol_xrmse = math.inf
                alpha_of_best = None
                
                # all alphas plot xrmse vs intensity
                plt.close()
                
                color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']
                for num,alpha_val in enumerate(sorted(list(all_alpha_dict.keys()))):

                    #overall_error vs intensity
                    plt.plot(intensity_values,all_alpha_dict[alpha_val]['xrmse'],label=f'{chr(945)}={alpha_val}',color=color_list[num])

                    best_xrmse_for_alpha = np.min(all_alpha_dict[alpha_val]['xrmse'])
                    if best_xrmse_for_alpha<best_alpha_for_protocol_xrmse:
                        best_alpha_for_protocol_xrmse = best_xrmse_for_alpha
                        alpha_of_best = alpha_val

                plt.xlabel('Intensity [a.u.]')
                plt.ylabel('$\\overline{R}$')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(os.path.join(all_alpha_path,f'xrmse_vs_intensity_stim_{stim}_m_{m}_all_alphas.pdf'),bbox_inches='tight')
                plt.close()

                
                
                # all alphas plot auc vs intensity
                plt.close()
                
                color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']
                for num,alpha_val in enumerate(sorted(list(all_alpha_dict.keys()))):

                    #overall_error vs intensity
                    plt.plot(intensity_values,all_alpha_dict[alpha_val]['auc'],label=f'{chr(945)}={alpha_val}',color=color_list[num])


                
                plt.xlabel('Intensity [a.u.]')
                plt.ylabel('AUC')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(os.path.join(all_alpha_path,f'auc_vs_intensity_stim_{stim}_m_{m}_all_alphas.pdf'),bbox_inches='tight')
                plt.close()

                #alpha all channel prediction
                
                tmp_rearranged ={}
                for alpha in sorted(all_alpha_dict_prediction.keys()):
                        psth = all_alpha_dict_prediction[alpha]['best_intensity_psth']
                        psth_std = all_alpha_dict_prediction[alpha]['best_intensity_psth_std']
                        ch_err = all_alpha_dict_prediction[alpha]['ch_wise_err']
                        best_intensity = all_alpha_dict_prediction[alpha]['intensity']


                        for ch in range(len(ch_err)):
                            if ch not in tmp_rearranged: 
                                tmp_rearranged[ch] = {'pred':[psth[ch]],'pred_std':[psth_std[ch]],'ch_err':[ch_err[ch]],'int':[best_intensity]}
                            else:
                                tmp_rearranged[ch]['pred'].append(psth[ch])
                                tmp_rearranged[ch]['ch_err'].append(ch_err[ch])
                                tmp_rearranged[ch]['int'].append(best_intensity)
                                tmp_rearranged[ch]['pred_std'].append(psth_std[ch])

                
                for ch in sorted(tmp_rearranged.keys()):
                    plt.close()
                    all_pred_alpha_ch = tmp_rearranged[ch]['pred']
                    ch_err_alpha_ch = tmp_rearranged[ch]['ch_err']
                    int_alpha_ch = tmp_rearranged[ch]['int']
                    pred_std_alpha_ch = tmp_rearranged[ch]['pred_std']
                    for alpha_ch in range(len(all_pred_alpha_ch)):
                        
                        plt.errorbar(np.array(list(range(1,len(all_pred_alpha_ch[alpha_ch])+1,1)))*int_time,all_pred_alpha_ch[alpha_ch],yerr=pred_std_alpha_ch[alpha_ch],label = f'Prediction: {chr(945)}={alphas_values[alpha_ch]},intensity={round(int_alpha_ch[alpha_ch],1)},{chr(949)}={round(ch_err_alpha_ch[alpha_ch],3)} ',color=color_list[alpha_ch],marker='o',linestyle='--',linewidth=0.5)
                    
                    plt.plot(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,ground_truth_stim[ch,:],label='Ground truth',color='black',marker='o',linestyle='--',linewidth=0.5)
                    
                    plt.xlabel(f'Time (ms)')
                    plt.ylabel('ISR [a.u.]')
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.savefig(os.path.join(all_alpha_path,f'channel_{stim_elec_list[ch]}_m_{m}_all_alpha_prediction.pdf'),bbox_inches='tight')
                    plt.close()
            
            
            plot_mea_map(unfiltered_elec_list,stim_elec_list,best_channel_int_bin_prediction,stim,'Binarized response [a.u.]',os.path.join(stim_channel_int_folder,f'stim_{stim}_binarized_prediction.pdf'),bin=True,exp_type=experiment_type)
            plot_mea_map(unfiltered_elec_list,stim_elec_list,best_channel_int_prediction,stim,'Response integral [a.u.]',os.path.join(stim_channel_int_folder,f'stim_{stim}_prediction_integral.pdf'),exp_type=experiment_type)
                            
            plot_mea_map(unfiltered_elec_list,stim_elec_list,ch_wise_error_int_prediction,stim,'$\\varepsilon$',os.path.join(stim_channel_int_folder,f'stim_{stim}_ch_wise_err_prediction.pdf'),exp_type=experiment_type)
            
            error_binarized_gt_vs_pred = np.abs((best_channel_int_bin_prediction-binarized_gt)) 

            plot_mea_map(unfiltered_elec_list,stim_elec_list,error_binarized_gt_vs_pred,stim,'',os.path.join(stim_channel_int_folder,f'stim_{stim}_binarized_prediction_error.pdf'),bin_err=True,exp_type=experiment_type)

            plot_mea_map(unfiltered_elec_list,stim_elec_list,max_amplitude_psth_to_save,stim,'Predicted response [a.u]',os.path.join(stim_channel_int_folder,f'stim_{stim}_amplitude_prediction.pdf'),exp_type=experiment_type)

            selected_m_alpha_best_int = pd.DataFrame({'m':[best_m_channel_int_prediction],'alpha':[best_alpha_channel_int_prediction],'int':[best_int_channel_int_prediction]})

            selected_m_alpha_best_int.to_csv(os.path.join(stim_channel_int_folder,'channel_map_selected_parameters.csv'),sep='\t')
            #heatmaps

            
            heatmap_auc_xrmse = pd.DataFrame(heatmap_auc_xrmse)
            

            plt.close()
            sns.heatmap(heatmap_auc_xrmse.pivot(index="m", columns="alpha", values="xrmse"),cmap='coolwarm')
            plt.xlabel(f'{chr(945)}')
            plt.title('$\\overline{R}$')
            plt.savefig(os.path.join(single_stim_path,'m_alpha_xrmse_value.pdf'),bbox_inches='tight')
            
            plt.close()


            sns.heatmap(heatmap_auc_xrmse.pivot(index="m", columns="alpha", values="auc"),cmap='coolwarm')
            plt.xlabel(f'{chr(945)}')
            plt.title('Response Prediction ROC AUC')
            plt.savefig(os.path.join(single_stim_path,'m_alpha_auc_value.pdf'),bbox_inches='tight')
            plt.close()

            sns.heatmap(heatmap_auc_xrmse.pivot(index="m", columns="alpha", values="best_intensity"),cmap='coolwarm')
            plt.xlabel(f'{chr(945)}')
            plt.title('Optimal intensity [a.u.]')
            plt.savefig(os.path.join(single_stim_path,'m_alpha_best_intensity.pdf'),bbox_inches='tight')

            plt.close()
            sns.heatmap(heatmap_auc_xrmse.pivot(index="m", columns="alpha", values="cc"),cmap='coolwarm')
            plt.xlabel(f'{chr(945)}')
            plt.title('$\\rho$')
            plt.savefig(os.path.join(single_stim_path,'m_alpha_cc_value.pdf'),bbox_inches='tight')
            plt.close()
            


            # all m plot xrmse vs intensity
            
            tmp_all_m = {}


            for m_val in sorted(list(all_m_dict.keys())):

                for alpha_val in sorted(list(all_m_dict[m_val].keys())):

                    if alpha_val not in tmp_all_m:
                        tmp_all_m[alpha_val] = {}

                    if m_val not in tmp_all_m[alpha_val]:
                        tmp_all_m[alpha_val][m_val]={'xrmse':all_m_dict[m_val][alpha_val]['xrmse'],'auc':all_m_dict[m_val][alpha_val]['auc']}
                    




                        
            plt.close()
            
            color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']

            for alpha_val in sorted(list(tmp_all_m.keys())):

                for num,m_val in enumerate(sorted(list(tmp_all_m[alpha_val].keys()))):

                    #overall_error vs intensity
                    plt.plot(intensity_values,tmp_all_m[alpha_val][m_val]['xrmse'],label=f'm={m_val}',color=color_list[num])


                
                plt.xlabel('Intensity [a.u.]')
                plt.ylabel('$\\overline{R}$')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(os.path.join(single_stim_path,f'xrmse_vs_intensity_stim_{stim}_all_m_{chr(945)}_{alpha_val}.pdf'),bbox_inches='tight')
                plt.close()


            # all m plot auc vs intensity
            plt.close()
            
            color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']

            for alpha_val in sorted(list(tmp_all_m.keys())):

                for num,m_val in enumerate(sorted(list(tmp_all_m[alpha_val].keys()))):

                    #overall_error vs intensity
                    plt.plot(intensity_values,tmp_all_m[alpha_val][m_val]['auc'],label=f'm={m_val}',color=color_list[num])


                
                plt.xlabel('Intensity [a.u.]')
                plt.ylabel('AUC')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(os.path.join(single_stim_path,f'AUC_vs_intensity_stim_{stim}_all_m_{chr(945)}_{alpha_val}.pdf'),bbox_inches='tight')
                plt.close()
            


            #m all channel prediction
            color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']
            tmp_rearranged ={}

            all_m_path = os.path.join(single_stim_path,'all_m')
            os.makedirs(all_m_path)


            all_m_ch_pred_tmp={}

            for m in sorted(list(all_m_ch_prediction_dict.keys())):
                    
                    for alpha in sorted(list(all_m_ch_prediction_dict[m].keys())):
                        
                        if alpha not in all_m_ch_pred_tmp:
                            all_m_ch_pred_tmp[alpha]={}

                        psth = all_m_ch_prediction_dict[m][alpha]['best_intensity_psth']
                        psth_std = all_m_ch_prediction_dict[m][alpha]['best_intensity_psth_std']
                        ch_err = all_m_ch_prediction_dict[m][alpha]['ch_wise_err']
                        best_intensity = all_m_ch_prediction_dict[m][alpha]['intensity']


                        for ch in range(len(ch_err)):
                            if ch not in all_m_ch_pred_tmp[alpha]: 
                                all_m_ch_pred_tmp[alpha][ch] = {'pred':[psth[ch]],'pred_std':[psth_std[ch]],'ch_err':[ch_err[ch]],'int':[best_intensity]}
                            else:
                                all_m_ch_pred_tmp[alpha][ch]['pred'].append(psth[ch])
                                all_m_ch_pred_tmp[alpha][ch]['ch_err'].append(ch_err[ch])
                                all_m_ch_pred_tmp[alpha][ch]['int'].append(best_intensity)
                                all_m_ch_pred_tmp[alpha][ch]['pred_std'].append(psth_std[ch])

            
            for alpha in sorted(list(all_m_ch_pred_tmp.keys())):
                
                tmp_alpha_path = os.path.join(all_m_path,f'alpha_{alpha}')
                if not os.path.exists(tmp_alpha_path):
                    os.makedirs(tmp_alpha_path)

                for ch in sorted(list(all_m_ch_pred_tmp[alpha].keys())):
                    plt.close()
                    all_pred_m_ch = all_m_ch_pred_tmp[alpha][ch]['pred']
                    ch_err_m_ch = all_m_ch_pred_tmp[alpha][ch]['ch_err']
                    int_m_ch = all_m_ch_pred_tmp[alpha][ch]['int']
                    pred_std_m_ch = all_m_ch_pred_tmp[alpha][ch]['pred_std']
                    for m_ch in range(len(all_pred_m_ch)):
                        
                        plt.errorbar(np.round(np.array(list(range(1,len(all_pred_m_ch[m_ch])+1,1)))*int_time,1),all_pred_m_ch[m_ch],yerr=pred_std_m_ch[m_ch],label = f'Prediction: m={m_values[m_ch]},intensity={round(int_m_ch[m_ch],1)},{chr(949)}={round(ch_err_m_ch[m_ch],3)} ',color=color_list[m_ch],marker='o',linestyle='--',linewidth=0.5)
                    
                    plt.plot(np.round(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,1),ground_truth_stim[ch,:],label='Ground truth',color='black',marker='o',linestyle='--',linewidth=0.5)
                    
                    
                    plt.xlabel(f'Time [ms]')
                    plt.ylabel('ISR [a.u.]')
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.savefig(os.path.join(tmp_alpha_path,f'channel_{stim_elec_list[ch]}_all_m_{chr(945)}_{alpha}_prediction.pdf'),bbox_inches='tight')
                    plt.close()
        
            
        #save all stimulation data
        stims_final_auc_xrmse_csv = pd.DataFrame(stims_final_auc_xrmse_csv)
        stims_final_auc_xrmse_csv.to_csv(os.path.join(pred_folder_path,'stimulus_prediction_best_intensity_data_auc_xrmse.csv'),sep='\t')
        
        all_stims_aux_xrmse_csv = pd.DataFrame(all_stims_aux_xrmse_csv)
        all_stims_aux_xrmse_csv.to_csv(os.path.join(pred_folder_path,'stimulus_prediction_all_data_auc_xrmse.csv'),sep='\t')


    #! IMPLEMENTED BUT NOT USED
    #! those functions were implemented thinking we could use the validation set to find the best set of m,alpha parameters
    #! but we switched to full characterization to analyze better the behavior of the model.




    def prediction_error(self,
                         predicted_stimulation:dict,
                         processed_stim:dict):
        
        # create folder to store prediction  data
        pred_folder_path = os.path.join(self.exp.results_path,'stimulus_prediction')
        if not os.path.exists(pred_folder_path):
            os.makedirs(pred_folder_path)

        prediction_error_results = {}


        for stim in processed_stim: # for each stimulation protocol
            
            prediction_error_results[stim]={}
            ground_truth = processed_stim[stim]['psth']/processed_stim[stim]['norm_factor_stim'] # normalized experimental psth
            channel_integral = processed_stim[stim]['integral']

            
            m_list = []
            alpha_list = []
            intensity_list = []
            intensity_err_list = []
            intensity_tau_list = []
            for m in predicted_stimulation[stim]:
                
                

                prediction_error_results[stim][m]={}
                
                for alpha in predicted_stimulation[stim][m]:
                    
                    m_list.append(m)
                    alpha_list.append(alpha)
                    

                    prediction_error_results[stim][m][alpha]={}

                    intensity_value = 0 # pointer to the intensity giving smallest error on prediction
                    intensity_err = math.inf # value of the error
                    intensity_tau =0

                    for intensity in predicted_stimulation[stim][m][alpha]:

                        prediction_error_results[stim][m][alpha][intensity]={}

                        predicted_psth = predicted_stimulation[stim][m][alpha][intensity]['mean']

                        pred_integral = self._PSTH_integral(predicted_psth)

                        _,_,overall_error,overall_tau = self._calculate_prediction_error(ground_truth,predicted_psth,channel_integral,pred_integral)                
                        
                        if overall_error < intensity_err:
                            intensity_err= overall_error
                            intensity_value = intensity
                            intensity_tau = overall_tau

                    intensity_list.append(intensity_value)
                    intensity_err_list.append(intensity_err)
                    intensity_tau_list.append(intensity_tau)

                    #prediction_error_results[stim][m][alpha][intensity_value]['ch_wise_error']=ch_wise_error
                    #prediction_error_results[stim][m][alpha][intensity_value]['ch_wise_tau']=ch_wise_tau
                    prediction_error_results[stim][m][alpha][intensity_value]['overall_error']=intensity_err
                    prediction_error_results[stim][m][alpha][intensity_value]['overall_tau']=intensity_tau

           

            tmp_to_save = pd.DataFrame({'m':m_list,'alpha':alpha_list,'intensity':intensity_list,'overall_err':intensity_err_list,'overall_tau (ms)':intensity_tau_list})
            tmp_to_save.to_csv(os.path.join(pred_folder_path,f'stim_{stim}_prediction_analysis.csv'),sep='\t')
            plt.close()
            tmp_to_save = tmp_to_save.pivot(index="m",columns='alpha',values='overall_err')
            sns.heatmap(tmp_to_save,cmap='coolwarm')
            plt.title('overall_error - XRMSE')
            plt.xlabel(chr(945))
            plt.savefig(os.path.join(pred_folder_path,f'stim_{stim}_prediction_results.pdf'))
            plt.close()
            
        
        #save_pickle_file(prediction_error_results,os.path.join(self.exp.results_path,'prediction_error_analysis.pickle'))
        #save_yaml_file(prediction_error_results,os.path.join(self.exp.results_path,'prediction_error_analysis.yaml'))


    def _find_best_mearc(self,
                        mearc_dicto:dict, # dictionary of meaRC model obtained by running cheracterize_meaRC
                        metric: str = 'rmse' #select the metric used to select the best model
                        ):
        '''
        The function finds the best meaRC model among the trained one by comparing the results on validation set.
        
        Input:
            - mearc_dicto: dictionary obtained as output from characterize_mearc
        '''

        final_res={}

        best = {}
        
        for m in mearc_dicto: # for each m parameter
            
            if m not in final_res:
                final_res[m]={}

            for alpha in mearc_dicto[m]: # for each alpha parameter
                
                if alpha not in final_res[m]:
                    final_res[m][alpha]= {}
                
                tmp_rep_res={} # temporary dictionary used to store the results of the repetition

                for rep in mearc_dicto[m][alpha]: # for each repetition

                    mearc_model = mearc_dicto[m][alpha][rep] # retrieve corresponding meaRC model

                    mearc_model_validation = mearc_model.results['validation']

                    for metric in mearc_model_validation: # for each metric used during the validation

                        if metric not in tmp_rep_res:
                            tmp_rep_res[metric] = [mearc_model_validation[metric]] 
                        else:
                            tmp_rep_res[metric].append(mearc_model_validation[metric])
                        
                        if metric not in best:
                            best[metric]=[math.inf]
                
                for metric in tmp_rep_res: # for each metric

                    avg_metric = np.mean(tmp_rep_res[metric]) # average across repetition
                    std_metric = np.std(tmp_rep_res[metric]) # std 

                    final_res[m][alpha][metric] = {'avg':avg_metric,'std':std_metric}

        alphas = []
        ms=[]
        avg_val = []
        std_val=[]

        first_m = list(final_res.keys())[0]
        alpha_list =list(final_res[first_m].keys())

        alpha_list = sorted(alpha_list)
        for m in final_res:
           
            for alpha in alpha_list:
                alphas.append(alpha)
                ms.append(m)
                for metric in final_res[m][alpha]:
                    
                    avg_value = final_res[m][alpha][metric]['avg']
                    std_value = final_res[m][alpha][metric]['std']
                
                    avg_val.append(avg_value)
                    std_val.append(std_value)
            
        tmp_to_save = pd.DataFrame({'m':ms,'alpha':alphas,f'avg_{metric}':avg_val,f'std_{metric}':std_val})
        tmp_to_save.to_csv(os.path.join(self.exp.results_path,'mearc_validation_full.csv'),sep='\t')
        plt.close()
        tmp_to_save = tmp_to_save.pivot(index="m",columns='alpha',values=f'avg_{metric}')
        fig,ax = plt.subplots(figsize=(5.7,4.7))
        sns.heatmap(tmp_to_save,cmap='coolwarm')
        
        plt.xlabel(chr(945))
        plt.title('Validation loss')
        plt.savefig(os.path.join(self.exp.results_path,f'{metric}_validation_results.pdf'),bbox_inches='tight')
        plt.close()
        
        for m in final_res:

            for alpha in final_res[m]:

                for metric in final_res[m][alpha]:

                    if final_res[m][alpha][metric]['avg']< best[metric][0]:
                        best[metric] = [final_res[m][alpha][metric]['avg'],final_res[m][alpha][metric]['std'],m,alpha]
        
        for metric in best:
            print('Best performing meaRC model:\n')
            print(f'Metric: {metric} - avg: {best[metric][0]} std:{best[metric][1]}')
            print(f'\t -m: {best[metric][2]}')
            print(f'\t -alpha: {best[metric][3]}')

        return best
        
    def optimize_mearc(self,
                    training:np.ndarray,
                    validation:np.ndarray,
                    mearc_dicto:dict, # dictionary obtained as result from calling mearc_characterize
                    metric: str = 'rmse', # metric
                    repeat:int=10, # final number of repetition of the meaRC model
                    rng:int = 1,
                    n_jobs: int = 1,
                    save_data:bool = True,
                    **kwargs
                    ):
        

        best_dicto = self._find_best_mearc(mearc_dicto,metric=metric)
    
        optimal_m = best_dicto[metric][2]

        mearc_m_models =  copy.deepcopy(mearc_dicto[optimal_m]) # retrieve all meaRC model with m

        del mearc_dicto # we eliminate all other meaRC models to free some space

        
        already_perfomed_repetition = len(list(mearc_m_models[list(mearc_m_models.keys())[0]].keys())) # we retrieve the number of already computed repetitions of each model


        for alpha in mearc_m_models:

            tmp_rng = rng + already_perfomed_repetition # the seed is adjusted to avoid having the same initialization 
            
            for rep in range(already_perfomed_repetition,repeat,1): # repetition are performed until the number expressed by repeat is reached
                
                meaRC_model = meaRC(training,validation,m=optimal_m,alpha=alpha,rng=tmp_rng,n_jobs=n_jobs,**kwargs)
                
                meaRC_model.run()

                mearc_m_models[alpha][rep] = meaRC_model
                
                tmp_rng+=1 # change seed for obtaining different initialization
        
        mearc_m_models_fin = {optimal_m: mearc_m_models}

        save_pickle_file(mearc_m_models_fin,os.path.join('optimal_mearc.pickle')) # the optimal meaRC model are saved as a pickle file.
        #save_yaml_file(mearc_m_models_fin,os.path.join('optimal_mearc.yaml'))

        alphas = []
        m=[optimal_m]
        avg_val = []
        std_val=[]

        alpha_list =list(mearc_m_models_fin[optimal_m].keys())
        alpha_list = sorted(alpha_list)
        for alpha in alpha_list:
            alphas.append(alpha)
            tmp = []
            for rep in mearc_m_models_fin[optimal_m][alpha]:
                
                mea_model = mearc_m_models_fin[optimal_m][alpha][rep]
                val_res = mea_model.results['validation'][metric]
                tmp.append(val_res)

            tmp_avg = np.mean(tmp)
            tmp_std = np.std(tmp)
            avg_val.append(tmp_avg)
            std_val.append(tmp_std)


        tmp_to_save = pd.DataFrame({'m':m*len(alphas),'alpha':alphas,f'avg_{metric}':avg_val,f'std_{metric}':std_val})
        tmp_to_save.to_csv(os.path.join(self.exp.results_path,'mearc_validation_optimal.csv'),sep='\t')
        #save_yaml_file(tmp_to_save,os.path.join(self.exp.results_path,'mearc_validation.yaml'))

        if save_data == True:
            save_pickle_file(mearc_m_models_fin,os.path.join(self.exp.results_path,'mea_optimal.pickle'))

        return mearc_m_models_fin

