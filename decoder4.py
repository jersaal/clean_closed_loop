import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = "1"

import cProfile
import pstats
import warnings
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.stats.distributions import chi2
from scipy.signal import hilbert
from scipy import fftpack
from fooof import FOOOF
from scipy.signal import butter, filtfilt, lfilter, remez
from neurodsp.filt import design_fir_filter
from scipy.stats import zscore
import yaml
from array import array
import keyboard
import io
import threading, time

# np.seterr(all='raise')
cp = cProfile.Profile()
cp.enable()


class Decoder():
    def __init__(self):
        self.all_frex_range = (1,31) #all frequencies to be included in background estimation
        self.all_frex = np.arange(self.all_frex_range[0], self.all_frex_range[1])  # frequencys to create filter
        self.hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)), axis=0)[:len(x)]
        self.sendHistoryFull = 0 #changed to 1 after enough data has been collected for 1/f fitting
        self.skipTime = 1 #seconds, how long to skip analysis after detecting noise
        self.numDataSend = 13

        # stuff for plotting..
        self.plotNumWins = 16
        self.plotBurstStarts = []
        self.plotBurstStops = []
        self.burstCounter = 0



    def load_config(self, filename):
        ''' Loads all data from the config file and saves in the instance
        		variables. '''
        with open(filename, 'r') as file:
            try:
                conf = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                pass
        self.seeg_channel = conf['experiment']['channel']
        self.largeWinSize = conf['experiment']['largeWinSize']
        self.filterWinSize = conf['experiment']['filterWinSize']
        self.smollWinSize = conf['experiment']['smollWinSize']
        self.threshWinSize = conf['experiment']['threshWinSize']
        self.sr = conf['experiment']['targetSR']
        self.test_mode = conf['experiment']['testMode']
        # self.actual_sr = 1024
        freq_range = conf['experiment']['freqRange']
        self.padLen = conf['experiment']['padLen']
        self.f_range = np.arange(int(freq_range[0]), int(freq_range[2])+1)
        # self.subj = conf['experiment']['subj']
        self.numCyclesThresh = conf['experiment']['numCyclesThresh'] #num of cycles for burst detection
        self.percentileThresh = conf['experiment']['powerThreshPercentile'] #threshold percentile for burst detection
        self.upperPercentileThresh = conf['experiment']['upperPercentileThresh']
        self.inlet_name = conf['streams']['inlet_name']
        self.outlet_name = conf['streams']['outlet_name']


        self.selected_frex_idx = np.where(np.logical_and(self.all_frex >= self.f_range[0],
                                                         self.all_frex <= self.f_range[-1]))[0]
        self.currBurstLen = np.zeros(len(self.selected_frex_idx))
        self.currBurstActive = np.zeros(len(self.selected_frex_idx), dtype=bool)
        self.currBurstPower = np.zeros(len(self.selected_frex_idx))
        self.currBurstTransfer = np.zeros(len(self.selected_frex_idx), dtype=bool)
        self.currBurstTransferFrex = np.zeros(len(self.selected_frex_idx))
        self.skipWindows = int(self.skipTime / self.smollWinSize)

        self.file_location = "C://closed_loop_results/" + self.subj + "/"
        self.blah = 0

        if not os.path.isdir(self.file_location):
            if not os.path.isdir("C://closed_loop_results/"):
                os.mkdir("C://closed_loop_results/")
            print(self.file_location + " does not exist. Creating...")
            os.mkdir(self.file_location)


        if os.path.exists(self.subj + '_sEEG_noise_baseline.bin'):
            os.remove(self.subj + '_sEEG_noise_baseline.bin')
        if os.path.exists(self.subj + '_sEEG_noise_baseline_ts.bin'):
            os.remove(self.subj + '_sEEG_noise_baseline_ts.bin')
        if os.path.exists(self.subj + '_sEEG.bin'):
            os.remove(self.subj + '_sEEG.bin')
        if os.path.exists(self.subj + '_sEEG_ts.bin'):
            os.remove(self.subj + '_sEEG_ts.bin')
        if os.path.exists(self.subj + '_info_markers.bin'):
            os.remove(self.subj + '_info_markers.bin')
        if os.path.exists(self.subj + '_info_markers_ts.bin'):
            os.remove(self.subj + '_info_markers_ts.bin')
        if os.path.exists(self.subj + '_unity_markers.bin'):
            os.remove(self.subj + '_unity_markers.bin')
        if os.path.exists(self.subj + '_unity_markers_ts.bin'):
            os.remove(self.subj + '_unity_markers_ts.bin')



        self.sEEG_noise_file = open(self.file_location + self.subj + '_sEEG_noise_baseline.bin', 'wb')
        self.sEEG_noise_ts_file = open(self.file_location + self.subj + '_sEEG_noise_baseline_ts.bin', 'wb')
        self.sEEG_file = open(self.file_location + self.subj + '_sEEG.bin', 'wb')
        # self.sEEG_ts_file = open(self.file_location + self.subj + '_sEEG_ts.bin', 'wb')
        self.info_marker_file = open(self.file_location + self.subj + '_info_markers.bin', 'wb')
        self.info_marker_ts_file = open(self.file_location + self.subj + '_info_markers_ts.bin', 'wb')
        self.unity_marker_file = open(self.file_location + self.subj + '_unity_markers.bin', 'wb')
        self.unity_marker_ts_file = open(self.file_location + self.subj + '_unity_markers_ts.bin', 'wb')

        self.unity_marker_file.write(('|').encode())


    #Creates outlet stream and connects to inlet stream
    def connect_streams(self, inlet_name, outlet_name):
        # Let's set up the inlets
        stream_name = outlet_name
        info = StreamInfo(stream_name, 'Marker', self.numDataSend, 0, 'float32', 'myuidw4321')
        self.outlet = StreamOutlet(info)


        print("looking for EEG stream...")
        results = resolve_stream("name", self.inlet_name)
        self.inlet = StreamInlet(results[0])
        print("connected to sEEG stream: " + self.inlet_name)
        self.sr = 1024

    #sends trigger to unity, given stuff to send
    def send_trigger(self, stream, stuff_to_send):
        print('Sending trigger to Unity.')
        self.info_marker_file.write(('Sending trigger to Unity.;').encode())
        float_arr = array('d', [time.time()])
        self.info_marker_ts_file.write(float_arr)
        sendme = np.array(stuff_to_send).astype(float)
        #ADD INITIAL MARKER
        string = (';osc pwr:' + str(sendme[0]) + ';osc len:' + str(sendme[1]) +
                                     ';freq idx:' + str(sendme[2]) + ';actual freq:' +
                                     str(sendme[3]) + ';powthresh:' + str(sendme[4]) + ';upperthresh:' + str(sendme[5]) +
                                     ';meanpower:' + str(sendme[6]) + ';sr:' + str(sendme[7]) + ';history complete:' +
                                     str(sendme[8]) + ';knee:' + str(sendme[9]) + ';exponent:' + str(sendme[10]) +
                                     ';offset:' + str(sendme[11]) + ';burstcounter:' + str(int(sendme[12])) + ';|')

        print("burst: " + str(self.burstCounter))
        self.unity_marker_file.write(string.encode())
        float_arr = array('d', [time.time()])
        self.unity_marker_ts_file.write(float_arr)
        self.outlet.push_sample(sendme)
        time.sleep(.01)
        self.trigger_sent = True
        self.burstCounter += 1

    #get stream info and set variables
    def get_stream_info(self, from_stream):
        found_channel = False
        info = self.inlet.info()
        info.desc().child_value("nominal_srate")
        self.actual_sr = int(info.nominal_srate())
        self.data_buffer = np.zeros(0)
        ch = info.desc().child("channels").child("channel")
        print('available channels: ')
        for k in range(info.channel_count()): #search through channels for the one indicated in the config file
            print(ch.child_value("label"))
            if ch.child_value("label") == self.seeg_channel:
                self.chSelect = k
                print("found " + str(self.seeg_channel))
                self.info_marker_file.write(("found " + str(self.seeg_channel)).encode())
                float_arr = array('d', [time.time()])
                self.info_marker_ts_file.write(float_arr)
                found_channel = True
            ch = ch.next_sibling()
        if found_channel == False:
            if self.test_mode == 1:
                warnings.warn('Channel ' + str(self.seeg_channel) + ' not found! Setting to 0 THIS IS ONLY FOR TESTING CASES')
            else:
                raise ValueError('Channel ' + str(self.seeg_channel) + ' not found!')
            self.info_marker_file.write(('Channel ' + str(self.seeg_channel) + ' not found! Setting to 0 THIS IS ONLY FOR TESTING CASES').encode())
            float_arr = array('d', [time.time()])
            self.info_marker_ts_file.write(float_arr)
            self.chSelect = 0
    #read chunk for eeg stream
    def read_chunk(self):
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.05, max_samples=1024)
        # if len(np.array(chunk).shape) == 1:
        #     print(chunk)
        if timestamps:
            self.data_buffer = np.append(self.data_buffer, np.array(chunk)[:, self.chSelect], axis=0)
            self.blah+=1
            # print(self.blah)
    #design fir filters
    def design_filters(self):
        kerns = []
        starts = []
        stops = []
        fWidth = 1
        for i, f in enumerate(self.all_frex):  # for each frequency, create a filter
            fStart = (f - fWidth / 2)
            fStop = (f + fWidth / 2)
            starts.append((f - fWidth / 2))
            stops.append(f + fWidth / 2)
            filterkern = design_fir_filter(self.sr, 'bandpass', (fStart, fStop), n_seconds=2)
            kerns.append(filterkern)
        return kerns, starts, stops

    #downsample data using scipy.resample
    def downsample_dat(self, dat):
        secs = len(dat)/self.actual_sr
        samps = secs * self.sr
        Y = scipy.signal.resample(dat, int(samps))
        return Y

    #filter the data for selected frequencies
    def filt_dat(self, dat, b):
        #pat the data
        #print('smoll')
        toPad = int(fftpack.next_fast_len(int(self.padLen*self.sr)))
        dat = np.pad(dat, toPad, mode='constant')
        datFiltered = np.zeros([len(dat), len(self.selected_frex_idx)])
        for i, f in enumerate(self.selected_frex_idx): #filter each frequency
            datFiltered[:, i] = lfilter(b[f], [1.0], dat)
        return np.abs(self.hilbert3(datFiltered))[toPad:-toPad, :] #return power

    #filter the data for the broad spectrum
    def full_filt_dat(self, dat, b):
        #print('full')
        toPad = int(fftpack.next_fast_len(int(self.padLen * self.sr)))
        padMe = np.zeros(toPad)
        dat = np.append(np.append(padMe, dat), padMe)
        datFiltered = np.zeros([len(dat), len(self.all_frex)])
        for i, f in enumerate(self.all_frex):
            datFiltered[:, i] = lfilter(b[i], [1.0], dat)
        return np.abs(self.hilbert3(datFiltered))[toPad:-toPad, :]

    #check for noise using a simple threshold
    def noise_check(self, dat):
        hilb = np.abs(self.hilbert3(zscore(dat)))
        thresh = chi2.ppf(.99, df=2) * np.mean(hilb)
        if np.amax(hilb[-int(self.smollWinSize * self.sr):]) > thresh:
            return True
        else:
            return False

    #fit the background spectrum using FOOOF
    def fooof_fit(self, power_history, fm):
        freqs = self.all_frex
        spectrum = np.mean(power_history, 0) #get spectrum from power history
        freq_range = self.all_frex_range #get the broad spectrum frequency range
        fm.add_data(freqs, spectrum, freq_range) #add data to the model and fit...
        fm.fit()
        self.offset, self.knee, self.exponent = fm.aperiodic_params_
        meanpower = np.power(10, self.offset) * (1 / (self.knee + np.power(freqs, self.exponent)))
        return meanpower

    #get the power thresholds
    def get_thresholds(self, meanpower):
        powthresh = chi2.ppf(self.percentileThresh, df=2) * meanpower[self.selected_frex_idx] / 2
        upperthresh = chi2.ppf(self.upperPercentileThresh, df=2) * meanpower[self.selected_frex_idx] / 2
        return powthresh, upperthresh

    #detect oscillations...
    def detect_osc(self, filtDat, powthresh, durthresh, upperthresh):
        self.trigger_sent = False
        burst_lens = []
        burst_freqs = []
        burst_stops = []
        burst_powers = []

        for selected_idx in range(len(self.selected_frex_idx)):  # for each frequency
            # Find where data passes thresh
            x = np.greater(filtDat[:, selected_idx], powthresh[selected_idx]).astype(int)  # power threshold
            # Let's find the edges of potential bursts
            # dx = np.zeros(len(x) - 1)
            dx = np.diff(x)
            # for i in range(len(dx)):
            #     dx[i] = x[i + 1] - x[i]  # this will identify the edges, with 1 for start, -1 for stop
            starts = np.where(dx == 1)[0] # Shows the +1 and -1 edges
            stops = np.where(dx == -1)[0]
            # Now let's determine if any oscillations start, end, or continue during this period
            if np.all(np.greater(filtDat[:, selected_idx], float(powthresh[selected_idx]))):  # is it all above thresh? AKA a continuation
                if self.currBurstLen[selected_idx] > 0:  # only increase the burst length if already in a burst
                    self.currBurstLen[selected_idx] += len(x) #increase by burst len the entire len of the window
                    self.currBurstPower[selected_idx] += np.sum(filtDat[:, selected_idx]) #add power
            elif np.logical_and(len(starts) == 0, len(stops) == 0):  # if none are detected, cancel any ongoing bursts
                self.currBurstLen[selected_idx] = 0
                self.currBurstActive[selected_idx] = False
                self.currBurstPower[selected_idx] = 0
            elif len(starts) == 0:  # ie, starts on episode and then stops
                #add them to list to be checked for threshold later
                burst_lens.append(self.currBurstLen[selected_idx] + stops[0])
                burst_freqs.append(selected_idx)
                burst_stops.append(stops[0])
                burst_powers.append(self.currBurstPower[selected_idx] + np.sum(filtDat[:stops[0], selected_idx]))
                #stop ongoing burst
                self.currBurstActive[selected_idx] = False
                self.currBurstLen[selected_idx] = 0
                self.currBurstPower[selected_idx] = 0
            elif len(stops) == 0:  # episode starts during window and continues
                self.currBurstLen[selected_idx] = len(x) - starts[0]
                self.currBurstActive[selected_idx] = True
                self.currBurstPower[selected_idx] = np.sum(filtDat[starts[0]:, selected_idx])
            else:  # has both starts and ends
                if starts[0] > stops[0]:  # if it starts during an episode
                    #check if this first burst is above threshold later...
                    burst_lens.append(self.currBurstLen[selected_idx] + stops[0])
                    burst_freqs.append(selected_idx)
                    burst_stops.append(stops[0])
                    burst_powers.append(self.currBurstPower[selected_idx] + np.sum(filtDat[:stops[0], selected_idx]))
                    self.currBurstLen[selected_idx] = 0
                    self.currBurstPower[selected_idx] = 0
                    if (len(stops) > 1): #if there are two or more stops
                        for st in range(len(stops) - 1):  # for each stop beyond the first, check for threshold
                            burst_lens.append(stops[st + 1] - starts[st])
                            burst_freqs.append(selected_idx)
                            burst_stops.append(stops[st+1])
                            burst_powers.append(np.sum(filtDat[starts[st]: stops[st+1], selected_idx]))
                    if (starts[-1] > stops[-1]):  # if it ends on a 'start', add to ongoing burst tracker
                        self.currBurstLen[selected_idx] = len(x) - starts[-1]
                        self.currBurstActive[selected_idx] = True
                        self.currBurstPower[selected_idx] = np.sum(filtDat[starts[-1]:, selected_idx])
                else: # it doesn't start on an episode (must have at least one start and stop)
                    #check the first burst for threshold
                    burst_lens.append(stops[0] - starts[0])
                    burst_freqs.append(selected_idx)
                    burst_stops.append(stops[0])
                    burst_powers.append(np.sum(filtDat[starts[0]:stops[0], selected_idx]))
                    if(len(starts) > 1):#if there is more than one start
                        if(len(starts) == len(stops)): #if starts are equal to stops, it must stop by the end
                            for st in range(len(starts) - 1): #add each burst to check later
                                burst_lens.append(stops[st + 1] - starts[st + 1])
                                burst_stops.append(stops[st+1])
                                burst_freqs.append(selected_idx)
                                burst_powers.append(np.sum(filtDat[starts[st+1]:stops[st+1], selected_idx]))
                        else: #it will end on 'started'
                            if (len(stops) > 1):  # if there are more starts and stops during window
                                for st in range(len(stops) - 1):  # for each additional stop
                                    burst_lens.append(stops[st + 1] - starts[st + 1])
                                    burst_freqs.append(selected_idx)
                                    burst_stops.append(stops[st+1])
                                    burst_powers.append(np.sum(filtDat[starts[st+1]:stops[st+1], selected_idx]))
                            #set the ongoing burst counter for the last 'burst'
                            self.currBurstLen[selected_idx] += len(x) - starts[-1]
                            self.currBurstActive[selected_idx] = True
                            self.currBurstPower[selected_idx] = np.sum(filtDat[starts[-1]:, selected_idx])

        #done with looping through frequencies...
        #
        # b = np.zeros(len(self.currBurstLen), dtype=bool)
        # b[burst_freqs] = True

        passed_thresh_len = []
        passed_thresh_f_idx = []
        passed_thresh_stops = []
        passed_thresh_starts = []
        passed_thresh_powers = []

        #check if any bursts pass threshold
        for j, l in enumerate(burst_lens): #for each burst
            if l >= durthresh[burst_freqs[j]] :#check if it passes threshold
                passed_thresh_len.append(l)
                passed_thresh_f_idx.append(burst_freqs[j])
                start = int(-self.smollWinSize*self.sr + burst_stops[j] - l)
                stop = int(-self.smollWinSize*self.sr + burst_stops[j])
                passed_thresh_starts.append(start)
                passed_thresh_stops.append(stop)
                passed_thresh_powers.append(burst_powers[j])
        #now we have a list of thresholds that have passed.
        if len(passed_thresh_len) > 0:
            #cancel all ongoing oscillations...
            self.currBurstLen = np.zeros(len(self.selected_frex_idx))
            self.currBurstActive = np.zeros(len(self.selected_frex_idx), dtype=bool)
            self.currBurstPower = np.zeros(len(self.selected_frex_idx))
            #prepare variables to send, using the longest burst
            power_to_send=passed_thresh_powers[int(np.argmax(passed_thresh_len))]
            len_to_send = passed_thresh_len[int(np.argmax(passed_thresh_len))]
            freq_to_send_f_idx = passed_thresh_f_idx[int(np.argmax(passed_thresh_len))]
            actual_freq_to_send = self.f_range[freq_to_send_f_idx]
            powthresh_to_send = powthresh[freq_to_send_f_idx]
            upperthresh_to_send = upperthresh[freq_to_send_f_idx]
            meanpower_to_send = power_to_send / len_to_send
            start_to_send = passed_thresh_starts[int(np.argmax(passed_thresh_len))]
            stop_to_send = passed_thresh_stops[int(np.argmax(passed_thresh_len))]
            self.plotBurstStarts.append((self.plotNumWins-1) * self.sr * self.smollWinSize + start_to_send) #used for plotting
            self.plotBurstStops.append((self.plotNumWins-1) * self.sr * self.smollWinSize + stop_to_send)

            sendme = [power_to_send, len_to_send, float(freq_to_send_f_idx),
                                            float(actual_freq_to_send), powthresh_to_send, upperthresh_to_send,
                                            meanpower_to_send, self.sr, 1, self.knee, self.exponent, self.offset,
                                            self.burstCounter]
            # sendme = [power_to_send, len_to_send, float(freq_to_send_f_idx),
            #                                 float(actual_freq_to_send), powthresh_to_send, upperthresh_to_send,
            #                                 meanpower_to_send, self.sr, 1, 0, 0, 0,
            #                                 self.burstCounter]

            self.send_trigger('to_Unity', sendme)

            # shitisactive = False
    def lets_exit(self):
        print('You pressed q, exiting...')
        self.info_marker_file.write(('You pressed q, exiting...' + ';').encode())
        float_arr = array('d', [time.time()])
        self.info_marker_ts_file.write(float_arr)
        print('final time: ' + str(time.time()))
        self.info_marker_file.write(('sEEG stop time:' + str(time.time()) + ';').encode())
        float_arr = array('d', [time.time()])
        self.info_marker_ts_file.write(float_arr)
        time.sleep(1)
        self.sEEG_file.close()
        self.info_marker_ts_file.close()
        self.info_marker_file.close()
        self.unity_marker_file.close()
        self.unity_marker_ts_file.close()
        quit()

    def plot_stuff(self, noise_tracker, noise_found, window_skipped, bMain, aMain, dat, ax):
        t = np.linspace(0, self.plotNumWins * self.smollWinSize,
                        int(self.plotNumWins * self.smollWinSize * self.sr))

        noise_tracker[:-1] = noise_tracker[1:]
        if noise_found or window_skipped:
            noise_tracker[-1] = 1
        else:
            noise_tracker[-1] = 0

        # print("t: " + str(len(t)))
        dat_to_plot = lfilter(bMain, aMain, dat[-int(self.plotNumWins * self.smollWinSize * self.sr):])
        # dat_to_plot = dat[-int(self.plotNumWins * self.smollWinSize * self.sr):]
        # print("dat: " + str(len(dat_to_plot)))
        ax.cla()
        np.amax(np.abs(dat_to_plot))
        ax.set_ylim([-np.amax(np.abs(dat_to_plot)), np.amax(np.abs(dat_to_plot))])
        ax.plot(t, dat_to_plot)
        skip_idx = np.where(noise_tracker == 1)[0]
        for i, j in enumerate(skip_idx):
            ax.plot(t[int(j * self.smollWinSize * self.sr):int((j + 1) * self.smollWinSize * self.sr)],
                     dat_to_plot[int(j * self.smollWinSize * self.sr):int((j + 1) * self.smollWinSize * self.sr)],
                     c='r')
        for i in range(len(self.plotBurstStarts)):
            ax.plot(t[int(self.plotBurstStarts[i]):int(self.plotBurstStops[i])],
                     dat_to_plot[int(self.plotBurstStarts[i]):int(self.plotBurstStops[i])],
                     c='g')
        # if len(self.plotBurstStarts) > 0:
        # pass
        plt.pause(.01)

        for i in range(len(self.plotBurstStarts)):  # for each start
            self.plotBurstStarts[i] = self.plotBurstStarts[i] - (self.smollWinSize * self.sr)
            if self.plotBurstStarts[i] < 0:
                self.plotBurstStarts[i] = 0
        for i in range(len(self.plotBurstStops)):  # for each stop
            self.plotBurstStops[i] = self.plotBurstStops[i] - (self.smollWinSize * self.sr)

        # and remove indices for bursts that would now be outside of the plot
        if len(self.plotBurstStarts) >= 1:
            if self.plotBurstStops[0] < 0:
                self.plotBurstStarts = self.plotBurstStarts[1:]
                self.plotBurstStops = self.plotBurstStops[1:]

    def plot_fit(self, power_history, meanpower):
                # pv, meanpower = self.bgfit(power_history)
                plt.cla()
                plt.plot((np.arange(0, len(self.all_frex))),
                         (np.mean(power_history, 0)), 'ko-')
                plt.plot((np.arange(0, len(self.all_frex))), (meanpower), 'r')
                plt.pause(.01)
    def plot_tf_trace(self, fullFilt, ax):
            ax.cla()
            curr_filt = fullFilt[-int(self.threshWinSize * self.sr):, :]
            self.total_filt[:-int(self.threshWinSize * self.sr), :] = self.total_filt[int(self.threshWinSize * self.sr):, :]
            self.total_filt[-int(self.threshWinSize*self.sr):, :] = curr_filt
            toPlot = self.total_filt.transpose()
            toPlot = toPlot * 15
            t = np.linspace(0, self.plotNumWins * self.smollWinSize,
                            int(self.plotNumWins * self.smollWinSize * self.sr))
            toPlot = np.flip(toPlot, axis = 0)
            ax.imshow(toPlot, aspect='auto', extent = np.concatenate((t[[0, -1]], self.all_frex[[0,-1]])))
            plt.show()
            plt.pause(.01)

    def thread2(self):
        while self.running:
            if keyboard.is_pressed('q'):  # if 'q' pressed, quit on the next loop so quit time matches # of samples...
                print('You pressed q, exiting...')
                self.running = False
                self.info_marker_file.write(('You pressed q, exiting...' + ';').encode())
                float_arr = array('d', [time.time()])
                self.info_marker_ts_file.write(float_arr)
                print('final time: ' + str(time.time()))
                self.info_marker_file.write(('sEEG stop time:' + str(time.time()) + ';').encode())
                float_arr = array('d', [time.time()])
                self.info_marker_ts_file.write(float_arr)
                time.sleep(1)
                self.sEEG_file.close()
                self.info_marker_ts_file.close()
                self.info_marker_file.close()
                self.unity_marker_file.close()
                self.unity_marker_ts_file.close()
                cp.disable()
                s = io.StringIO()
                # sortby= pstats.SortKey.TOTTIME
                ps = pstats.Stats(cp, stream=s).sort_stats('tottime')
                ps.print_stats()
                print(s.getvalue())
                quit()
            time.sleep(10)
    #main function
    def run(self):

        self.running = True
        # plt.figure()
        # fig, axs = plt.subplots(2, 1, figsize=(15, 6))
        print("Starting decoder. PLEASE PRESS 'Q' TO EXIT!")

        threading.Thread(target=self.thread2).start()

        var = input("Enter patient identifier: ")
        self.subj = var
        #script will run while False
        #load config variables
        self.load_config('config.yml')

        self.total_filt = np.zeros([int(self.sr * self.plotNumWins * (self.smollWinSize/self.threshWinSize)), len(self.all_frex)])
        # connect to streams
        eeg_stream_name = self.inlet_name
        unity_stream_name = self.outlet_name
        self.connect_streams(eeg_stream_name, unity_stream_name)
        #find channel, get sr, etc...
        self.get_stream_info(eeg_stream_name)
        #create filters
        b, filter_f_starts, filter_f_stops = self.design_filters()

        # self.total_filt = np.zeros([10*self.sr, len(self.all_frex)])
        total_filt2 = np.zeros(10*self.sr)

        if True:
            nyq = 0.5 * self.sr  # nyquist rate
            fStart = (1 / 2) / nyq
            fStop = (37 / 2) / nyq
            bMain, aMain = butter(3, [fStart, fStop], btype='bp')  # can also be bs, lp, and hp
            noise_tracker = np.zeros(self.plotNumWins)
        #initialize variables
        history_full = False
        time_to_quit = False
        cycle_counter = 0 #used to keep track of cycles for timing purposes
        full_filter_cycle = 1 #keep track of cycles to determine if bg spectrum should be re-fit
        beg_filter_cycles = 0 #Keep track of cycles to determine when baseline data has been fully collected
        skip_counter = 0 #used to count the additional windows to skip after noise detected
        total_skipped_windows = 0
        power_history = np.zeros([int(self.largeWinSize / self.threshWinSize), len(self.all_frex)]) #Store a history of bg spectrums
        full_filt_cycle_avg = [] #keep track of time for BG spec est. passes
        other_cycle_avg  = [] #keep track of time for normal passes
        send_samples = 0
        cycle_prints = 0

        #set duration threshold for burst detection
        durthresh = self.numCyclesThresh * self.sr / self.all_frex[self.selected_frex_idx]

        #initialize FOOOF object for BG fitting
        fm = FOOOF(aperiodic_mode='knee', verbose=False)
        previous_window_noise_found = False

        #print some stuff out & write to stream
        print('Actual SR: ' + str(self.actual_sr))
        self.info_marker_file.write((';Actual SR: ' + str(self.actual_sr) + ';').encode())
        float_arr = array('d', [time.time()])
        self.info_marker_ts_file.write(float_arr)
        print('Target SR: ' + str(self.sr))
        self.info_marker_file.write(('Target SR: ' + str(self.sr) + ';').encode())
        float_arr = array('d', [time.time()])
        self.info_marker_ts_file.write(float_arr)
        #check if we need to downsample
        if self.actual_sr != self.sr:
            print("So we will have to downsample")
            self.info_marker_file.write(("So we will have to downsample" + ';').encode())
            float_arr = array('d', [time.time()])
            self.info_marker_ts_file.write(float_arr)
        print('Frequency range: ' + str(self.f_range))
        self.info_marker_file.write(('Frequency range: ' + str(self.f_range) + ';').encode())
        float_arr = array('d', [time.time()])
        self.info_marker_ts_file.write(float_arr)


        #start collecting data to fill the noise buffer
        do_once = True
        print('Starting data collection.')
        self.info_marker_file.write(('Starting data collection: ' + str(time.time()) + ';').encode())
        float_arr = array('d', [time.time()])
        self.info_marker_ts_file.write(float_arr)
        print('Going to fill noise buffer first: ' + str(self.largeWinSize / self.threshWinSize) + 's')
        self.info_marker_file.write(('Going to fill noise buffer first: ' + str(self.largeWinSize / self.threshWinSize) + 's' + ';').encode())
        float_arr = array('d', [time.time()])
        self.info_marker_ts_file.write(float_arr)
        start_time = time.time()

        while (self.running): #main loop
            # if time_to_quit == True:
            #     self.exit()
            #pull in data
            received_Data = self.read_chunk()
            #Print out every 10 seconds while filling the noise buffer
            if self.data_buffer.shape[0] >= send_samples * self.actual_sr and history_full == False:
                print(str(send_samples) + 's')
                send_samples += 10
            #check if the noise buffer is full
            if self.data_buffer.shape[0] > self.largeWinSize * self.actual_sr + self.smollWinSize * self.actual_sr:
                window_skipped = False
                cycle_start = time.time()
                if do_once == True: #print out info after collecting noise baseline
                    print('Should take ~' + str(self.largeWinSize + self.smollWinSize) + ' seconds.')
                    print('actually took: ' + str(time.time() - start_time))
                    float_arr = array('d', self.data_buffer)
                    self.sEEG_noise_file.write(float_arr)
                    self.sEEG_noise_file.close()
                    ts = np.linspace(start_time, time.time(), self.data_buffer.shape[0])
                    float_arr = array('d', ts)
                    self.sEEG_noise_ts_file.write(float_arr)
                    self.sEEG_noise_ts_file.close()
                    print("finished collecting noise data")
                    self.info_marker_file.write(("finished collecting noise data" + ';').encode())
                    float_arr = array('d', [time.time()])
                    self.info_marker_ts_file.write(float_arr)
                    self.info_marker_file.write(('Should take ~' + str(self.largeWinSize + self.smollWinSize) + ' seconds.' + ';').encode())
                    float_arr = array('d', [time.time()])
                    self.info_marker_ts_file.write(float_arr)
                    self.info_marker_file.write(('actually took: ' + str(time.time() - start_time) + ';').encode())
                    float_arr = array('d', [time.time()])
                    self.info_marker_ts_file.write(float_arr)
                    print('going to collect data for 1/f fit')
                    self.info_marker_file.write(('going to collect data for 1/f fit ' + str(time.time()) + ';').encode())
                    float_arr = array('d', [time.time()])
                    self.info_marker_ts_file.write(float_arr)
                    do_once = False
                    start_time = time.time()
                    prev_final_time = time.time()


                #since buffer won't be exact right length, leave extra data on the buffer
                extraLen = int(
                    self.data_buffer.shape[0] - (self.largeWinSize * self.actual_sr + self.smollWinSize * self.actual_sr))
                dat = self.data_buffer[:-extraLen]  # chop of beginning and extra data on the end, and select one channel
                self.data_buffer = self.data_buffer[-int(extraLen + (self.largeWinSize * self.actual_sr)):]  # leave extra data on data buffer
                #if the actual isn't target sr, downsample the data
                if self.actual_sr != self.sr:
                    dat = self.downsample_dat(dat)
                #write sEEG data to binary file
                float_arr = array('d', dat[-int(self.smollWinSize*self.sr):])
                self.sEEG_file.write(float_arr)
                #subtract DC offset
                dat = dat - np.mean(dat)
                #remove drift
                dat = scipy.signal.detrend(dat)
                #check for noise
                noise_found = self.noise_check(dat)

                if beg_filter_cycles >= cycle_prints and self.sendHistoryFull == 0: #print stuff out while we are filling baseline data for BG
                    print('bgfit cycles: ' + str(beg_filter_cycles) + '. Done at ' + str(int(self.largeWinSize / self.threshWinSize)))
                    self.info_marker_file.write(('bgfit cycles: ' + str(beg_filter_cycles) + '. Done at ' + str(int(self.largeWinSize / self.threshWinSize)) + ';').encode())
                    float_arr = array('d', [time.time()])
                    self.info_marker_ts_file.write(float_arr)
                    cycle_prints += 10
                if noise_found: #if we find noise, skip the window as well as the next X windows
                    skip_counter = self.skipWindows
                    print("Noise detected. Skipping next " + str(skip_counter) + " windows.")
                    self.info_marker_file.write(("Noise detected. Skipping next " + str(skip_counter) + " windows." + ';').encode())
                    float_arr = array('d', [time.time()])
                    self.info_marker_ts_file.write(float_arr)
                    #cancel any ongoing bursts
                    self.currBurstLen = np.zeros(len(self.selected_frex_idx))
                    self.currBurstActive = np.zeros(len(self.selected_frex_idx), dtype=bool)
                    self.currBurstPower = np.zeros(len(self.selected_frex_idx))
                    total_skipped_windows += 1
                    noise_found = True
                elif skip_counter > 0: #keep skipping if noise was recently found
                    window_skipped = True
                    total_skipped_windows += 1
                    skip_counter -= 1
                else: #actually analyze the data
                    #check if it's time to calculate BG spectrum
                    thresh_cycle = full_filter_cycle == int(self.threshWinSize / self.smollWinSize)
                    if thresh_cycle: #if its time to calculate the spectrum
                        #filter all frequencys for the duration set by self.filterWinSize
                        fullFilt = self.full_filt_dat(dat[-int(self.filterWinSize*self.sr):], b)
                        #adjust power history to include new data
                        power_history[:-1,:] = power_history[1:,:]
                        power_history[-1, :] = np.mean(fullFilt, axis = 0)
                        full_filter_cycle = 1 #reset counter
                        beg_filter_cycles += 1

                        if (beg_filter_cycles == int(self.largeWinSize / self.threshWinSize)): #check if enough baseline data has been collected
                            #set some variables and print a bunch of stuff out
                            history_full = True
                            do_once = True
                            total_skipped_windows = 0
                            print("history full, you can start exp. now")
                            self.info_marker_file.write(("history full, you can start exp. now" + ';').encode())
                            float_arr = array('d', [time.time()])
                            self.info_marker_ts_file.write(float_arr)
                            print("should have taken ~" + str((cycle_counter + skip_counter) * self.smollWinSize))
                            self.info_marker_file.write(("should have taken ~" + str((cycle_counter + skip_counter) * self.smollWinSize) + ';').encode())
                            float_arr = array('d', [time.time()])
                            self.info_marker_ts_file.write(float_arr)
                            print("took: " + str(time.time() - start_time))
                            self.info_marker_file.write(("took: " + str(time.time() - start_time) + ';').encode())
                            float_arr = array('d', [time.time()])
                            self.info_marker_ts_file.write(float_arr)
                            print('sEEG start time: ' + str(time.time()))
                            self.info_marker_file.write(('sEEG start time:' + str(time.time()) + ';').encode())
                            float_arr = array('d', [time.time()])
                            self.info_marker_ts_file.write(float_arr)
                            start_time = time.time()
                            self.sendHistoryFull = 1
                            stuff_to_send = np.zeros(self.numDataSend)
                            stuff_to_send[8] = self.sendHistoryFull
                            self.send_trigger('to_Unity', stuff_to_send)
                            cycle_counter = 0
                            test_time = time.time()
                            skip_counter = 0
                        if history_full:  # if history is full and thresh cycle, we can calcualte thresholds
                            #fit the bg spectrum
                            meanpower = self.fooof_fit(power_history, fm)
                            #calculate power thresholds
                            powthresh, upperthresh = self.get_thresholds(meanpower)
                            #grab data that will be used for burst detection
                            dat4BurstCheck = fullFilt[-int(self.smollWinSize * self.sr):, self.selected_frex_idx]
                            self.detect_osc(dat4BurstCheck, powthresh, durthresh, upperthresh)
                            # self.plot_fit(power_history, meanpower)
                            # self.plot_tf_trace(fullFilt, axs[1])

                    else: #not a filter cycle
                        full_filter_cycle += 1 #iterate cycle counter
                        if history_full: #if we are done collecting baseline data
                            #filter the data (just the size set by self.filterWinSize)
                            partFilt = self.filt_dat(dat[-int(self.filterWinSize*self.sr):], b)
                            #check for bursts (using the smollWinSize)
                            dat4BurstCheck = partFilt[-int(self.smollWinSize * self.sr):, :]
                            self.detect_osc(dat4BurstCheck, powthresh, durthresh, upperthresh)
                    # self.plot_stuff(noise_tracker, noise_found, window_skipped, bMain, aMain, dat, axs[0])
                if history_full: #if history full, we can print some info every 60 seconds
                    if cycle_counter + skip_counter == int(60 / self.smollWinSize):
                        print("should be: ~" + str(60) + ', is: ' + str(time.time() - test_time))
                        self.info_marker_file.write(
                            ("should be: ~" + str(60) + ', is: ' + str(time.time() - test_time) + ';').encode())
                        float_arr = array('d', [time.time()])
                        self.info_marker_ts_file.write(float_arr)
                        cycle_counter = 0
                        skip_counter = 0
                        test_time = time.time()
                        self.info_marker_file.write (("avg filt cycle time: " + str(np.mean(full_filt_cycle_avg)) + ';').encode())
                        float_arr = array('d', [time.time()])
                        self.info_marker_ts_file.write(float_arr)
                        print("avg filt cycle time: " + str(np.mean(full_filt_cycle_avg)))
                        self.info_marker_file.write (("avg other cycle time: " + str(np.mean(other_cycle_avg)) + ';').encode())
                        float_arr = array('d', [time.time()])
                        self.info_marker_ts_file.write(float_arr)
                        print("avg other cycle time: " + str(np.mean(other_cycle_avg)) + ';')
                        total_skipped_windows = 0
                        full_filt_cycle_avg = []
                        other_cycle_avg = []

                #check for filtering issues by comparing amplitude at beginning of window to an amplitude predicted from previous window
                #this will ensure that our filter is smooth between windows
                if noise_found == True or window_skipped == True: #check for noise
                    previous_window_noise_found = True
                elif history_full: #make sure we are past the baseline phase
                    current_first_sample = dat4BurstCheck[0, :] #grab first sample to compare to sample predicted
                    if previous_window_noise_found == True: #check if noise was found in previous window
                        previous_window_noise_found = False
                        pred_from_prev = current_first_sample #set current and predicted sample to be the same (as we can't compare due to noise)
                    if do_once: #special case for first window bc no previous window to compare it to (similar to ^^^)
                        pred_from_prev = current_first_sample
                        do_once = False
                    diff = np.abs(np.subtract(current_first_sample, pred_from_prev)) #get the difference between the first sample and the predicted first sample
                    if np.any(diff > 0.15): #check if there is a large jump between samples
                        warnings.warn("Filtering not smooth. If frequent, increase pad length or filter length")
                        self.info_marker_file.write(("Filtering not smooth. If frequent, increase pad length or filter length" + ';').encode())
                        float_arr = array('d', [time.time()])
                        self.info_marker_ts_file.write(float_arr)
                    #predict which sample should come next given current slope
                    testVal = np.zeros(dat4BurstCheck.shape[1])
                    for zCh in range(dat4BurstCheck.shape[1]):
                        pv = np.polyfit(np.arange(len(dat4BurstCheck[-5:, zCh])),
                                        dat4BurstCheck[-5:, zCh], 1)
                        testVal[zCh] = np.polyval(pv, 5)
                    pred_from_prev = testVal

                cycle_counter+=1 #increment cycle counter
                cycle_end_time = time.time() - cycle_start

                #append filter lengths if noise was not found
                if noise_found != True and window_skipped != True:
                    if thresh_cycle:
                        full_filt_cycle_avg.append(cycle_end_time)
                    else:
                        other_cycle_avg.append(cycle_end_time)

            # if keyboard.is_pressed('q'):  # if 'q' pressed, quit on the next loop so quit time matches # of samples...
            #     print('You pressed q, exiting...')
            #     self.info_marker_file.write(('You pressed q, exiting...' + ';').encode())
            #     float_arr = array('d', [time.time()])
            #     self.info_marker_ts_file.write(float_arr)
            #     print('final time: ' + str(time.time()))
            #     self.info_marker_file.write(('sEEG stop time:' + str(time.time()) + ';').encode())
            #     float_arr = array('d', [time.time()])
            #     self.info_marker_ts_file.write(float_arr)
            #     time.sleep(1)
            #     self.sEEG_file.close()
            #     self.info_marker_ts_file.close()
            #     self.info_marker_file.close()
            #     self.unity_marker_file.close()
            #     self.unity_marker_ts_file.close()
            #     cp.disable()
            #     s = io.StringIO()
            #     # sortby= pstats.SortKey.TOTTIME
            #     ps=pstats.Stats(cp, stream=s).sort_stats('tottime')
            #     ps.print_stats()
            #     print(s.getvalue())
            #     quit()








                if False: # plot the power to make sure it's smoooooth
                    if history_full:
                        curr_filt = dat4BurstCheck[:,0]  # perform TF analysis
                        total_filt2[:-int(self.smollWinSize*self.sr)] = total_filt2[int(self.smollWinSize*self.sr):]
                        total_filt2[-int(self.smollWinSize*self.sr):] = curr_filt
                        plt.cla()
                        plt.ylim([0, 35])
                        plt.plot(total_filt2)
                        plt.axhline(powthresh[0], c='r')
                        plt.pause(.01)


                if False:
                    plt.cla()
                    plt.plot(lfilter(bMain, aMain, dat[-int(40 * self.smollWinSize * self.sr):]))
                    plt.pause(.01)





            #old
            if False: #tf and trace
                if history_full == True:
                    # if thresh_cycle:
                    # toPlot = lfilter(bMain, aMain, dat[-int(self.plotNumWins * self.smollWinSize * self.sr):])
                    t = np.linspace(0, self.plotNumWins * self.smollWinSize,
                                    int(self.plotNumWins * self.smollWinSize * self.sr))
                    noise_tracker[:-1] = noise_tracker[1:]
                    if noise_found or window_skipped:
                        noise_tracker[-1] = 1
                    else:
                        noise_tracker[-1] = 0
                    dat_to_plot = lfilter(bMain, aMain, dat[-int(self.plotNumWins * self.smollWinSize * self.sr):])

                    axs[0].cla()
                    axs[0].set_ylim([-150, 150])
                    axs[0].set_xlim([0, self.plotNumWins * self.smollWinSize])
                    axs[0].plot(t, dat_to_plot)
                    axs[0].set_xticks(np.arange(min(t), max(t) + 1, 1.0))
                    axs[0].grid(axis='x', alpha=.3, linestyle='-')
                    # for i, f in enumerate(self.frexIdx):  # for each frequency

                    skip_idx = np.where(noise_tracker == 1)[0]
                    for i, j in enumerate(skip_idx):
                        axs[0].plot(t[int(j * self.smollWinSize * self.sr):int((j + 1) * self.smollWinSize * self.sr)],
                                 dat_to_plot[int(j * self.smollWinSize * self.sr):int((j + 1) * self.smollWinSize * self.sr)],
                                 c='r')
                    for i in range(len(self.plotBurstStarts)):
                        axs[0].plot(t[int(self.plotBurstStarts[i]):int(self.plotBurstStops[i])],
                                    dat_to_plot[int(self.plotBurstStarts[i]):int(self.plotBurstStops[i])],
                                    c='g')

                    # every time there is a new 'end' there should be a countdown until a screenshot is taken
                    # screenshot should happen in half of the total plotwindow.
                    # Plotted every time smollwin happen
                    # we have plotnumwins/2 would be the middle plot, or how many to wait

                    for i in range(len(self.plotBurstStarts)):  # for each start
                        self.plotBurstStarts[i] = self.plotBurstStarts[i] - (self.smollWinSize * self.sr)
                        if self.plotBurstStarts[i] < 0:
                            self.plotBurstStarts[i] = 0
                    for i in range(len(self.plotBurstStops)):  # for each stop
                        self.plotBurstStops[i] = self.plotBurstStops[i] - (self.smollWinSize * self.sr)

                    # and remove indices for bursts that would now be outside of the plot
                    if len(self.plotBurstStarts) >= 1:
                        if self.plotBurstStops[0] < 0:
                            self.plotBurstStarts = self.plotBurstStarts[1:]
                            self.plotBurstStops = self.plotBurstStops[1:]


                    if thresh_cycle:
                        axs[1].cla()
                        curr_filt = fullFilt[-int(self.threshWinSize * self.sr):, :]
                        total_filt[:-int(2*self.sr),:] = total_filt[self.sr:-self.sr, :]
                        total_filt[-int(2*self.sr):-self.sr,:] = curr_filt
                        toPlot = total_filt.transpose()
                        toPlot = toPlot * 15
                        t = np.linspace(0, self.plotNumWins * self.smollWinSize,
                                        int(self.plotNumWins * self.smollWinSize * self.sr))
                        axs[1].pcolormesh(t, self.all_frex, toPlot, vmin=0, vmax=1e3)
                        plt.show()


                        # curr_filt = fullFilt[-int(self.threshWinSize * self.sr):, :]  # perform TF analysis

                        # print(wat)
                        # # curr_filt = curr_filt[-int(self.smollWinSize*self.sr):]
                        # total_filt[:-int(self.threshWinSize * self.sr),:] = total_filt[int(self.threshWinSize * self.sr):, :]
                        # total_filt[-int(self.threshWinSize * self.sr):, :] = curr_filt
                        # print("wat")
                        #
                        # toPlot = total_filt.transpose()
                        # toPlot = toPlot * 15
                        #
                        # t = np.linspace(0, self.plotNumWins * self.smollWinSize,
                        #                 int(self.plotNumWins * self.smollWinSize * self.sr))
                        # axs[1].pcolormesh(t, self.all_frex, toPlot, vmin=0, vmax=1e3)
                        # plt.show()
                        # print('wat')
                    # if noise_found:
                    plt.pause(.01)


            if False: #
                curr_filt = dat4BurstCheck[:,0]  # perform TF analysis
                total_filt[:-int(self.smollWinSize*self.sr)] = total_filt[int(self.smollWinSize*self.sr):]
                total_filt[-int(self.smollWinSize*self.sr):] = curr_filt
                plt.cla()
                plt.ylim([0, 35])
                plt.plot(total_filt)
                plt.axhline(powthresh[0], c='r')
                plt.pause(.01)

            if False: #plot signal w/ marked segments
                t = np.linspace(0, self.plotNumWins * self.smollWinSize,
                                int(self.plotNumWins * self.smollWinSize * self.sr))

                noise_tracker[:-1] = noise_tracker[1:]
                if noise_found or window_skipped:
                    noise_tracker[-1] = 1
                else:
                    noise_tracker[-1] = 0

                # print("t: " + str(len(t)))
                dat_to_plot = lfilter(bMain, aMain, dat[-int(self.plotNumWins * self.smollWinSize * self.sr):])
                # dat_to_plot = dat[-int(self.plotNumWins * self.smollWinSize * self.sr):]
                # print("dat: " + str(len(dat_to_plot)))
                plt.cla()
                plt.ylim([-150, 150])
                plt.plot(t, dat_to_plot)
                skip_idx = np.where(noise_tracker == 1)[0]
                for i, j in enumerate(skip_idx):
                    plt.plot(t[int(j*self.smollWinSize*self.sr):int((j+1) * self.smollWinSize * self.sr)],
                             dat_to_plot[int(j*self.smollWinSize*self.sr):int((j+1) * self.smollWinSize * self.sr)],
                             c='r')
                for i in range(len(self.plotBurstStarts)):
                    plt.plot(t[int(self.plotBurstStarts[i]):int(self.plotBurstStops[i])],
                             dat_to_plot[int(self.plotBurstStarts[i]):int(self.plotBurstStops[i])],
                             c='g')
                # if len(self.plotBurstStarts) > 0:
                # pass
                plt.pause(.01)

                for i in range(len(self.plotBurstStarts)):  # for each start
                    self.plotBurstStarts[i] = self.plotBurstStarts[i] - (self.smollWinSize * self.sr)
                    if self.plotBurstStarts[i] < 0:
                        self.plotBurstStarts[i] = 0
                for i in range(len(self.plotBurstStops)):  # for each stop
                    self.plotBurstStops[i] = self.plotBurstStops[i] - (self.smollWinSize * self.sr)

                # and remove indices for bursts that would now be outside of the plot
                if len(self.plotBurstStarts) >= 1:
                    if self.plotBurstStops[0] < 0:
                        self.plotBurstStarts = self.plotBurstStarts[1:]
                        self.plotBurstStops = self.plotBurstStops[1:]

            #when we have enough power history, we can go on
            #
            if False: #plot the fit
                if history_full == True:
                    #Plot the power fit
                    if True:
                        if thresh_cycle:
                            # pv, meanpower = self.bgfit(power_history)
                            plt.cla()
                            plt.plot((np.arange(0, len(self.all_frex))),
                                     (np.mean(power_history, 0)), 'ko-')
                            plt.plot((np.arange(0, len(self.all_frex))), (meanpower), 'r')
                            plt.pause(.01)










                #

            #
            #
            #
            # #

            #
            # plt.cla()
            # plt.ylim([-20, 20])
            # plt.plot(np.abs(self.data_buffer[-int(self.sr):, 4]))
            # plt.pause(.01)


            # prev_filt = curr_filt[-int(self.smollWinSize):]





            # cur_filt = self.filtered_buffer[]
#
# def thread1():
#     dec = Decoder()
#     dec.run()



if __name__ == '__main__':
    # input_stream_name = 'dev_sEEG'
    # cp.enable()
    # global var
    # var = input("Enter patient identifier: ")
    # self.subj = var
    # threading.Thread(target=thread1).start()
    # threading.Thread(target=thread2).start()
    dec = Decoder()
    dec.run()




