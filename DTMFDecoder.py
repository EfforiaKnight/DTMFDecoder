#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from wave import open as open_wave
import argparse

# Disable warning of divide by Null/Zero/NiNF/None
np.seterr(divide='ignore', invalid='ignore')

# 1633Hz for A, B, C, D column tones
DTMF_FREQ = [697, 770, 852, 941, 1209, 1336, 1477]
DTMF_KEYPAD = [["1", "2", "3"],
               ["4", "5", "6"],
               ["7", "8", "9"],
               ["*", "0", "#"]]


class DTMFDecoder:
    def __init__(self, N_fft=256, tone_duration=0.04, kernel=3,
                 td_scalar=5, fd_scalar=3, lpf="convolution", debug=False):
        # Config parameters
        self.N_fft = N_fft
        self.N = kernel
        self.td_scalar = td_scalar
        self.fd_scalar = fd_scalar
        self.tone_duration = tone_duration
        self.lpf = lpf

        self.debug = debug

        # Debug variables
        if self.debug:
            fig, self.time_ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    def plot_time(self, signal, ax, title=""):
        ''' Plot sginal in time domain.
        '''
        ax.set_title(title)
        ax.set_xlabel("Time [samples]")
        ax.set_ylabel("Magnitude")
        ax.plot(signal)

    def plot_spectrum(self, freq, S, ax, title=""):
        ''' Plot sginal in frequency domain.
        '''
        ax.set_title(title)
        # ax.set_xlabel("Frequency [Hz]")
        # ax.set_ylabel("Magnitude")
        ax.plot(freq, S)

    def plot_signal_and_spectrum(self, s, fs):
        ''' Plot sginal in time domain and in frequency domain.
        '''
        plt.subplots(1, 2, figsize=(20, 6))

        plt.subplot(121)
        plt.plot(s)
        plt.xlabel("Time [sec]")
        plt.ylabel("Magnitude")
        plt.ylim([-1.5, 1.5])
        plt.grid()

        # spectrum of s (fs samples taken)
        # plot only the 1st half of spectrum (since it's symmetric)
        plt.subplot(122)

        amps = np.abs(np.fft.rfft(s, fs))
        freqs = np.fft.rfftfreq(fs, 1 / fs)

        plt.xlabel("Freq [Hz]")
        plt.ylabel("Magnitude")
        plt.plot(freqs, amps)
        plt.grid()

    def moving_average(self, data, window_width=3):
        ''' Moving average filter
        '''
        cumsum_vec = np.cumsum(np.insert(data, 0, 0), dtype=float)
        ma_vec = (cumsum_vec[window_width:] -
                  cumsum_vec[:-window_width]) / window_width
        return ma_vec

    def group_nearest(self, samples, threshold):
        """ [Generator function] Groups frequencies where the diff is lower than threshold.

        Parameters
        ----------
        samples : (ndarray)
            Sample candidates. Form (freq, amp)
        threshold: (integer)
            Threshold.

        Returns
        ----------
        object : (generator object)
            Grouped numpy array
        """
        grp = []
        last = samples[0][0]  # First time idx
        for (time, amp) in samples:
            if time - last > threshold:
                yield np.mean(grp, axis=0)
                grp = []
            grp.append((time, amp))
            last = time
        yield np.mean(grp, axis=0)

    def dtmf_freq_idx(self, samples_freq):
        """ Find the closest frequency from samples_freq to DTMF frequencies.

        Parameters
        ----------
        samples_freq : (ndarray)
            List of frequencies from fftfreq

        Returns
        ----------
        idxs : (list)
            List of indexes from DTMF_FREQ list that correspond to samples_freq
        """
        idxs = []
        for freq in samples_freq:
            idxs.append((np.abs(DTMF_FREQ - freq)).argmin())
        return idxs

    def get_key(self, dtmf_idx):
        ''' Get the key from DTMF keypad

        Parameters
        ----------
        dtmf_idx : (integer)
            Index/es of DTMF keypad

        Returns
        ----------
        key : (string)
            Key/Number from DTMF keypad
        '''
        row = dtmf_idx[0]
        col = dtmf_idx[1] - len(DTMF_KEYPAD)

        return DTMF_KEYPAD[row][col]

    def normalize(self, array):
        """ Normalizes positive array only.

        Parameters
        ----------
        array : (ndarray)
            signal array

        Returns
        ----------
        array : (ndarray)
            normilized signal array
        """
        high, low = abs(max(array)), abs(min(array))
        return array / max(high, low)

    def read_wave(self, filename):
        """ Reads a wave file.

        Parameters
        ----------
        filename: (string)

        Returns
        ----------
        array : (ndarray)
            normilized signal array
        """
        fp = open_wave(filename, 'r')

        nchannels = fp.getnchannels()
        nframes = fp.getnframes()
        sampwidth = fp.getsampwidth()
        framerate = fp.getframerate()

        z_str = fp.readframes(nframes)

        fp.close()

        dtype_map = {1: np.int8, 2: np.int16}
        if sampwidth not in dtype_map:
            raise ValueError('sampwidth {} unknown'.format(sampwidth))

        array = np.fromstring(z_str, dtype=dtype_map[sampwidth])

        # if it's in stereo, just pull out the first channel
        if nchannels == 2:
            array = array[::2]

        return framerate, array, nframes

    def find_candidates(self, sig):
        ''' Filter and find candidates whos amplitude are above threshold

        Parameters
        ----------
        sig: (ndarray)

        Returns
        ----------
        tone_candidates : (ndarray)
            Frequency and amplitude of the candidates
        """
        '''
        # Executing convolution on smaller window is faster than moving average
        # But very computation time consuming on bigger window
        # for N = 3
        # moving_average: 674 µs ± 13.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        # convolve:       274 µs ± 44.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

        # for N = 1000
        # moving_average: 689 µs ± 46.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        # convolve:       14 ms ± 166 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        if self.lpf == "convolution":
            s_filtered = np.abs(np.convolve(sig, np.ones(self.N) / self.N, mode='valid'))
        elif self.lpf == "moving_average":
            s_filtered = np.abs(self.moving_average(sig, window_width=self.N))
        else:
            raise Exception("{} is not supported!".format(self.lpf))

        s_filtered = self.normalize(s_filtered)
        if self.debug:
            self.plot_time(s_filtered, self.time_ax[1], title="Filtered Signal in Time Domain")

        sig_mean = np.mean(s_filtered)

        tone_candidates = np.array([(sample, amp) for (sample, amp) in enumerate(
            s_filtered) if amp > sig_mean * self.td_scalar])

        return tone_candidates

    def freq_analyze(self, s, td_pieces, smpl_freqs, smpl_duration):
        ''' Analyze each time piece in frequency domain.

        Parameters
        ----------
        s : (ndarray)
            Original signal
        td_pieces : (ndarray)
            Time Domain pieces (candidates) to decode
        smpl_freqs : (ndarray)
            Span of frequencies
        smpl_duration : (float)
            Duration of the sample, sample_rate x tone_time_duration
        Returns
        ----------
        freq_cand : (ndarray)
            Frequency candidates
        start : (integer)
            Start offset
        stop : (integer)
            Stop offset
        duration : (integer)
            Duration of the signal
        '''
        decoded = []
        time_offset = []

        if self.debug:
            N = len(td_pieces)
            fig, ax = plt.subplots(N, sharex=True, figsize=(5, 9))
            fig.text(0.5, 0, 'Frequency [Hz]', ha='center')
            fig.text(0, 0.5, 'Magnitude', va='center', rotation='vertical')
            fig.tight_layout()

        for n, td_piece in enumerate(td_pieces):
            start = int(td_piece[0] - smpl_duration / 2)
            stop = int(td_piece[0] + smpl_duration / 2)
            time_offset.append((start, stop))

            # Amplitudes of the samples after fft
            smpl_fft = np.abs(np.fft.rfft(s[start:stop], self.N_fft))
            smpl_fft = self.normalize(smpl_fft)
            if self.debug:
                self.plot_spectrum(smpl_freqs, smpl_fft, ax[n],
                                   title="Candidate # {}".format(n))

            # Take only highest amp frequencies
            freq_cand = smpl_freqs[np.argwhere(smpl_fft > np.average(smpl_fft) * self.fd_scalar)]

            # Convert freq candidates to DTMF indexes
            DTMF_idx = list(set(self.dtmf_freq_idx(freq_cand)))

            # Check if we have exactly two candidates
            if len(DTMF_idx) == 2:
                decoded.append(self.get_key(DTMF_idx))
            else:
                decoded.append(None)

        return decoded, time_offset

    def decode_signal(self, filename="phonecall.wav"):
        ''' Decodes DTMF signal.

        Parameters
        ----------
        filename: (string)

        Returns
        ----------
        decoded : (string)
            Decoded string
        '''

        fs, s, frames = self.read_wave(filename)

        if self.debug:
            self.plot_time(s, self.time_ax[0], title="Signal from {}".format(filename))

        smpl_duration = fs * self.tone_duration

        tone_candidates = self.find_candidates(s)
        grp_cand = list(self.group_nearest(tone_candidates, threshold=400))
        # print(len(grp_cand))

        # Frequency span of the samples
        smpl_freqs = np.fft.rfftfreq(self.N_fft, 1 / fs)

        # Analyze each time group (candidates)
        (decoded, time_offset) = self.freq_analyze(s, grp_cand, smpl_freqs, smpl_duration)

        # Print appropriete message to the user
        if all(decoded):
            print("Decoded numbers are:\n{}".format(decoded))
            for i, (start, stop) in enumerate(time_offset):
                print("Num: {}, Start: {} [Sample], Stop: {} [Sample]".format(decoded[i], start, stop))
        elif any(decoded):
            print("Could not decode all candidates. Decoded numbers\
                    are:\n{}\nTry changing the tone duration".format(decoded))
        else:
            print("Could not decode DTMF from {} try changing the tone duration".format(filename))

        if self.debug:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTMF Decoder")

    parser.add_argument(
        "filename", type=str, help="filename.wav file")
    parser.add_argument("--debug", action="store_true",
            help="debug flag, set to see plots")

    conf = parser.add_argument_group("Configuration options")
    conf.add_argument(
            "-t", "--tone-duration", default=0.04, type=float,
            help="tone duration sets duration of sample window [default: 0.04]")
    conf.add_argument(
            "--Nfft", default=256, type=int,
            help="number of points along transformation axis [default: 256]")
    conf.add_argument(
            "--tds", default=5, type=int,
            help="Threshold scalar for time domain [default: 5]")
    conf.add_argument(
            "--fds", default=3, type=int,
            help="Threshold scalar for frequency domain [default: 3]")

    lpf_conf = parser.add_argument_group("LPF configuration options")
    lpf_conf.add_argument(
            "-N", "--kernel", default=3, type=int,
            help="filter kernel size [default: 3]")
    lpf_conf.add_argument(
            "--lpf", default="convolution", type=str, choices=["convolution", "moving_average"],
            help="filter type [default: convolution]")

    args = parser.parse_args()
    test = DTMFDecoder(debug=args.debug, tone_duration=args.tone_duration,
            kernel=args.kernel, N_fft=args.Nfft, td_scalar=args.tds,
            fd_scalar=args.fds, lpf=args.lpf)
    try:
        test.decode_signal(filename=args.filename)
    except:
        if args.debug:
            plt.show()

