DTMF Decoder
============
DTMF Decoder written in python.
[DTMF wiki](https://en.wikipedia.org/wiki/Dual-tone_multi-frequency_signaling)

Requirements
------------
* numpy
* matplotlib

**Install with:**  
`pip install -r requirement.txt`

Usage
-----
```
usage: DTMFDecoder.py [-h] [--debug] [-t TONE_DURATION] [--Nfft NFFT]
                      [--tds TDS] [--fds FDS] [-N KERNEL]
                      [--lpf {convolution,moving_average}]
                      filename

DTMF Decoder

positional arguments:
  filename              filename.wav file

optional arguments:
  -h, --help            show this help message and exit
  --debug               debug flag, set to see plots

Configuration options:
  -t TONE_DURATION, --tone-duration TONE_DURATION
                        tone duration sets duration of sample window [default:
                        0.04]
  --Nfft NFFT           number of points along transformation axis [default:
                        256]
  --tds TDS             Threshold scalar for time domain [default: 5]
  --fds FDS             Threshold scalar for frequency domain [default: 3]

LPF configuration options:
  -N KERNEL, --kernel KERNEL
                        filter kernel size [default: 3]
  --lpf {convolution,moving_average}
                        filter type [default: convolution]
```
