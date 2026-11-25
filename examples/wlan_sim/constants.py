import numpy as np


N_CHANNELS = 8
N_WIDTHS = 4

SLOT = 9
SIFS = 16
DIFS = SIFS + 2 * SLOT
PIFS = SIFS + SLOT
RTS_DURATION = 52
CTS_DURATION = 44
ACK_DURATION = 32
HE_SU_PREAMBLE = 100
OFDM_SYMBOL = 16
MAX_PPDU_DURATION = 5484

SERVICE_FIELD = 16
MPDU_DELIMITER = 32
MAC_HEADER = 320

NOISE_DBM = -95.
CCA_PRIMARY = (-82., -79., -76.)
CCA_SECONDARY = (-72., -69., -66.)

MCS_THRESHOLDS = np.array([-79., -77., -74., -70., -66., -65., -64., -59., -57., -54., -52., -48., -46.])
MODULATION_BITS = np.array([1, 2, 2, 4, 4, 6, 6, 6, 8, 8, 10, 10, 12, 12])
CODING_RATES = np.array([1 / 2, 1 / 2, 3 / 4, 1 / 2, 3 / 4, 1 / 2, 2 / 3, 3 / 4, 3 / 4, 5 / 6, 3 / 4, 5 / 6, 3 / 4,
                         5 / 6])
N_SUBCARRIERS = np.array([234, 468, 980, 1960])

OP, SCB, AM, PU = 0, 1, 2, 3

IDLE, RTS, SIFS_CTS, CTS, SIFS_DATA, DATA, SIFS_ACK, ACK, TIMEOUT = range(9)
