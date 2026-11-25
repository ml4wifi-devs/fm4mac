import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from examples.wlan_sim.constants import AM, OP, SCB


KOMONDOR_DIR = os.environ.get('KOMONDOR_DIR', os.path.expanduser('~/tools/Komondor-v4.2.1/Code'))
CB_MODELS = {OP: 0, SCB: 1, AM: 4}
HEADER = (
    'node_code;node_type;wlan_code;x(m);y(m);z(m);central_freq(GHz);channel_bonding_model;primary_channel;'
    'min_channel_allowed;max_channel_allowed;tx_power(dBm);sensitivity(dBm);traffic_model;traffic_load(pkts/s);'
    'packet_length(bits);num_packets_aggregated;capture_effect_model;capture_effect_thr(dB);constant_PER;'
    'pifs_activated;backoff_type;cw_adaptation;cw_min;cw_max;cw_stage;rts_cts_enabled'
)


def nodes_csv(ap_pos, sta_pos, serving, primary, width, policy, params):
    """
    Komondor nodes CSV of APs with the STAs they serve (`sta_pos` of shape [n_stas, 3], `serving` the AP index of
    each STA). APs without STAs are left out (they are silent in the simulator).
    """

    rows = [HEADER]
    for i in range(len(ap_pos)):
        stas = np.flatnonzero(serving == i)
        if len(stas) == 0:
            continue
        lo = (int(primary[i]) >> int(width[i])) << int(width[i])
        hi = lo + (1 << int(width[i])) - 1
        common = (
            f'{params.frequency:g};{CB_MODELS[int(policy[i])]};{int(primary[i])};{lo};{hi};{params.tx_power:g};'
            f'{params.cca:g};99;{{load}};{params.packet_length};{params.n_agg};0;{params.capture_threshold:g};0;0;0;'
            f'{int(params.cw_adaptation)};{params.cw_min};{params.cw_max};{params.cw_stage_max};{int(params.rts_cts)}'
        )
        x, y, z = np.pad(np.asarray(ap_pos[i], dtype=float), (0, 3 - len(ap_pos[i])))
        rows.append(f'AP_{i};0;W{i};{x:.4f};{y:.4f};{z:.4f};' + common.format(load=1000))
        for j in stas:
            x, y, z = np.pad(np.asarray(sta_pos[j], dtype=float), (0, 3 - len(sta_pos[j])))
            rows.append(f'STA_{j};1;W{i};{x:.4f};{y:.4f};{z:.4f};' + common.format(load=0))
    return '\n'.join(rows)


def parse_output(stdout, serving, n_aps):
    """
    Per-AP and per-STA throughput in Mbps from the node reports Komondor prints (APs in CSV order).
    """

    ap_thr = [float(v) for v in re.findall(r'Throughput = ([\d.]+) Mbps', stdout)]
    sta_lines = re.findall(r'Throughput: \{(.*)\}', stdout)
    sta_thr = [[float(v) for v in re.findall(r'([\d.]+) Mbps', line)] for line in sta_lines]
    aps = [i for i in range(n_aps) if (serving == i).any()]
    assert len(ap_thr) == len(sta_thr) == len(aps), (len(ap_thr), len(sta_thr), len(aps))
    per_ap, per_sta = np.zeros(n_aps), np.zeros(len(serving))
    for i, thr, stas in zip(aps, ap_thr, sta_thr):
        per_ap[i] = thr
        per_sta[np.flatnonzero(serving == i)] = stas
    return per_ap, per_sta


def run(ap_pos, sta_pos, serving, primary, width, policy, params, seed=1, komondor_dir=KOMONDOR_DIR, logs_dir=None):
    """
    Runs Komondor for `params.sim_time` and returns the per-AP and per-STA throughput in Mbps. Komondor reads
    `../config_models` and writes `../output`, so each run gets its own working tree. With `logs_dir`, the node
    logs are copied there (for trace comparison).
    """

    serving = np.asarray(serving)
    csv = nodes_csv(ap_pos, sta_pos, serving, primary, width, policy, params)
    with tempfile.TemporaryDirectory() as work:
        shutil.copy(os.path.join(komondor_dir, 'config_models'), work)
        os.makedirs(os.path.join(work, 'main'))
        os.makedirs(os.path.join(work, 'output'))
        nodes = os.path.join(work, 'nodes.csv')
        with open(nodes, 'w') as f:
            f.write(csv)

        stdout = subprocess.run(
            [os.path.join(komondor_dir, 'main', 'komondor_main'), '--nodes', nodes, '--out', 'out.txt', '--code', 'run',
             '--time', str(params.sim_time / 1e6), '--seed', str(seed), '--logs-sys', '0', '--logs-node', '1',
             '--save-node', str(int(logs_dir is not None))],
            cwd=os.path.join(work, 'main'), check=True, capture_output=True, text=True
        ).stdout
        if logs_dir is not None:
            shutil.copytree(os.path.join(work, 'output'), logs_dir, dirs_exist_ok=True)

    return parse_output(stdout, serving, len(ap_pos))
