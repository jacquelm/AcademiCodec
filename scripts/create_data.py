import glob
import sys
import os
import argparse
import numpy as np
import pandas as pd 
import allosaurus.app
from parselmouth.praat import call

if os.path.exists(
    "/home/maxime/Documents/Code/Enregistrement"
):
    sys.path.insert(
        0, "/home/maxime/Documents/Code/Enregistrement"
    )
else:
    sys.path.insert(
        0, "/home/maxime/Documents/Code/Enregistrement"
    )

import utils


def get_parser():
    parser = argparse.ArgumentParser(
        description="Create data files with acoustic parameters for speech signals"
    )
    parser.add_argument(
        "--folder_recs",
        type=str,
        default="RECS/noise/wav",
        help="The folder with the audio files",
    )
    parser.add_argument(
        "--folder_scripts",
        type=str,
        default="RECS/noise/scripts",
        help="The folder with the scripts files",
    )
    parser.add_argument(
        "--data_speaker",
        type=str,
        default="RECS/noise/speakers.txt",
        help="The files with the speakers informations",
    )
    parser.add_argument(
        "--nexus_path",
        type=str,
        default="wav/nexus_calibration/calibration_1khz.wav",
        help="The calibration file of the nexus",
    )
    parser.add_argument("--nexus_volt", type=int, default=1, help="Nexus voltage")
    parser.add_argument("--frame_length", type=float, default=0.025, help="Frame length in seconds")
    parser.add_argument("--frame_shift", type=float, default=0.01, help="Frame shift in seconds")
    parser.add_argument("--subframe_duration", type=float, default=0.005, help="Sub-frame duration in seconds")

    return parser

def main(args):
    df_script_total = None
    speaker_total = np.zeros((0))
    session_total = np.zeros((0))
    file_total = np.zeros((0))
    style_total = np.zeros((0))
    duration_total = np.zeros((0))
    time_total = np.zeros((0))
    f0_total = np.zeros((0))
    intensity_total = np.zeros((0))
    vad_frame_total = np.zeros((0))
    vad_total = np.zeros((0))
    f1_total = np.zeros((0))
    f2_total = np.zeros((0))
    f3_total = np.zeros((0))
    spectral_tilt_total = np.zeros((0))
    spectral_balance_total = np.zeros((0))
    spectral_centroid_total = np.zeros((0))
    stationarity_total = np.zeros((0))

    folder_recs = args.folder_recs + '/*'
    folder_scripts = args.folder_scripts + '/*'
    df_speaker = pd.read_csv(args.data_speaker, sep=",")
    audio, _, _ = utils.load_signals(args.nexus_path)
    calib = np.max(abs(audio))
    model = allosaurus.app.read_recognizer()

    for script, folder_wav in zip([x for x in sorted(glob.glob(folder_scripts)) if "todo" not in x], [x for x in sorted(glob.glob(folder_recs)) if "data" not in x]):
        (speaker_temp,session_temp,file_temp,style_temp,duration_temp,
        time_temp,intensity_temp,vad_frame_temp,vad_temp,stationarity_temp,
        f0_temp,f1_temp,f2_temp,f3_temp,spectral_tilt_temp,
        spectral_balance_temp,spectral_centroid_temp,df_script) = utils.calculate_acoustic(script, folder_wav, df_script_total, model, df_speaker, calib, args)
        speaker_total = np.append(speaker_total,speaker_temp)
        session_total = np.append(session_total, session_temp)
        file_total = np.append(file_total, file_temp)
        style_total = np.append(style_total,style_temp)
        duration_total = np.append(duration_total, duration_temp)
        time_total = np.append(time_total, time_temp)
        f0_total = np.append(f0_total, f0_temp)
        intensity_total = np.append(intensity_total, intensity_temp)
        vad_frame_total = np.append(vad_frame_total, vad_frame_temp)
        vad_total = np.append(vad_total, vad_temp)
        f1_total = np.append(f1_total, f1_temp)
        f2_total = np.append(f2_total, f2_temp)
        f3_total = np.append(f3_total, f3_temp)
        spectral_tilt_total = np.append(spectral_tilt_total, spectral_tilt_temp)
        spectral_balance_total = np.append(spectral_balance_total, spectral_balance_temp)
        spectral_centroid_total = np.append(spectral_centroid_total, spectral_centroid_temp)
        stationarity_total = np.append(stationarity_total, stationarity_temp)
        df_script_total = pd.concat([df_script_total, df_script])

    # Add the data to Pandas
    df_recs = pd.DataFrame(
        np.column_stack(
            [
                speaker_total,
                session_total,
                file_total,
                style_total,
                duration_total,
                time_total,
                intensity_total,
                vad_frame_total,
                vad_total,
                stationarity_total,
                12 * np.log2(f0_total),
                12 * np.log2(f1_total),
                12 * np.log2(f2_total),
                12 * np.log2(f3_total),
                spectral_tilt_total,
                spectral_balance_total,
                12 * np.log2(spectral_centroid_total),
            ]
        ),
        columns=[
            "speaker",
            "session",
            "file",
            "style",
            "duration",
            "time",
            "Intensity",
            "VAD Frame",
            "VAD",
            "Stationarity",
            "F0",
            "F1",
            "F2",
            "F3",
            "Tilt",
            "Balance",
            "Centroid",
        ],
    )
    df_recs = df_recs.replace({'style': {"silence": "Normal", "noise1": "65dB", "noise2": "75dB", "noise3": "85dB"}})
    df_recs.to_hdf(os.path.join(folder_recs.replace("/*",""),"wav_data"), 'df', 'w')
    df_script_total.to_hdf(os.path.join(folder_recs.replace("/*",""),"script_data"), 'df', 'w')

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)