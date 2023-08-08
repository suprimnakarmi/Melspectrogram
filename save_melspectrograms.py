import numpy as np 
import pandas as pd
import concurrent.futures
import librosa
import os
# import matplotlib
# matplotlib.use("Agg")
import traceback
import librosa.display

import matplotlib.pyplot as plt

all_files, labels= [], []
for i in os.listdir("../Coswara-Data/data"):
    if i!=".DS_Store":
        file=pd.read_csv(f"../Coswara-Data/data/{i}/{i}.csv")
        for j in range(len(file)):
            # folder = file.loc[j]['record_date'].replace("-","")
            file_name = file.loc[j]['id']
            all_files.append(f"../Coswara-Data/Extracted_data/{i}/{file_name}/cough-heavy.wav")
            if file.loc[j]["covid_status"][0:8].lower()=="positive":
                labels.append(1)
            else:
                labels.append(0)


def save_spectogram(all_files,labels):
    

    c=0
    
    try:
        y,sr=librosa.load(all_files,sr=None)

        # print('here')
        spec = librosa.feature.melspectrogram(y=y,
                                            sr=sr, 
                                                n_fft=2048, 
                                                hop_length=512, 
                                                win_length=None, 
                                                # window='hann', 
                                                center=True, 
                                                pad_mode='reflect', 
                                                power=2.0,
                                        n_mels=128)
        m_spec = librosa.amplitude_to_db(spec, ref=np.max)
        print(m_spec.shape)
        file_name= all_files.split("/")[-2]
        librosa.display.specshow(m_spec, sr=sr,)
        # plt.savefig(f"../co/{labels}/{file_name}.png",bbox_inches='tight', pad_inches=-0.5)
        plt.savefig(f"../co/{labels}/{file_name}.png",bbox_inches='tight')
        plt.close()

    except Exception as e:
        print("ERRORS in file", all_files)
        print(traceback.format_exc())
        c+=1

    finally:
        plt.close()
   

# os.cpu_count()
def main():
    # num_workers=3
    # print(len(all_files), len(labels))
    for file,label in zip(all_files, labels):
        save_spectogram(file, label)
    # with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
            # executor.map(save_spectogram,all_files[:10],labels[:10])

if __name__ == "__main__":
    main()
# save_spectogram(all_files,labels)