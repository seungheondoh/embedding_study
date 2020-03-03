import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

label_path = '../../../media/bach3/seungheon/MSD_allMusicTags/ground_truth_assignments/AMG_Multilabel_tagsets'
label_name = ['msd_amglabels_all.h5','msd_amglabels_genres.h5','msd_amglabels_moods.h5','msd_amglabels_styles.h5','msd_amglabels_themes.h5']

save_path = '../dataset'
type_dict = {
        'all' : 0,
        'genres' : 1,
        'moods' : 2,
        'styles' : 3,
        'themes' : 4
        }
def get_doc(label_path, label_name, types='themes'):
    df = pd.read_hdf(os.path.join(label_path, label_name[type_dict[types]]))
    df['all'] = df["genres"] + df["moods"] + df["styles"] + df["themes"]
    
    tag_corpus = list(df['all'])
    print(len(tag_corpus),tag_corpus[0])
if __name__ == '__main__':
    get_doc(label_path, label_name, types='all')