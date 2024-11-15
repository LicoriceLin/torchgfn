import pandas as pd
from pathlib import Path
from math import log
import pandas as pd
from torch.utils.data import Dataset

db_csv=Path(__file__).parent/"5LSO.db.csv"
offline_db = pd.read_csv(db_csv).set_index("seq")
offline_db = offline_db[offline_db["score"] > 0]
offline_db["score"] = offline_db["score"].apply(log) 
'''
-dG should be considered directly as log rewards.
'''

def offline_query(seq:str,default=0.02):
    offline_db["score"].get(seq, default)



class OfflineSeqOnlyDataSet(Dataset):
    def __init__(self, offline_db:pd.DataFrame=offline_db):
        super().__init__()
        self.seqs = offline_db[offline_db["score"] > 0].index.to_list()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index) -> str:
        return self.seqs[index]

# offline_db["score"] = offline_db["score"].apply(
#             lambda x: 0.02 if x < 0.02 else x / 10
#         )
