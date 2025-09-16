import os
import torch
import numpy as np
import pandas as pd
import tqdm
import json
from mgca.constants import *
from torch.utils.data import Dataset
from mgca.datasets.utils import get_imgs, read_from_dicom


np.random.seed(101)
JP_list = ['pleural', 'lung', 'pulmonary', 'lungs', 'chest', 'silhouette', 'mediastinal', 
           'cardiomediastinal', 'heart', 'hilar', 'tube', 'osseous', 'lobe', 'vascular', 
           'thoracic', 'catheter', 'interval', 'bibasilar', 'aorta', 'vasculature', 'interstitial', 
           'svc', 'spine', 'silhouettes', 'rib', 'otracheal', 'bony', 'sternotomy', 
           'stomach', 'retrocardiac', 'aortic', 'basilar', 'picc', 'clips', 'costophrenic', 
           'abdomen', 'atrium', 'wires', 'venous', 'nasogastric', 'fluid', 'ventricle', 
           'pacemaker', 'jugular', 'bronchovascular', 'vascularity', 'enteric', 'hila', 'diaphragm', 
           'perihilar', 'port-a-cath', 'arch', 'hemithorax', 'subclavian', 'tissue', 'cavoatrial', 
           'knob', 'vertebral', 'tracheostomy', 'valve', 'pacer', 'artery', 'hiatal', 
           'trachea', 'vein', 'cabg', 'subcutaneous', 'tubes', 'esophagus', 'stent', 
           'vessels', 'cervical', 'sternal', 'neck', 'junction']

BL_list = ['effusion', 'pneumothorax', 'consolidation', 'focal', 'cardiac', 'atelectasis', 'edema', 
           'opacity', 'effusions', 'opacities', 'pneumonia', 'congestion', 'heiaphragm', 'cardiomegaly', 
           'carina', 'opacification', 'degenerative', 'fracture', 'fractures', 'chronic', 'mediastinum', 
           'calcifications', 'infection', 'disease', 'emphysema', 'tortuosity', 'calcification', 'consolidations', 
           'calcified', 'thickening', 'parenchymal', 'atherosclerotic', 'nodular',  'hernia', 'deformity', 
           'engorgement', 'collapse', 'nodule', 'multifocal', 'infectious', 'pneumothoraces', 'density', 
           'diffuse', 'streaky']

ALL_LABELS = JP_list + BL_list  

class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CheXpertImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, img_type="Frontal", data_pct=0.01, imsize=256):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(CHEXPERT_DATA_DIR):
            raise RuntimeError(f"{CHEXPERT_DATA_DIR} does not exist!")

        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(CHEXPERT_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(CHEXPERT_VALID_CSV)
        elif split == "test":
            self.df = pd.read_csv(CHEXPERT_TEST_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # filter image type
        if img_type != "All":
            self.df = self.df[self.df[CHEXPERT_VIEW_COL] == img_type]

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        # get path
        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:])))

        # fill na with 0s
        self.df = self.df.fillna(0)

        # replace uncertains
        # uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        uncertain_mask = {k: -1 for k in CHEXPERT_TASKS}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        self.path = self.df["Path"].values
        # self.labels = self.df.loc[:, CHEXPERT_COMPETITION_TASKS].values
        self.labels = self.df.loc[:, CHEXPERT_TASKS].values
        
    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)
        # get labels
        y = self.labels[index]
        y = torch.tensor(y)
        return x, y

    def __len__(self):
        return len(self.df)


# class MIMICImageDataset(BaseImageDataset):
#     def __init__(self, split="train", transform=None, data_pct=1.0, img_type="Frontal", imsize=256):
#         super().__init__(split=split, transform=transform)
#         if not os.path.exists(MIMIC_CXR_DATA_DIR):
#             raise RuntimeError(
#                 "MIMIC CXR data directory %s does not exist!" % MIMIC_CXR_DATA_DIR)

#         # read in csv file
#         if split == "train":
#             self.df = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
#         elif split == "valid":
#             self.df = pd.read_csv(MIMIC_CXR_VALID_CSV)
#         else:
#             self.df = pd.read_csv(MIMIC_CXR_TEST_CSV)

#         # filter image type
#         if img_type != "All":
#             self.df = self.df[self.df[MIMIC_CXR_VIEW_COL].isin(["PA", "AP"])]

#         # get a fraction of dataset
#         if data_pct != 1.0 and split == "train":
#             self.df = self.df.sample(frac=data_pct, random_state=42)

#         # get path
#         self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
#             lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

#         # fill na with 0s
#         self.df = self.df.fillna(0)

#         # replace uncertains
#         # uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
#         uncertain_mask = {k: -1 for k in CHEXPERT_TASKS}
#         self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

#         self.imsize = imsize

#     def __getitem__(self, index):
#         row = self.df.iloc[index]
#         # get image
#         img_path = row["Path"]
#         x = get_imgs(img_path, self.imsize, self.transform)
#         # get labels
#         # y = list(row[CHEXPERT_COMPETITION_TASKS])
#         y = list(row[CHEXPERT_TASKS])
#         y = torch.tensor(y)
#         return x, y

#     def __len__(self):
#         return len(self.df)

# class MIMICImageDataset(BaseImageDataset):
#     def __init__(self, split="train", transform=None, data_pct=1.0, img_type="Frontal", imsize=256):
#         super().__init__(split=split, transform=transform)
#         if not os.path.exists(MIMIC_CXR_DATA_DIR):
#             raise RuntimeError(f"MIMIC CXR data directory {MIMIC_CXR_DATA_DIR} does not exist!")

#         if split == "train":
#             df_all = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
#             csv_path = '/rds/general/user/lw1824/home/chex/chex/dataset/MIMIC-CXR/mimic_cxr_processed/mimic-cxr-frontal-report-cig_anatboxes-cig_anatlabels-cig_anatphrases.train.csv'
#             self.df = df_all.sample(frac=0.9, random_state=42)
#         elif split == "valid":
#             df_all = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
#             csv_path = '/rds/general/user/lw1824/home/chex/chex/dataset/MIMIC-CXR/mimic_cxr_processed/mimic-cxr-frontal-report-cig_anatboxes-cig_anatlabels-cig_anatphrases.val.csv'
#             self.df = df_all.drop(df_all.sample(frac=0.9, random_state=42).index)
#         else:
#             csv_path = '/rds/general/user/lw1824/home/chex/chex/dataset/MIMIC-CXR/mimic_cxr_processed/mimic-cxr-frontal-report-cig_anatboxes-cig_anatlabels-cig_anatphrases.test.csv'
#             self.df = pd.read_csv(MIMIC_CXR_TEST_CSV)

#         if img_type != "All":
#             self.df = self.df[self.df[MIMIC_CXR_VIEW_COL].isin(["PA", "AP"])]

#         if data_pct != 1.0 and split == "train":
#             self.df = self.df.sample(frac=data_pct, random_state=42)

#         self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
#             lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:]))
#         )

#         self.df = self.df.fillna(0)
#         self.imsize = imsize

#         data_df = pd.read_csv(csv_path)

#         # 构造 sample_id
#         data_df['sample_id'] = data_df[['subject_id','study_id','dicom_id']].astype(str).agg('/'.join, axis=1)

#         # 构造图像路径（如果用 jpg 模式）
#         img_dir = os.path.join('/rds/general/user/lw1824/home/chex/chex/dataset/MIMIC-CXR/mimic-cxr-jpg_2-0-0', "files")
#         data_df['image_path'] = img_dir \
#             + '/p' + data_df.subject_id.str.slice(stop=2) \
#             + '/p' + data_df.subject_id \
#             + '/s' + data_df.study_id \
#             + '/' + data_df.dicom_id + '.jpg'

#         # 解析文本
#         data_df['sentences'] = data_df['report_sentences'].apply(lambda x: json.loads(x))
        
        
        
        
#     def __getitem__(self, index):
#         row = self.df.iloc[index]

#         # 1. 图像
#         img_path = row["Path"]
#         x = get_imgs(img_path, self.imsize, self.transform)

#         # 2. 从句子解析 labels
#         sentences = json.loads(row["report_sentences"]) if isinstance(row["report_sentences"], str) else []
#         y = self._get_labels_from_sentences(sentences)  # tensor [119]

#         return {"imgs": x, "labels": y}

#     def __len__(self):
#         return len(self.df)

#     # -------------------------------
#     def _get_labels_from_sentences(self, sentences):
#         """
#         输入: sentences = ["There is pleural effusion.", "Lungs are clear"]
#         输出: tensor([0/1,...], shape=[119])
#         """
#         labels = torch.zeros(len(ALL_LABELS), dtype=torch.float32)

#         for sent in sentences:
#             sent = sent.lower()
#             for i, keyword in enumerate(ALL_LABELS):
#                 if keyword in sent:   # 简单字符串匹配
#                     labels[i] = 1.0

#         return labels

import os
import json
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

# 你已有的全局变量
MIMIC_CXR_DATA_DIR = "/rds/general/user/lw1824/home/chex/chex/dataset/MIMIC-CXR"


class MIMICImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1.0, img_type="Frontal", imsize=256):
        super().__init__(split=split, transform=transform)
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"MIMIC CXR data directory {MIMIC_CXR_DATA_DIR} does not exist!")

        # -------- 1. 图像 CSV (jpg release) --------
        if split == "train":
            df_all = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
            csv_path = os.path.join(MIMIC_CXR_DATA_DIR,
                "mimic_cxr_processed/mimic-cxr-frontal-report-cig_anatboxes-cig_anatlabels-cig_anatphrases.train.csv")
            self.df = df_all.sample(frac=0.9, random_state=42) 
        elif split == "valid":
            df_all = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
            csv_path = os.path.join(MIMIC_CXR_DATA_DIR,
                "mimic_cxr_processed/mimic-cxr-frontal-report-cig_anatboxes-cig_anatlabels-cig_anatphrases.val.csv")
            self.df = df_all.drop(df_all.sample(frac=0.9, random_state=42).index)
        else:
            csv_path = os.path.join(MIMIC_CXR_DATA_DIR,
                "mimic_cxr_processed/mimic-cxr-frontal-report-cig_anatboxes-cig_anatlabels-cig_anatphrases.test.csv")
            self.df = pd.read_csv(MIMIC_CXR_TEST_CSV)

        if img_type != "All":
            self.df = self.df[self.df[MIMIC_CXR_VIEW_COL].isin(["PA", "AP"])]

        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.df = self.df.fillna(0)
        self.imsize = imsize

        # 提取 sample_id (从 jpg Path 提取 subject_id/study_id/dicom_id)
        self.df["sample_id"] = self.df["Path"].apply(
        lambda x: "/".join([
            x.split("/")[-3].lstrip("p"),  # subject_id 去掉 p
            x.split("/")[-2].lstrip("s"),  # study_id 去掉 s
            x.split("/")[-1].replace(".jpg", "")  # dicom_id
        ])
)
        sample_ids = self.df["sample_id"].tolist()
        print("[INFO] 随机抽查 self.df['Path'] 的内容：")
        for sid in random.sample(sample_ids, min(3, len(sample_ids))):
            print(f"  Path: {sid}")
        # -------- 2. 文本 CSV (processed release) --------
        data_df = pd.read_csv(csv_path)

        # 🔎 只保留 subject_id >= 10000000 (即 p10+)
        data_df = data_df[data_df["subject_id"].astype(int) >= 10000000]

        # 构造 sample_id = subject_id/study_id/dicom_id
        data_df["sample_id"] = data_df[["subject_id", "study_id", "dicom_id"]].astype(str).agg("/".join, axis=1)
        # 随机打印 data_df["sample_id"] 的几个内容
        print("[INFO] 随机抽查 data_df['sample_id'] 的内容：")
        sample_ids = data_df["sample_id"].tolist()
        for sid in random.sample(sample_ids, min(3, len(sample_ids))):
            print(f"  sample_id: {sid}")

        # 解析文本
        data_df["sentences"] = data_df["report_sentences"].apply(lambda x: json.loads(x))

        # -------- 3. 对齐 --------
        df_merged = pd.merge(self.df, data_df, on="sample_id", how="inner")

        self.df = df_merged  # 合并后的 dataframe，包含 Path + sentences
        self.data_df = data_df  # 方便调试时单独看文本

        # ✅ 调试：确认图像和文本是否正常取出
        print(f"[INFO] Loaded {len(self.df)} samples (p10+ only, merged). Checking a few samples...")
        for i in range(min(3, len(self.df))):  # 随机抽查3个
            row = self.df.iloc[random.randint(0, len(self.df) - 1)]
            # print(f"  sample_id: {row['sample_id']}")
            # print(f"  image_path: {row['Path']}")
            # print(f"  sentences: {row['sentences'][:3]}")
            try:
                img = Image.open(os.path.join(MIMIC_CXR_DATA_DIR,row["Path"]))
                print(f"  ✅ Image opened successfully, size={img.size}")
            except Exception as e:
                print(f"  ❌ Failed to open image: {e}")

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # 1. 图像路径
        img_path = os.path.join(MIMIC_CXR_DATA_DIR, row["Path"])
        x = get_imgs(img_path, self.imsize, self.transform)
        # 2. 从句子解析 labels
        sentences = self.data_df[self.data_df["sample_id"] == row["sample_id"]]["sentences"].values
        if len(sentences) > 0:
            sentences = sentences[0]
        else:
            sentences = []
        y = self._get_labels_from_sentences(sentences)  # tensor [len(ALL_LABELS)]
        # print(sentences)
        # print(y)

        return {"imgs": x, "labels": y}


    def __len__(self):
        return len(self.df)

    def _get_labels_from_sentences(self, sentences):
        labels = torch.zeros(len(ALL_LABELS), dtype=torch.float32)
        for sent in sentences:
            sent = sent.lower()
            for i, keyword in enumerate(ALL_LABELS):
                if keyword in sent:   # 简单字符串匹配
                    labels[i] = 1.0
        return labels



    # -------------------------------
    def _get_labels_from_sentences(self, sentences):
        """
        输入: sentences = ["There is pleural effusion.", "Lungs are clear"]
        输出: tensor([0/1,...], shape=[len(ALL_LABELS)])
        """
        labels = torch.zeros(len(ALL_LABELS), dtype=torch.float32)

        for sent in sentences:
            sent = sent.lower()
            for i, keyword in enumerate(ALL_LABELS):
                if keyword in sent:   # 简单字符串匹配
                    labels[i] = 1.0

        return labels


class RSNAImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, phase="classification", data_pct=0.01, imsize=256) -> None:
        super().__init__(split=split, transform=transform)

        if not os.path.exists(RSNA_DATA_DIR):
            raise RuntimeError(f"{RSNA_DATA_DIR} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(RSNA_TRAIN_CSV)
        elif self.split == "valid":
            self.df = pd.read_csv(RSNA_VALID_CSV)
        elif self.split == "test":
            self.df = pd.read_csv(RSNA_TEST_CSV)
        else:
            raise ValueError(f"split {split} does not exist!")

        if phase == "detection":
            self.df = self.df[self.df["Target"] == 1]

        self.df["Path"] = self.df["patientId"].apply(
            lambda x: RSNA_IMG_DIR / (x + ".dcm"))

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["Path"]
        x = read_from_dicom(img_path, self.imsize, self.transform)
        y = float(row["Target"])
        y = torch.tensor([y])
        return x, y


class COVIDXImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=0.01, imsize=256) -> None:
        super().__init__(split=split, transform=transform)

        if not os.path.exists(COVIDX_DATA_DIR):
            raise RuntimeError(f"{COVIDX_DATA_DIR} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(COVIDX_TRAIN_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: COVIDX_DATA_DIR / f"train/{x}")
        elif self.split == "valid":
            self.df = pd.read_csv(COVIDX_VALID_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: COVIDX_DATA_DIR / f"train/{x}")
        elif self.split == "test":
            self.df = pd.read_csv(COVIDX_TEST_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: COVIDX_DATA_DIR / f"test/{x}")
        else:
            raise ValueError(f"split {split} does not exist!")

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["filename"]
        x = get_imgs(img_path, self.imsize, self.transform)
        y = float(row["labels"])
        y = torch.tensor([y])

        return x, y





