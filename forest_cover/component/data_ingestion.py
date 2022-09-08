from forest_cover.entity.config_entity import DataIngestionConfig
import sys,os
from forest_cover.exception import ForestException
from forest_cover.logger import logging
from forest_cover.entity.artifact_entity import DataIngestionArtifact
import tarfile
import numpy as np
from six.moves import urllib
import zipfile
import pandas as pd
import shutil
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise ForestException(e,sys)
    

    def extract_tgz_file(self,tgz_file_path:str):
        try:
            
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir,exist_ok=True)

            logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir: [{raw_data_dir}]")

            with zipfile.ZipFile(tgz_file_path, 'r') as zip_ref:
                zip_ref.extractall(raw_data_dir)
            logging.info(f"Extracting zip file: {tgz_file_path} into dir: {raw_data_dir}")

        except Exception as e:
            raise ForestException(e,sys) from e
    
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            housing_file_path = os.path.join(raw_data_dir,file_name)

            logging.info(f"Reading csv file: [{housing_file_path}]")
            housing_data_frame = pd.read_csv(housing_file_path)

            housing_data_frame["income_cat"] = pd.cut(
                housing_data_frame["Cover_Type"],
                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                labels=[1,2,3,4,5]
            )
            
            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index,test_index in split.split(housing_data_frame, housing_data_frame["income_cat"]):
                strat_train_set = housing_data_frame.loc[train_index].drop(["income_cat"],axis=1)
                strat_test_set = housing_data_frame.loc[test_index].drop(["income_cat"],axis=1)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)
            

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise ForestException(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            tgz_file_path = self.data_ingestion_config.zip_download_dir
            self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise ForestException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")
