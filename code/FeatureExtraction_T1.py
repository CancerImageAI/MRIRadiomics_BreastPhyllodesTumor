# -*- coding: utf-8 -*- 
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor,imageoperations
import six
import sys, os
import pandas as pd
from pandas import DataFrame as DF
import scipy
import matplotlib.pyplot as plt
import xlrd
from tqdm import tqdm

def readDCM_Img(FilePath):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    return image

def Extract_Features(image,mask,params_path):
    paramsFile = os.path.abspath(params_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info

if __name__ == '__main__':
    TrainingList_workbook = xlrd.open_workbook('../TrainingList.xls')
    TrainingList_sheet = TrainingList_workbook.sheet_by_name(sheet_name="Sheet1")
    Training_PatientID = TrainingList_sheet.col_values(0)[1:]
    Training_Class = np.array(TrainingList_sheet.col_values(2)[1:])
    ZQ_Feature = []
    ZQ1_Feature = []
    for i in tqdm(range(len(Training_PatientID))):
        PatientID = Training_PatientID[i]
        if len(PatientID.split('B'))==2 and len(PatientID.split('BD'))==1:
            image_root = '../Benign/Image'
            mask_root = '../Benign/ROI'
        elif len(PatientID.split('BD'))==2:
            image_root = '../borderline/Image'
            mask_root = '../borderline/ROI'
        else:
            image_root = '../malignant/Image'
            mask_root = '../malignant/ROI'
        ZQ_image_path = os.path.join(image_root,PatientID+'/ZQ0')
        ZQ_image = readDCM_Img(ZQ_image_path)
        ZQ_mask_path = os.path.join(mask_root,PatientID+'ZQ2.nii.gz')
        ZQ_mask = sitk.ReadImage(ZQ_mask_path)
        ZQ_features, ZQ_feature_info = Extract_Features(ZQ_image, ZQ_mask, 'params_T1.yaml')
        ZQ_features['PatientID'] = PatientID
        ZQ_features['Class'] = Training_Class[i]
        ZQ_Feature.append(ZQ_features)
        
        ZQ1_image_path = os.path.join(image_root,PatientID+'/ZQ1')
        ZQ1_image = readDCM_Img(ZQ1_image_path)
        ZQ1_mask_path = os.path.join(mask_root,PatientID+'ZQ2.nii.gz')
        ZQ1_mask = sitk.ReadImage(ZQ1_mask_path)
        ZQ1_features, ZQ1_feature_info = Extract_Features(ZQ1_image, ZQ1_mask, 'params_T1.yaml')
        ZQ1_features['PatientID'] = PatientID
        ZQ1_features['Class'] = Training_Class[i]
        ZQ1_Feature.append(ZQ1_features)        

    df = DF(ZQ_Feature).fillna('0')
    df.to_csv('../result/Train_ZQ_Feature.csv',index=False,sep=',')
    df = DF(ZQ1_Feature).fillna('0')
    df.to_csv('../result/Train_ZQ1_Feature.csv',index=False,sep=',')
    
    TestingList_workbook = xlrd.open_workbook('../TestingList.xls')
    TestingList_sheet = TestingList_workbook.sheet_by_name(sheet_name="Sheet1")
    Testing_PatientID = TestingList_sheet.col_values(0)[1:]
    Testing_Class = np.array(TestingList_sheet.col_values(2)[1:])
    ZQ_Feature = []
    ZQ1_Feature = []
    for i in tqdm(range(len(Testing_PatientID))):
        PatientID = Testing_PatientID[i]
        if len(PatientID.split('B'))==2 and len(PatientID.split('BD'))==1:
            image_root = '../Benign/Image'
            mask_root = '../Benign/ROI'
        elif len(PatientID.split('BD'))==2:
            image_root = '../borderline/Image'
            mask_root = '../borderline/ROI'
        else:
            image_root = '../malignant/Image'
            mask_root = '../malignant/ROI'
        ZQ_image_path = os.path.join(image_root,PatientID+'/ZQ0')
        ZQ_image = readDCM_Img(ZQ_image_path)
        ZQ_mask_path = os.path.join(mask_root,PatientID+'ZQ2.nii.gz')
        ZQ_mask = sitk.ReadImage(ZQ_mask_path)
        ZQ_features, ZQ_feature_info = Extract_Features(ZQ_image, ZQ_mask, 'params_T1.yaml')
        ZQ_features['PatientID'] = PatientID
        ZQ_features['Class'] = Testing_Class[i]
        ZQ_Feature.append(ZQ_features)
        
        ZQ1_image_path = os.path.join(image_root,PatientID+'/ZQ1')
        ZQ1_image = readDCM_Img(ZQ1_image_path)
        ZQ1_mask_path = os.path.join(mask_root,PatientID+'ZQ2.nii.gz')
        ZQ1_mask = sitk.ReadImage(ZQ1_mask_path)
        ZQ1_features, ZQ1_feature_info = Extract_Features(ZQ1_image, ZQ1_mask, 'params_T1.yaml')
        ZQ1_features['PatientID'] = PatientID
        ZQ1_features['Class'] = Testing_Class[i]
        ZQ1_Feature.append(ZQ1_features)        

    df = DF(ZQ_Feature).fillna('0')
    df.to_csv('../result/Test_ZQ_Feature.csv',index=False,sep=',')
    df = DF(ZQ1_Feature).fillna('0')
    df.to_csv('../result/Test_ZQ1_Feature.csv',index=False,sep=',')
    