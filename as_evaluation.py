# -*- coding: utf-8 -*-
"""AS_Evaluation.ipynb

Author: Nancy Girdhar
Location: L3i, La Rochelle University, La Rochelle, France
Date: 15 January, 2023

## Article Error Rate
"""
import numpy as np

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def calCoverageScore(independentBlock,dependentBlock,predArticleBlockDict,articleBlockDict):
  groundTruthDependent = list()
  groundTruthIndependent = list()
  for gk,gv in articleBlockDict.items():
    if gk!='' and len(gv) > 1:
      groundTruthDependent.append(gv)
    else:
      groundTruthIndependent.append(gv)
  
  groundTruthDependent    = sorted([element for innerList in groundTruthDependent for element in innerList])
  groundTruthIndependent  = sorted([element for innerList in groundTruthIndependent for element in innerList])
  
  #print("groundTruthDependent:",groundTruthDependent)
  #print("groundTruthIndependent:",groundTruthIndependent)

  predictedDependent  = sorted(dependentBlock)
  predictedIndependent = list()
  for pk,pv in predArticleBlockDict.items():
    predictedIndependent.append(pv)
  
  predictedIndependent = [element for innerList in predictedIndependent for element in innerList]

  tempPredDependent = predictedIndependent 

  predictedIndependent = list(np.setdiff1d(predictedDependent, predictedIndependent))

  predictedDependent = tempPredDependent

  for b in independentBlock:
    predictedIndependent.append(b)
  
  predictedIndependent = sorted(predictedIndependent)

  #print("predictedDependent:",predictedDependent)
  #print("predictedIndependent:",predictedIndependent)

  commonDependent   = intersection(groundTruthDependent, predictedDependent)
  commonIndependent = intersection(groundTruthIndependent, predictedIndependent)

  #print("commonDependent:",commonDependent)
  #print("commonIndependent:",commonIndependent)


  ratioDependent = len(commonDependent) / len(groundTruthDependent)
  rationIndependent = len(commonIndependent) / len(groundTruthIndependent)


  totalPredictedBlock = predictedDependent + predictedIndependent

  totalGroundTruthBlock = groundTruthDependent + groundTruthIndependent

  segmentation = len(totalPredictedBlock)/len(totalGroundTruthBlock)

  return segmentation


"""## Article Coverage Rate (ACR)"""

def evalArticleCoverage(predArticleBlockDict,groundTruthArticleBlockDict):
    """
    Input parameters:
    predArticleBlockDict= dictionary of predicted article with their respective regions(blocks)
    groundTruthArticleBlockDict = dictionary of ground truth article with their respective regions(blocks)

    This function calculates the coverage per article (ACR) ie. the correctly detected regions(blcoks) of that article compare to the ground truth and overall coverall of the articles in a page (mACR)
    """

    ACS = dict()
    for pa,pv in predArticleBlockDict.items():
        dif1 = np.setdiff1d(pv, groundTruthArticleBlockDict[pa])
        dif2 = np.setdiff1d(groundTruthArticleBlockDict[pa],pv)

        n_a = len(dif1) + len(dif2)
        d_a = len(Union(pv,groundTruthArticleBlockDict[pa]))

        ACS[pa] = 1 - (n_a / d_a)

    isum = 0.0
    for aesk,aesv  in ACS.items():
        isum = isum +  aesv

    meanACS = isum/len(ACS)

    return ACS,meanACS

def evalArticleErrorScore(predArticleBlockDict,groundTruthArticleBlockDict):
    """
    Input parameters:
    predArticleBlockDict= dictionary of predicted article with their respective regions(blocks)
    groundTruthArticleBlockDict = dictionary of ground truth article with their respective regions(blocks)

    This function calculates the coverage per article (ACR) ie. the correctly detected regions(blcoks) of that article compare to the ground truth and overall coverall of the articles in a page (mACR)
    """
    AES = dict()
    for pa,pv in predArticleBlockDict.items():
        dif1 = np.setdiff1d(pv, groundTruthArticleBlockDict[pa])
        dif2 = np.setdiff1d(groundTruthArticleBlockDict[pa],pv)

        n_a = len(dif1) + len(dif2)
        d_a = len(Union(pv,groundTruthArticleBlockDict[pa]))

        AES[pa] = n_a / d_a

    isum = 0.0
    for aesk,aesv  in AES.items():
        isum = isum +  aesv

    meanAES = isum/len(AES)

    return AES,meanAES