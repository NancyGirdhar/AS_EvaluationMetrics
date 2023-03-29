from as_evaluation import calCoverageScore, evalArticleCoverage, evalArticleErrorScore


def main():

# total predicted articles P
    P = [1, 2, 3]  
    # total ground truth articles GT
    GT = [1, 2]
    # total regions articles R
    R = [1, 2, 3, 4, 5, 6]
    ir = [1]
    pr = [2, 3, 4, 5, 6]
    # predicted regions for xth article
    PR= {"a1":[1 , 2 , 5], "a2":[], "a3": [6]}
    # ground truth regions for xth article
    GTR= {"a1":[1 , 4 , 5], "a2":[3], "a3": []}

# Article Evaluation Error (AER)

    s= calCoverageScore(ir,pr,PR,GTR,blockDict2id)
    print("Segmentation", s)
    ACS,meanACS= evalArticleCoverage(PR,GTR)
    print("ACS", ACS)
    print("meanACS", meanACS)
    
    AES,meanAES=evalArticleErrorScore(PR,GTR)
    print("AES", AES)
    print("meanAES", meanAES)

#python_main_function.py
if __name__ == "__main__":
    main()
