```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import glob
import os
from tqdm import tqdm
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

```


```python
# Define the path the full dataset file
file_path = 'Input/diabetic_data.csv'

# Read the CSV file into a DataFrame
full_dataset = pd.read_csv(file_path)

full_dataset

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encounter_id</th>
      <th>patient_nbr</th>
      <th>race</th>
      <th>gender</th>
      <th>age</th>
      <th>weight</th>
      <th>admission_type_id</th>
      <th>discharge_disposition_id</th>
      <th>admission_source_id</th>
      <th>time_in_hospital</th>
      <th>...</th>
      <th>citoglipton</th>
      <th>insulin</th>
      <th>glyburide-metformin</th>
      <th>glipizide-metformin</th>
      <th>glimepiride-pioglitazone</th>
      <th>metformin-rosiglitazone</th>
      <th>metformin-pioglitazone</th>
      <th>change</th>
      <th>diabetesMed</th>
      <th>readmitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2278392</td>
      <td>8222157</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[0-10)</td>
      <td>?</td>
      <td>6</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149190</td>
      <td>55629189</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[10-20)</td>
      <td>?</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>...</td>
      <td>No</td>
      <td>Up</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>&gt;30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64410</td>
      <td>86047875</td>
      <td>AfricanAmerican</td>
      <td>Female</td>
      <td>[20-30)</td>
      <td>?</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>500364</td>
      <td>82442376</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[30-40)</td>
      <td>?</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>...</td>
      <td>No</td>
      <td>Up</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16680</td>
      <td>42519267</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[40-50)</td>
      <td>?</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>No</td>
      <td>Steady</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>101761</th>
      <td>443847548</td>
      <td>100162476</td>
      <td>AfricanAmerican</td>
      <td>Male</td>
      <td>[70-80)</td>
      <td>?</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>...</td>
      <td>No</td>
      <td>Down</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>&gt;30</td>
    </tr>
    <tr>
      <th>101762</th>
      <td>443847782</td>
      <td>74694222</td>
      <td>AfricanAmerican</td>
      <td>Female</td>
      <td>[80-90)</td>
      <td>?</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>No</td>
      <td>Steady</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>101763</th>
      <td>443854148</td>
      <td>41088789</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[70-80)</td>
      <td>?</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>No</td>
      <td>Down</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>101764</th>
      <td>443857166</td>
      <td>31693671</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[80-90)</td>
      <td>?</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>10</td>
      <td>...</td>
      <td>No</td>
      <td>Up</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>101765</th>
      <td>443867222</td>
      <td>175429310</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[70-80)</td>
      <td>?</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
<p>101766 rows × 50 columns</p>
</div>




```python
# Get summary of the dataset
full_dataset.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 101766 entries, 0 to 101765
    Data columns (total 50 columns):
     #   Column                    Non-Null Count   Dtype 
    ---  ------                    --------------   ----- 
     0   encounter_id              101766 non-null  int64 
     1   patient_nbr               101766 non-null  int64 
     2   race                      101766 non-null  object
     3   gender                    101766 non-null  object
     4   age                       101766 non-null  object
     5   weight                    101766 non-null  object
     6   admission_type_id         101766 non-null  int64 
     7   discharge_disposition_id  101766 non-null  int64 
     8   admission_source_id       101766 non-null  int64 
     9   time_in_hospital          101766 non-null  int64 
     10  payer_code                101766 non-null  object
     11  medical_specialty         101766 non-null  object
     12  num_lab_procedures        101766 non-null  int64 
     13  num_procedures            101766 non-null  int64 
     14  num_medications           101766 non-null  int64 
     15  number_outpatient         101766 non-null  int64 
     16  number_emergency          101766 non-null  int64 
     17  number_inpatient          101766 non-null  int64 
     18  diag_1                    101766 non-null  object
     19  diag_2                    101766 non-null  object
     20  diag_3                    101766 non-null  object
     21  number_diagnoses          101766 non-null  int64 
     22  max_glu_serum             5346 non-null    object
     23  A1Cresult                 17018 non-null   object
     24  metformin                 101766 non-null  object
     25  repaglinide               101766 non-null  object
     26  nateglinide               101766 non-null  object
     27  chlorpropamide            101766 non-null  object
     28  glimepiride               101766 non-null  object
     29  acetohexamide             101766 non-null  object
     30  glipizide                 101766 non-null  object
     31  glyburide                 101766 non-null  object
     32  tolbutamide               101766 non-null  object
     33  pioglitazone              101766 non-null  object
     34  rosiglitazone             101766 non-null  object
     35  acarbose                  101766 non-null  object
     36  miglitol                  101766 non-null  object
     37  troglitazone              101766 non-null  object
     38  tolazamide                101766 non-null  object
     39  examide                   101766 non-null  object
     40  citoglipton               101766 non-null  object
     41  insulin                   101766 non-null  object
     42  glyburide-metformin       101766 non-null  object
     43  glipizide-metformin       101766 non-null  object
     44  glimepiride-pioglitazone  101766 non-null  object
     45  metformin-rosiglitazone   101766 non-null  object
     46  metformin-pioglitazone    101766 non-null  object
     47  change                    101766 non-null  object
     48  diabetesMed               101766 non-null  object
     49  readmitted                101766 non-null  object
    dtypes: int64(13), object(37)
    memory usage: 38.8+ MB



```python
# Get the unique values for each column and their counts

for col in list(full_dataset.columns):
    counts = full_dataset[col].value_counts()
    print(counts)
    print("==================================")
```

    encounter_id
    2278392      1
    190792044    1
    190790070    1
    190789722    1
    190786806    1
                ..
    106665324    1
    106657776    1
    106644876    1
    106644474    1
    443867222    1
    Name: count, Length: 101766, dtype: int64
    ==================================
    patient_nbr
    88785891     40
    43140906     28
    1660293      23
    88227540     23
    23199021     23
                 ..
    11005362      1
    98252496      1
    1019673       1
    13396320      1
    175429310     1
    Name: count, Length: 71518, dtype: int64
    ==================================
    race
    Caucasian          76099
    AfricanAmerican    19210
    ?                   2273
    Hispanic            2037
    Other               1506
    Asian                641
    Name: count, dtype: int64
    ==================================
    gender
    Female             54708
    Male               47055
    Unknown/Invalid        3
    Name: count, dtype: int64
    ==================================
    age
    [70-80)     26068
    [60-70)     22483
    [50-60)     17256
    [80-90)     17197
    [40-50)      9685
    [30-40)      3775
    [90-100)     2793
    [20-30)      1657
    [10-20)       691
    [0-10)        161
    Name: count, dtype: int64
    ==================================
    weight
    ?            98569
    [75-100)      1336
    [50-75)        897
    [100-125)      625
    [125-150)      145
    [25-50)         97
    [0-25)          48
    [150-175)       35
    [175-200)       11
    >200             3
    Name: count, dtype: int64
    ==================================
    admission_type_id
    1    53990
    3    18869
    2    18480
    6     5291
    5     4785
    8      320
    7       21
    4       10
    Name: count, dtype: int64
    ==================================
    discharge_disposition_id
    1     60234
    3     13954
    6     12902
    18     3691
    2      2128
    22     1993
    11     1642
    5      1184
    25      989
    4       815
    7       623
    23      412
    13      399
    14      372
    28      139
    8       108
    15       63
    24       48
    9        21
    17       14
    16       11
    19        8
    10        6
    27        5
    12        3
    20        2
    Name: count, dtype: int64
    ==================================
    admission_source_id
    7     57494
    1     29565
    17     6781
    4      3187
    6      2264
    2      1104
    5       855
    3       187
    20      161
    9       125
    8        16
    22       12
    10        8
    14        2
    11        2
    25        2
    13        1
    Name: count, dtype: int64
    ==================================
    time_in_hospital
    3     17756
    2     17224
    1     14208
    4     13924
    5      9966
    6      7539
    7      5859
    8      4391
    9      3002
    10     2342
    11     1855
    12     1448
    13     1210
    14     1042
    Name: count, dtype: int64
    ==================================
    payer_code
    ?     40256
    MC    32439
    HM     6274
    SP     5007
    BC     4655
    MD     3532
    CP     2533
    UN     2448
    CM     1937
    OG     1033
    PO      592
    DM      549
    CH      146
    WC      135
    OT       95
    MP       79
    SI       55
    FR        1
    Name: count, dtype: int64
    ==================================
    medical_specialty
    ?                                49949
    InternalMedicine                 14635
    Emergency/Trauma                  7565
    Family/GeneralPractice            7440
    Cardiology                        5352
                                     ...  
    SportsMedicine                       1
    Speech                               1
    Perinatology                         1
    Neurophysiology                      1
    Pediatrics-InfectiousDiseases        1
    Name: count, Length: 73, dtype: int64
    ==================================
    num_lab_procedures
    1      3208
    43     2804
    44     2496
    45     2376
    38     2213
           ... 
    120       1
    132       1
    121       1
    126       1
    118       1
    Name: count, Length: 118, dtype: int64
    ==================================
    num_procedures
    0    46652
    1    20742
    2    12717
    3     9443
    6     4954
    4     4180
    5     3078
    Name: count, dtype: int64
    ==================================
    num_medications
    13    6086
    12    6004
    11    5795
    15    5792
    14    5707
          ... 
    70       2
    75       2
    81       1
    79       1
    74       1
    Name: count, Length: 75, dtype: int64
    ==================================
    number_outpatient
    0     85027
    1      8547
    2      3594
    3      2042
    4      1099
    5       533
    6       303
    7       155
    8        98
    9        83
    10       57
    11       42
    13       31
    12       30
    14       28
    15       20
    16       15
    17        8
    21        7
    20        7
    18        5
    22        5
    19        3
    27        3
    24        3
    26        2
    23        2
    25        2
    33        2
    35        2
    36        2
    29        2
    34        1
    39        1
    42        1
    28        1
    37        1
    38        1
    40        1
    Name: count, dtype: int64
    ==================================
    number_emergency
    0     90383
    1      7677
    2      2042
    3       725
    4       374
    5       192
    6        94
    7        73
    8        50
    10       34
    9        33
    11       23
    13       12
    12       10
    22        6
    16        5
    18        5
    19        4
    20        4
    15        3
    14        3
    25        2
    21        2
    28        1
    42        1
    46        1
    76        1
    37        1
    64        1
    63        1
    54        1
    24        1
    29        1
    Name: count, dtype: int64
    ==================================
    number_inpatient
    0     67630
    1     19521
    2      7566
    3      3411
    4      1622
    5       812
    6       480
    7       268
    8       151
    9       111
    10       61
    11       49
    12       34
    13       20
    14       10
    15        9
    16        6
    19        2
    17        1
    21        1
    18        1
    Name: count, dtype: int64
    ==================================
    diag_1
    428    6862
    414    6581
    786    4016
    410    3614
    486    3508
           ... 
    373       1
    314       1
    684       1
    217       1
    V51       1
    Name: count, Length: 717, dtype: int64
    ==================================
    diag_2
    276     6752
    428     6662
    250     6071
    427     5036
    401     3736
            ... 
    E918       1
    46         1
    V13        1
    E850       1
    927        1
    Name: count, Length: 749, dtype: int64
    ==================================
    diag_3
    250     11555
    401      8289
    276      5175
    428      4577
    427      3955
            ...  
    657         1
    684         1
    603         1
    E826        1
    971         1
    Name: count, Length: 790, dtype: int64
    ==================================
    number_diagnoses
    9     49474
    5     11393
    8     10616
    7     10393
    6     10161
    4      5537
    3      2835
    2      1023
    1       219
    16       45
    10       17
    13       16
    11       11
    15       10
    12        9
    14        7
    Name: count, dtype: int64
    ==================================
    max_glu_serum
    Norm    2597
    >200    1485
    >300    1264
    Name: count, dtype: int64
    ==================================
    A1Cresult
    >8      8216
    Norm    4990
    >7      3812
    Name: count, dtype: int64
    ==================================
    metformin
    No        81778
    Steady    18346
    Up         1067
    Down        575
    Name: count, dtype: int64
    ==================================
    repaglinide
    No        100227
    Steady      1384
    Up           110
    Down          45
    Name: count, dtype: int64
    ==================================
    nateglinide
    No        101063
    Steady       668
    Up            24
    Down          11
    Name: count, dtype: int64
    ==================================
    chlorpropamide
    No        101680
    Steady        79
    Up             6
    Down           1
    Name: count, dtype: int64
    ==================================
    glimepiride
    No        96575
    Steady     4670
    Up          327
    Down        194
    Name: count, dtype: int64
    ==================================
    acetohexamide
    No        101765
    Steady         1
    Name: count, dtype: int64
    ==================================
    glipizide
    No        89080
    Steady    11356
    Up          770
    Down        560
    Name: count, dtype: int64
    ==================================
    glyburide
    No        91116
    Steady     9274
    Up          812
    Down        564
    Name: count, dtype: int64
    ==================================
    tolbutamide
    No        101743
    Steady        23
    Name: count, dtype: int64
    ==================================
    pioglitazone
    No        94438
    Steady     6976
    Up          234
    Down        118
    Name: count, dtype: int64
    ==================================
    rosiglitazone
    No        95401
    Steady     6100
    Up          178
    Down         87
    Name: count, dtype: int64
    ==================================
    acarbose
    No        101458
    Steady       295
    Up            10
    Down           3
    Name: count, dtype: int64
    ==================================
    miglitol
    No        101728
    Steady        31
    Down           5
    Up             2
    Name: count, dtype: int64
    ==================================
    troglitazone
    No        101763
    Steady         3
    Name: count, dtype: int64
    ==================================
    tolazamide
    No        101727
    Steady        38
    Up             1
    Name: count, dtype: int64
    ==================================
    examide
    No    101766
    Name: count, dtype: int64
    ==================================
    citoglipton
    No    101766
    Name: count, dtype: int64
    ==================================
    insulin
    No        47383
    Steady    30849
    Down      12218
    Up        11316
    Name: count, dtype: int64
    ==================================
    glyburide-metformin
    No        101060
    Steady       692
    Up             8
    Down           6
    Name: count, dtype: int64
    ==================================
    glipizide-metformin
    No        101753
    Steady        13


    Name: count, dtype: int64
    ==================================
    glimepiride-pioglitazone
    No        101765
    Steady         1
    Name: count, dtype: int64
    ==================================
    metformin-rosiglitazone
    No        101764
    Steady         2
    Name: count, dtype: int64
    ==================================
    metformin-pioglitazone
    No        101765
    Steady         1
    Name: count, dtype: int64
    ==================================
    change
    No    54755
    Ch    47011
    Name: count, dtype: int64
    ==================================
    diabetesMed
    Yes    78363
    No     23403
    Name: count, dtype: int64
    ==================================
    readmitted
    NO     54864
    >30    35545
    <30    11357
    Name: count, dtype: int64
    ==================================



```python
print("Missing values in each column\n\n")
print("==================================")

for col in list(full_dataset.columns):
    missing_count = full_dataset.loc[(full_dataset[col] == '?') | (full_dataset[col].isnull())].shape[0]
    if (missing_count > 0):
        print(col , ": " , missing_count)
        print('Missing percentage: ', round((missing_count/101766.0)*100, 2), "%")
        print("==================================")
```

    Missing values in each column
    
    
    ==================================
    race :  2273
    Missing percentage:  2.23 %
    ==================================
    weight :  98569
    Missing percentage:  96.86 %
    ==================================
    payer_code :  40256
    Missing percentage:  39.56 %
    ==================================
    medical_specialty :  49949
    Missing percentage:  49.08 %
    ==================================
    diag_1 :  21
    Missing percentage:  0.02 %
    ==================================
    diag_2 :  358
    Missing percentage:  0.35 %
    ==================================
    diag_3 :  1423
    Missing percentage:  1.4 %
    ==================================
    max_glu_serum :  96420
    Missing percentage:  94.75 %
    ==================================
    A1Cresult :  84748
    Missing percentage:  83.28 %
    ==================================



```python
# -------------------------
# High Missing Rate (~95%) 
# -------------------------

# Dropping weight & max_glu_serum. These have too little information to be reliable predictors. 
# Even imputation won't create meaningful signal from around 5% of observed data.

# ------------------------------
# Moderate Missing Rate (40-85%) 
# ------------------------------

# A1Cresult (83.28%)
# An A1C test measures the average amount of sugar in your blood over the past few months. Healthcare providers use 
# it to help diagnose prediabetes and Type 2 diabetes and to monitor how well your diabetes treatment plan is 
# working.

# medical_specialty (49.08%)
# This refers to the medical specialty of the admitting physician. This can be important as this can indirectly give
# information about pre-existing conditions.

# payer_code (39.56%)
# Integer identifier corresponding to 23 distinct values, for example, Blue Cross\Blue Shield, Medicare, and 
# self-pay. (However this was removed as a feature according to the PDF file)

# ------------------------------
# Low Missing Rate (< 3%) 
# ------------------------------

# race (2.23%): Impute with mode

# diag_1 (0.02%): Impute with mode

# diag_2 (0.35%): Impute with "None"

# diag_3 (1.4%): Impute with "None"


# # Handling missing values

# # 1. Drop high-missingness columns
# df_clean = df.drop(['weight', 'max_glu_serum'], axis=1)

# # 2. Handle medium-missingness categoricals
# df_clean['medical_specialty'].fillna('Unknown', inplace=True)
# df_clean['payer_code'].fillna('Missing', inplace=True)
# df_clean['A1Cresult'].fillna('Not_Tested', inplace=True)

# # 3. Handle low-missingness features
# df_clean['race'].fillna(df_clean['race'].mode()[0], inplace=True)
# df_clean['diag_1'].fillna(df_clean['diag_1'].mode()[0], inplace=True)
# df_clean['diag_2'].fillna('None', inplace=True)
# df_clean['diag_3'].fillna('None', inplace=True)

```


```python
# Drop payer_code, weight, & max_glu_serum

df_clean = full_dataset.drop(['weight', 'max_glu_serum', 'payer_code'], axis=1)
df_clean 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encounter_id</th>
      <th>patient_nbr</th>
      <th>race</th>
      <th>gender</th>
      <th>age</th>
      <th>admission_type_id</th>
      <th>discharge_disposition_id</th>
      <th>admission_source_id</th>
      <th>time_in_hospital</th>
      <th>medical_specialty</th>
      <th>...</th>
      <th>citoglipton</th>
      <th>insulin</th>
      <th>glyburide-metformin</th>
      <th>glipizide-metformin</th>
      <th>glimepiride-pioglitazone</th>
      <th>metformin-rosiglitazone</th>
      <th>metformin-pioglitazone</th>
      <th>change</th>
      <th>diabetesMed</th>
      <th>readmitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2278392</td>
      <td>8222157</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[0-10)</td>
      <td>6</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>Pediatrics-Endocrinology</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149190</td>
      <td>55629189</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[10-20)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>?</td>
      <td>...</td>
      <td>No</td>
      <td>Up</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>&gt;30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64410</td>
      <td>86047875</td>
      <td>AfricanAmerican</td>
      <td>Female</td>
      <td>[20-30)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>?</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>500364</td>
      <td>82442376</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[30-40)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>?</td>
      <td>...</td>
      <td>No</td>
      <td>Up</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16680</td>
      <td>42519267</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[40-50)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>?</td>
      <td>...</td>
      <td>No</td>
      <td>Steady</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>101761</th>
      <td>443847548</td>
      <td>100162476</td>
      <td>AfricanAmerican</td>
      <td>Male</td>
      <td>[70-80)</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>?</td>
      <td>...</td>
      <td>No</td>
      <td>Down</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>&gt;30</td>
    </tr>
    <tr>
      <th>101762</th>
      <td>443847782</td>
      <td>74694222</td>
      <td>AfricanAmerican</td>
      <td>Female</td>
      <td>[80-90)</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>?</td>
      <td>...</td>
      <td>No</td>
      <td>Steady</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>101763</th>
      <td>443854148</td>
      <td>41088789</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[70-80)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>?</td>
      <td>...</td>
      <td>No</td>
      <td>Down</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>101764</th>
      <td>443857166</td>
      <td>31693671</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[80-90)</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>10</td>
      <td>Surgery-General</td>
      <td>...</td>
      <td>No</td>
      <td>Up</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>101765</th>
      <td>443867222</td>
      <td>175429310</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[70-80)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>?</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
<p>101766 rows × 47 columns</p>
</div>




```python
y = df_clean['readmitted'] # Target
X = df_clean.drop('readmitted', axis=1) # Features

```


```python
from sklearn.model_selection import train_test_split

# We want to set aside 20% for final testing
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,      # 20% goes to test
    random_state=42,     # Reproducibility
    stratify=y           # Keep same class proportions
)

print("Datapoints for training & validation: ", X_temp.shape[0])
print("Datapoints for testing (in lockbox): ", X_test.shape[0])
```

    Datapoints for training & validation:  81412
    Datapoints for testing (in lockbox):  20354



```python
from collections import Counter
counts = Counter(list(y_temp.values))
total = len(list(y_temp.values))

print("Values in the 'readmitted' column in training & validation dataset")
print("------------------------------------------------------")

for value, count in counts.items():
    percent = count / total * 100
    print(f"{value}: {count} ({percent:.2f}%)")
```

    Values in the 'readmitted' column in training & validation dataset
    ------------------------------------------------------
    >30: 28436 (34.93%)
    NO: 43891 (53.91%)
    <30: 9085 (11.16%)



```python
from collections import Counter
counts = Counter(list(y_test.values))
total = len(list(y_test.values))

print("Values in the 'readmitted' column in testing dataset")
print("------------------------------------------------------")

for value, count in counts.items():
    percent = count / total * 100
    print(f"{value}: {count} ({percent:.2f}%)")
```

    Values in the 'readmitted' column in testing dataset
    ------------------------------------------------------
    NO: 10973 (53.91%)
    <30: 2272 (11.16%)
    >30: 7109 (34.93%)



```python
full_dataset.loc[full_dataset['diag_2'] == '?'][['encounter_id', 'diag_1', 'diag_2', 'diag_3']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encounter_id</th>
      <th>diag_1</th>
      <th>diag_2</th>
      <th>diag_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2278392</td>
      <td>250.83</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>66</th>
      <td>715086</td>
      <td>250.11</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>216</th>
      <td>2735964</td>
      <td>250.03</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>263</th>
      <td>2948334</td>
      <td>250.8</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>431</th>
      <td>3902532</td>
      <td>250.13</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99621</th>
      <td>415526432</td>
      <td>428</td>
      <td>?</td>
      <td>428</td>
    </tr>
    <tr>
      <th>100559</th>
      <td>427825172</td>
      <td>599</td>
      <td>?</td>
      <td>41</td>
    </tr>
    <tr>
      <th>100787</th>
      <td>430828958</td>
      <td>250.01</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>101192</th>
      <td>436145102</td>
      <td>781</td>
      <td>?</td>
      <td>250.02</td>
    </tr>
    <tr>
      <th>101719</th>
      <td>443256548</td>
      <td>250.13</td>
      <td>?</td>
      <td>?</td>
    </tr>
  </tbody>
</table>
<p>358 rows × 4 columns</p>
</div>




```python
full_dataset[['A1Cresult']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A1Cresult</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>101761</th>
      <td>&gt;8</td>
    </tr>
    <tr>
      <th>101762</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>101763</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>101764</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>101765</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>101766 rows × 1 columns</p>
</div>




```python
# clustering for batch effects - PCA/UMAP/tSNE 
# Remove data points that are missing/none/?
# Send notebook to tobi
# do the preprocessing

```
