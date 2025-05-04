# ZeroR
Correctly Classified Instances         500               65.1042 %
Incorrectly Classified Instances       268               34.8958 %
Kappa statistic                          0     
Mean absolute error                      0.4545
Root mean squared error                  0.4766
Relative absolute error                100      %
Root relative squared error            100      %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.000    0.000    ?          0.000    ?          ?        0.497     0.348     yes
                 1.000    1.000    0.651      1.000    0.789      ?        0.497     0.650     no
Weighted Avg.    0.651    0.651    ?          0.651    ?          ?        0.497     0.544     

=== Confusion Matrix ===

   a   b   <-- classified as
   0 268 |   a = yes
   0 500 |   b = no

# IBk
=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances         521               67.8385 %
Incorrectly Classified Instances       247               32.1615 %
Kappa statistic                          0.294 
Mean absolute error                      0.3221
Root mean squared error                  0.5663
Relative absolute error                 70.8745 %
Root relative squared error            118.8086 %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.545    0.250    0.539      0.545    0.542      0.294    0.637     0.451     yes
                 0.750    0.455    0.755      0.750    0.752      0.294    0.637     0.725     no
Weighted Avg.    0.678    0.384    0.679      0.678    0.679      0.294    0.637     0.629     

=== Confusion Matrix ===

   a   b   <-- classified as
 146 122 |   a = yes
 125 375 |   b = no

# NaiveBayes
Correctly Classified Instances         577               75.1302 %
Incorrectly Classified Instances       191               24.8698 %
Kappa statistic                          0.4425
Mean absolute error                      0.2819
Root mean squared error                  0.426 
Relative absolute error                 62.0134 %
Root relative squared error             89.3796 %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.604    0.170    0.656      0.604    0.629      0.443    0.815     0.661     yes
                 0.830    0.396    0.797      0.830    0.813      0.443    0.815     0.888     no
Weighted Avg.    0.751    0.317    0.747      0.751    0.749      0.443    0.815     0.809     

=== Confusion Matrix ===

   a   b   <-- classified as
 162 106 |   a = yes
  85 415 |   b = no
  
# SMO
Correctly Classified Instances         586               76.3021 %
Incorrectly Classified Instances       182               23.6979 %
Kappa statistic                          0.4448
Mean absolute error                      0.237 
Root mean squared error                  0.4868
Relative absolute error                 52.1399 %
Root relative squared error            102.1318 %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.530    0.112    0.717      0.530    0.609      0.455    0.709     0.544     yes
                 0.888    0.470    0.779      0.888    0.830      0.455    0.709     0.765     no
Weighted Avg.    0.763    0.345    0.757      0.763    0.753      0.455    0.709     0.688     

=== Confusion Matrix ===

   a   b   <-- classified as
 142 126 |   a = yes
  56 444 |   b = no


# MLP
Correctly Classified Instances         579               75.3906 %
Incorrectly Classified Instances       189               24.6094 %
Kappa statistic                          0.4607
Mean absolute error                      0.2942
Root mean squared error                  0.4226
Relative absolute error                 64.7259 %
Root relative squared error             88.6682 %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.657    0.194    0.645      0.657    0.651      0.461    0.803     0.654     yes
                 0.806    0.343    0.814      0.806    0.810      0.461    0.803     0.879     no
Weighted Avg.    0.754    0.291    0.755      0.754    0.754      0.461    0.803     0.800     

=== Confusion Matrix ===

   a   b   <-- classified as
 176  92 |   a = yes
  97 403 |   b = no

# OneR
Correctly Classified Instances         544               70.8333 %
Incorrectly Classified Instances       224               29.1667 %
Kappa statistic                          0.3242
Mean absolute error                      0.2917
Root mean squared error                  0.5401
Relative absolute error                 64.1722 %
Root relative squared error            113.3051 %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.474    0.166    0.605      0.474    0.531      0.329    0.654     0.470     yes
                 0.834    0.526    0.747      0.834    0.788      0.329    0.654     0.731     no
Weighted Avg.    0.708    0.400    0.698      0.708    0.699      0.329    0.654     0.640     

=== Confusion Matrix ===

   a   b   <-- classified as
 127 141 |   a = yes
  83 417 |   b = no


# J48
Correctly Classified Instances         551               71.7448 %
Incorrectly Classified Instances       217               28.2552 %
Kappa statistic                          0.3893
Mean absolute error                      0.3213
Root mean squared error                  0.452 
Relative absolute error                 70.6986 %
Root relative squared error             94.8268 %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.634    0.238    0.588      0.634    0.610      0.390    0.753     0.552     yes
                 0.762    0.366    0.795      0.762    0.778      0.390    0.753     0.819     no
Weighted Avg.    0.717    0.321    0.723      0.717    0.720      0.390    0.753     0.726     

=== Confusion Matrix ===

   a   b   <-- classified as
 170  98 |   a = yes
 119 381 |   b = no

# Random Forest
Correctly Classified Instances         578               75.2604 %
Incorrectly Classified Instances       190               24.7396 %
Kappa statistic                          0.4527
Mean absolute error                      0.3099
Root mean squared error                  0.4068
Relative absolute error                 68.1887 %
Root relative squared error             85.3379 %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.634    0.184    0.649      0.634    0.642      0.453    0.817     0.682     yes
                 0.816    0.366    0.806      0.816    0.811      0.453    0.817     0.891     no
Weighted Avg.    0.753    0.302    0.751      0.753    0.752      0.453    0.817     0.818     

=== Confusion Matrix ===

   a   b   <-- classified as
 170  98 |   a = yes
  92 408 |   b = no