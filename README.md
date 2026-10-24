### Breast Cancer Classification - Flask App
Can predict whether a breast tumor is **benign** or **malignant** from 9 cell measurements (each rated 1-10), using a Kernel SVM model. 

live app: 
https://breast-cancer-hoyne-614dfedd1ad4.herokuapp.com/


### Model
- **Dataset:** UCI Breast Cancer Wisconsin (original), 683 removing missing values
- **Model:** Kernel SVM (RBF), 96.49% test accuracy (75/25 stratified split, random_state=29)
- **Why:** Had the best accuracy of the 8 classifiers from: https://github.com/jannarhoyne-sys/breast-cancer-classification
