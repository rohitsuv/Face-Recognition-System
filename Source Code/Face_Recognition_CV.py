
# coding: utf-8




from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import imread
from scipy.spatial.distance import euclidean as dist



#Importing Training Images
img1n = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject01.normal.jpg')
img2n = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject02.normal.jpg')
img3n = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject03.normal.jpg')
img7n = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject07.normal.jpg')
img10n = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject10.normal.jpg')
img11n = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject11.normal.jpg')
img14n = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject14.normal.jpg')
img15n = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject15.normal.jpg')




#Importing Testing Images
img1c = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject01.centerlight.jpg')
img1h = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject01.happy.jpg')
img7c = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject07.centerlight.jpg')
img7h = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject07.happy.jpg')
img11c = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject11.centerlight.jpg')
img11h = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject11.happy.jpg')
img12n = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject12.normal.jpg')
img14h = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject14.happy.jpg')
img14s = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/subject14.sad.jpg')
img_apple = imread('/Users/rohitsuvarna/NYU/Comp Vision/CVProject2/Face dataset/apple1_gray.jpg')





#Stacking rows together
train_image_list = [img1n,img2n,img3n,img7n,img10n,img11n,img14n,img15n]
col_vecs = [img.flatten() for img in train_image_list]
col_vecs





#Average Face
m = np.average(col_vecs,axis=0)
m = np.average(col_vecs,axis=0)
mean_face = m.reshape((231,195))
plt.imshow(mean_face,plt.cm.gray)


#Subtracting the average face
col_vecs_reduced = [(vec - m) for vec in col_vecs ]





#Computing matrix A
A = np.vstack(col_vecs_reduced)
A = A.T


#Covariance matrix C
C = np.dot(A,A.T)
L = np.dot(A.T,A)
Eig_v = np.linalg.eig(L)
Eig_vals = Eig_v[0]
Eig_vecs = Eig_v[1]
V = Eig_vecs
Eig_vals


#Finding Eigen Space U
U = np.dot(A,V)




#Projecting Training Faces on Face Space
Omega_training = [np.dot(U.T,train_vec) for train_vec in col_vecs_reduced]



#Face Recognition
comp_image_list = [img1n,img2n,img3n,img7n,img10n,img11n,img14n,img15n,img1c,img1h,img7c,img7h,img11c,img11h,img12n,img14h,img14s,img_apple]
Img_dict = {0:1,1:2,2:3,3:7,4:10,5:11,6:14,7:15}
Actual = [1,2,3,7,10,11,14,15,1,1,7,7,11,11,12,14,14,0]
Predicted = []


#Thresholds 
T_0 = 7000000000000
T_1 = 140000000




def face_detect(img,U,Omega_training,T_0,T_1):
    '''
    Returns the predicted subject if recognised
    else returns 0 for a non-face, -1 for unknown face
    '''
    I   = img.flatten()
    I = I - m
    Omega_I = np.dot(U.T,I)
    I_R = np.dot(U,Omega_I)
    d_0 = dist(I_R,I)
    print('d_o is %d' % d_0)
    dist_array = [dist(Omega_I,Omega) for Omega in Omega_training]
    res = min(dist_array)
    print('d_1 is %d' %res)
    index = Img_dict[dist_array.index(res)]
    if d_0 > T_0:
        return 0
    else:
        if res > T_1:
            return -1
        return index
    


Predicted = [face_detect(img,U,Omega_training,T_0,T_1) for img in comp_image_list]



def print_results(act,pred):
    total = len(act)
    correct = 0
    for i in range(len(act)):
        print("Subject %d is identified as Subject %d"% (act[i],pred[i]))
        if act[i] == pred[i]:
            correct += 1
    print("Got %d right out of %d" % (correct,total)) 
        


#Predictions
print_results(Actual,Predicted)


#Eigenfaces:
    
for i in range(8):
    eigen_face = U[:,i]
    eigen_face = eigen_face.reshape((231,195))
    fig, ax = plt.subplots(1,1)
    im2 = ax.imshow(eigen_face,plt.cm.gray)
    plt.show()
    
    

#Other output for test images:
    
test_image_list = [img1c,img1h,img7c,img7h,img11c,img11h,img12n,img14h,img14s,img_apple]

 #Helper function to print output
def output_func(img,U,Omega_training,T_0,T_1):
    sh = img.shape
    I   = img.flatten()
    I = I - m
    I_res = I.reshape(sh)
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title('I - m face')
    im1 = ax1.imshow(I_res,plt.cm.gray)
    Omega_I = np.dot(U.T,I)
    print('PCA coefficients are \n')
    print(Omega_I)
    I_R = np.dot(U,Omega_I)
    Ir_res = I_R.reshape(sh)
    ax2.set_title('Reconstructed face I_r')
    im2 = ax2.imshow(Ir_res,plt.cm.gray)
    dist_array = [dist(Omega_I,Omega) for Omega in Omega_training]
    print('The distances are \n')
    print(dist_array)
    plt.show()
    print('\n\n\n\n\n\n')
       

for img in test_image_list:
    output_func(img,U,Omega_training,T_0,T_1)





