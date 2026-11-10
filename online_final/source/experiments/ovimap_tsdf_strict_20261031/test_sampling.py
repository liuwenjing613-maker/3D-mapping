import numpy as np
from baseline import sample,Frame,Camera
def main():
 cam=Camera(4,4,1.,1.,0.,0.,1000.)
 mask=np.zeros((4,4),np.uint8);mask[0,0]=2;mask[1,1]=28
 frame=Frame(0,np.zeros((4,4,3),np.uint8),np.ones((4,4),np.float32),mask,np.eye(4),cam)
 row=sample(frame,{2:100,28:300});assert row['ids'].tolist()==[100]
 mask[:]=0;mask[1,1]=28
 row=sample(frame,{28:300});assert row['world'].shape==(0,3) and len(row['ids'])==0
 print('UNSAMPLED_MASK_TEST_PASS')
if __name__=='__main__':main()
