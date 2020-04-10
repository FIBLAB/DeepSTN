from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np

from keras.optimizers import Adam
from keras.layers import Input,Activation,Dropout,BatchNormalization,AveragePooling2D,ZeroPadding2D,Multiply
from keras.layers import Lambda,Reshape,Concatenate,Add,Permute,TimeDistributed
from keras.models import Model
from keras.layers.convolutional import Conv2D
import PPT3_network.metrics as metrics

# Relu-BN-Conv2D 3x3
def conv_unit0(Fin,Fout,drop,H,W):
    unit_input=Input(shape=(Fin,H,W))
    
    unit_conv=Activation('relu')(unit_input)
    unit_conv=BatchNormalization()(unit_conv)
    unit_conv=Dropout(drop)(unit_conv)
    unit_output=Conv2D(filters=Fout,kernel_size=(3,3),padding="same")(unit_conv)
    unit_model=Model(inputs=unit_input,outputs=unit_output)
    print('kernel=(3,3)')
    return unit_model

# Relu-BN-Conv2D 1x1
def conv_unit1(Fin,Fout,drop,H,W):
    unit_input=Input(shape=(Fin,H,W))
    
    unit_conv=Activation('relu')(unit_input)
    unit_conv=BatchNormalization()(unit_conv)
    unit_conv=Dropout(drop)(unit_conv)
    unit_output=Conv2D(filters=Fout,kernel_size=(1,1),padding="same")(unit_conv)
    unit_model=Model(inputs=unit_input,outputs=unit_output)
    print('kernel=(1,1)')
    return unit_model

# efficient version of Res_plus
def Res_plus_E(name,F,Fplus,rate,drop,H,W):
    cl_input=Input(shape=(F,H,W))

    #normal channels
    cl_conv1A=conv_unit0(F,F-Fplus,drop,H,W)(cl_input)

    #separated channels
    if rate == 1:
        cl_conv1B=cl_input
    if rate !=1:
        cl_conv1B=AveragePooling2D(pool_size=(rate,rate),strides=(rate,rate),padding="valid")(cl_input)

    HR,WR=int(np.floor(H/rate)),int(np.floor(W/rate))
    cl_conv1B=Activation('relu')(cl_conv1B)
    cl_conv1B=BatchNormalization()(cl_conv1B)

    cl_conv1B=Conv2D(filters=Fplus,kernel_size=(1,1),use_bias=False,padding="same")(cl_conv1B)
    cl_conv1B=Reshape((Fplus,1,HR,WR),input_shape=(Fplus,HR,WR))(cl_conv1B)
    attention=Conv2D(filters=H*W,kernel_size=(HR,WR),use_bias=False,W_constraint=nonneg(),padding="valid")
    cl_conv1B=TimeDistributed(attention)(cl_conv1B)
    cl_conv1B=Reshape((Fplus,H,W),input_shape=(Fplus,H*W,1,1))(cl_conv1B)

    #merge
    cl_conv1=Concatenate(axis=1)([cl_conv1A,cl_conv1B])

    cl_conv2=conv_unit0(F,F,drop,H,W)(cl_conv1)

    cl_out=Add()([cl_input,cl_conv2])

    cl_model=Model(inputs=cl_input,outputs=cl_out,name=name)

    return cl_model

# new resdual block
def Res_plus(name,F,Fplus,rate,drop,H,W):
    cl_input=Input(shape=(F,H,W))
    
    cl_conv1A=conv_unit0(F,F-Fplus,drop,H,W)(cl_input)
    
    if rate == 1:
        cl_conv1B=cl_input
    if rate !=1:
        cl_conv1B=AveragePooling2D(pool_size=(rate,rate),strides=(rate,rate),padding="valid")(cl_input)
   
    cl_conv1B=Activation('relu')(cl_conv1B)
    cl_conv1B=BatchNormalization()(cl_conv1B) 
    
    plus_conv=Conv2D(filters=Fplus*H*W,kernel_size=(int(np.floor(H/rate)),int(np.floor(W/rate))),padding="valid")

    cl_conv1B=plus_conv(cl_conv1B)

    cl_conv1B=Reshape((Fplus,H,W))(cl_conv1B)

    cl_conv1=Concatenate(axis=1)([cl_conv1A,cl_conv1B])
    
    cl_conv2=conv_unit0(F,F,drop,H,W)(cl_conv1)
    
    cl_out=Add()([cl_input,cl_conv2])
    
    cl_model=Model(inputs=cl_input,outputs=cl_out,name=name)

    return cl_model

# normal residual block
def Res_normal(name,F,drop,H,W):
    cl_input=Input(shape=(F,H,W))
    
    cl_conv1=conv_unit0(F,F,drop,H,W)(cl_input)
   
    cl_conv2=conv_unit0(F,F,drop,H,W)(cl_conv1)
    
    cl_out=Add()([cl_input,cl_conv2])
    
    cl_model=Model(inputs=cl_input,outputs=cl_out,name=name)

    return cl_model
    
def cpt_slice(x, h1, h2):  
    return x[:,h1:h2,:,:]  

# transfer Time vector to a number (e.g. corresponding to filters = 1 in Conv2D)  
def T_trans(T,T_F,H,W):

    T_in=Input(shape=(T+7,H,W))
    T_mid=Conv2D(filters=T_F,kernel_size=(1,1),padding="same")(T_in)
    T_act=Activation('relu')(T_mid)
    T_fin=Conv2D(filters=1,kernel_size=(1,1),padding="same")(T_act)
    T_mul=Activation('relu')(T_fin)
    T_model=Model(inputs=T_in,outputs=T_mul)

    return T_model    
  
# transfer Time vector and PoI matrix to time-weighted PoI matrix    
def PT_trans(name,P_N,PT_F,T,T_F,H,W,isPT_F):
    if 1:
        poi_in=Input(shape=(P_N,H,W))
        # T_times/day + 7days/week 
        time_in=Input(shape=(T+7,H,W))

        if P_N>=2:
            T_x0 =T_trans(T,T_F,H,W)(time_in)
            T_x1 =T_trans(T,T_F,H,W)(time_in)
        if P_N>=3:
            T_x2 =T_trans(T,T_F,H,W)(time_in)
        if P_N>=4:
            T_x3 =T_trans(T,T_F,H,W)(time_in)
        if P_N>=5:
            T_x4 =T_trans(T,T_F,H,W)(time_in)
        if P_N>=6:
            T_x5 =T_trans(T,T_F,H,W)(time_in)
        if P_N>=7:
            T_x6 =T_trans(T,T_F,H,W)(time_in)
        if P_N>=8:
            T_x7 =T_trans(T,T_F,H,W)(time_in)
        if P_N>=9:
            T_x8 =T_trans(T,T_F,H,W)(time_in)
        if P_N>=10:
            T_x9 =T_trans(T,T_F,H,W)(time_in)
        if P_N>=11:
            T_x10=T_trans(T,T_F,H,W)(time_in)
        if P_N>=12:
            T_x11=T_trans(T,T_F,H,W)(time_in)
        if P_N>=13:
            T_x12=T_trans(T,T_F,H,W)(time_in)
        if P_N>=14:
            T_x13=T_trans(T,T_F,H,W)(time_in)
        if P_N>=15:
            T_x14=T_trans(T,T_F,H,W)(time_in)
        if P_N>=16:
            T_x15=T_trans(T,T_F,H,W)(time_in)
         
        
        if P_N==1:
            T_x=T_trans(T,T_F,H,W)(time_in)
        if P_N==2:
            T_x=Concatenate(axis=1)([T_x0,T_x1])
        if P_N==3:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2])
        if P_N==4:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3])
        if P_N==5:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4])
        if P_N==6:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5])
        if P_N==7:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6])
        if P_N==8:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6,T_x7])
        if P_N==9:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6,T_x7,T_x8])
        if P_N==10:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6,T_x7,T_x8,T_x9])
        if P_N==11:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6,T_x7,T_x8,T_x9,T_x10])
        if P_N==12:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6,T_x7,T_x8,T_x9,T_x10,T_x11])
        if P_N==13:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6,T_x7,T_x8,T_x9,T_x10,T_x11,T_x12])
        if P_N==14:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6,T_x7,T_x8,T_x9,T_x10,T_x11,T_x12,T_x13])
        if P_N==15:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6,T_x7,T_x8,T_x9,T_x10,T_x11,T_x12,T_x13,T_x14])
        if P_N==16:
            T_x=Concatenate(axis=1)([T_x0,T_x1,T_x2,T_x3,T_x4,T_x5,T_x6,T_x7,T_x8,T_x9,T_x10,T_x11,T_x12,T_x13,T_x14,T_x15])
            
        poi_time=Multiply()([poi_in,T_x])
        if isPT_F:
            poi_time=Conv2D(filters=PT_F,kernel_size=(1,1),padding="same")(poi_time)
            print('PT_F = YES')
        else:
            print('PT_F = NO')
        PT_model=Model(inputs=[poi_in,time_in],outputs=poi_time,name=name)

        return PT_model

# main model
def DeepSTN(H=21,W=12,channel=2,      #H-map_height W-map_width channel-map_channel
            c=3,p=4,t=4,              #c-closeness p-period t-trend
            pre_F=64,conv_F=64,R_N=2, #pre_F-prepare_conv_featrue conv_F-resnet_conv_featrue R_N-resnet_number
            is_plus=True,             #use ResPlus or mornal convolution
            is_plus_efficient=False,  #use the efficient version of ResPlus
            plus=8,rate=2,            #rate-pooling_rate
            is_pt=True,               #use PoI and Time or not
            P_N=6,T_F=28,PT_F=6,T=24, #P_N-poi_number T_F-time_feature PT_F-poi_time_feature T-T_times/day 
            drop=0,
            is_summary=True, #show detail
            lr=0.0002,
            kernel1=1, #kernel1 decides whether early-fusion uses conv_unit0 or conv_unit1, 1 recommended
            isPT_F=1): #isPT_F decides whether PT_model uses one more Conv after multiplying PoI and Time, 1 recommended
    
    all_channel = channel * (c+p+t)
            
    cut0 = int( 0 )
    cut1 = int( cut0 + channel*c )
    cut2 = int( cut1 + channel*p )
    cut3 = int( cut2 + channel*t )
       
    cpt_input=Input(shape=(all_channel,H,W))
            
    c_input=Lambda(cpt_slice,arguments={'h1':cut0,'h2':cut1})(cpt_input)
    p_input=Lambda(cpt_slice,arguments={'h1':cut1,'h2':cut2})(cpt_input)
    t_input=Lambda(cpt_slice,arguments={'h1':cut2,'h2':cut3})(cpt_input)
    
    c_out1=Conv2D(filters=pre_F,kernel_size=(1,1),padding="same")(c_input)
    p_out1=Conv2D(filters=pre_F,kernel_size=(1,1),padding="same")(p_input)
    t_out1=Conv2D(filters=pre_F,kernel_size=(1,1),padding="same")(t_input)
    
    if is_pt:
        poi_in=Input(shape=(P_N,H,W))
        # T_times/day + 7days/week 
        time_in=Input(shape=(T+7,H,W))

        PT_model=PT_trans('PT_trans',P_N,PT_F,T,T_F,H,W,isPT_F)
        
        poi_time=PT_model([poi_in,time_in])
 
        cpt_con1=Concatenate(axis=1)([c_out1,p_out1,t_out1,poi_time])
        if kernel1:
            cpt=conv_unit1(pre_F*3+PT_F*isPT_F+P_N*(not isPT_F),conv_F,drop,H,W)(cpt_con1)
        else:
            cpt=conv_unit0(pre_F*3+PT_F*isPT_F+P_N*(not isPT_F),conv_F,drop,H,W)(cpt_con1)
    
    else:
        cpt_con1=Concatenate(axis=1)([c_out1,p_out1,t_out1])
        if kernel1:
            cpt=conv_unit1(pre_F*3,conv_F,drop,H,W)(cpt_con1)
        else:
            cpt=conv_unit0(pre_F*3,conv_F,drop,H,W)(cpt_con1)  
     
    if is_plus:
        if is_plus_efficient:
            for i in range(R_N):
                cpt=Res_plus_E('Res_plus_'+str(i+1),conv_F,plus,rate,drop,H,W)(cpt)
        else:
            for i in range(R_N):
                cpt=Res_plus('Res_plus_'+str(i+1),conv_F,plus,rate,drop,H,W)(cpt)

    else:  
        for i in range(R_N):
            cpt=Res_normal('Res_normal_'+str(i+1),conv_F,drop,H,W)(cpt)

    cpt_conv2=Activation('relu')(cpt)
    cpt_out2=BatchNormalization()(cpt_conv2)
    cpt_conv1=Dropout(drop)(cpt_out2)
    cpt_conv1=Conv2D(filters=channel,kernel_size=(1, 1),padding="same")(cpt_conv1)
    cpt_out1=Activation('tanh')(cpt_conv1)
            
    if is_pt:
        DeepSTN_model=Model(inputs=[cpt_input,poi_in,time_in],outputs=cpt_out1)
    else:
        DeepSTN_model=Model(inputs=cpt_input,outputs=cpt_out1)

    DeepSTN_model.compile(loss='mse', optimizer=Adam(lr), metrics=[metrics.rmse,metrics.mae])
    
    if is_summary:
        DeepSTN_model.summary()

            
    print('***** pre_F : ',pre_F       )
    print('***** conv_F: ',conv_F      )
    print('***** R_N   : ',R_N         )
    
    print('***** plus  : ',plus*is_plus)
    print('***** rate  : ',rate*is_plus)
    
    print('***** P_N   : ',P_N*is_pt   )
    print('***** T_F   : ',T_F*is_pt   )
    print('***** PT_F  : ',PT_F*is_pt*isPT_F )            
    print('***** T     : ',T           ) 
    
    print('***** drop  : ',drop        )
     
    return DeepSTN_model