import matplotlib.pyplot as plt
from math import ceil
import numpy as np

Norm_energy = {'MAC':1,'RF':1,'NoC':2,'GL_Buffer':6,'DRAM':200} 
# MAC energy consumption is normalized under 16-bit operation
PE = (12,14) # Defined by spatial arrays of eyeriss architecture
# pw = [0,0,0,0,0,0,0,0] # Pruning ratio of filter weights
pw = [0.625,0.375,0.125,0.195,0.121,0.008,0.234,0.4] # Pruning ratio of filter weights
# qw = [16,16,16,16,16,16,16,16] # qw represents quantization or bitwidth of weights
# qa = [16,16,16,16,16,16,16,16] # qa represents quantization or bitwidth of activations
qw = [5,3,3,6,4,4,7,4] # qw represents quantization or bitwidth of weights
qa = [5,6,5,7,4,7,7,6] # qa represents quantization or bitwidth of activations


class conv(object): # Convolution layer class
    # p represents pruning quantity or sparsity in each layer
    def __init__(self,size,channel,kernel,stride,padding,qw,qa,pw):
        self.size = size
        self.channel = channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.w_quant = qw
        self.a_quant = qa
        self.w_pruning = pw

    def ofmap(self,ifmap_size):
        ofmap_param = {}
        ofmap_size = (ifmap_size + 2 * self.padding - self.size) / self.stride + 1
        ofmap_param['size'] = int(ofmap_size)
        ofmap_param['channel'] = self.kernel
        return ofmap_param

    # Calculate the number of MAC operations, scaled by normalized energy comsumption per MAC.  
    def Comp_Cal(self,ofmap_size):
        self.MAC_Num = (self.size**2) * self.channel * self.kernel * (ofmap_size**2) * (1 - self.w_pruning)
        self.Comp_Energy = self.MAC_Num * Norm_energy['MAC'] * (self.w_quant / 16) * (self.a_quant / 16)
        return self.Comp_Energy

    # Calculate the number of data accesses at each memory hierarchy, scaled by corresponding energy. 
    # Here row stationary is applied, hence data reuse and partial summation 
    def Data_Cal(self,ifmap_size,ofmap_size):
        ifmap_size += 2 * self.padding
        replic = PE[0] // self.size           
        fold   = ceil(ofmap_size / PE[1])     
        # here compression technique for data access in pruning is not considered.
        # Energy consumption of weights
        DRAM_Read_Weight = (self.size**2) * self.channel * self.kernel * (1 - self.w_pruning)
        self.DRAM_Weight = DRAM_Read_Weight * Norm_energy['DRAM'] * (self.w_quant / 16)

        # Convolutional (filter weight) reuse in RS dataflow 
        Buffer_Read_Weight = (self.size**2) * self.channel * self.kernel * (1 - self.w_pruning)
        self.Buffer_Weight = Buffer_Read_Weight * Norm_energy['GL_Buffer'] * (self.w_quant / 16)

        if replic >= fold:  
            NoC_Read_Weight =  ofmap_size * self.size**2 * self.channel * self.kernel * (1 - self.w_pruning)
        else:
            NoC_Read_Weight = replic * PE[1] * self.size**2 * self.channel * self.kernel * (1 - self.w_pruning)
        self.NoC_Weight = NoC_Read_Weight * Norm_energy['NoC'] * (self.w_quant / 16)

        RF_Read_Weight = self.size**2 * ofmap_size**2 * self.channel * self.kernel * (1 - self.w_pruning)
        self.RF_Weight = RF_Read_Weight * Norm_energy['RF'] * (self.w_quant / 16)

        # Total energy consumption of weight access
        self.Total_Weight = self.DRAM_Weight + self.Buffer_Weight + self.NoC_Weight + self.RF_Weight


        # Energy consumption of input activations
        DRAM_Read_Activ = ifmap_size**2 * self.channel
        self.DRAM_Activ = DRAM_Read_Activ * Norm_energy['DRAM'] * (self.a_quant / 16)

        # Convolutional (input activation) reuse in RS dataflow
        if replic >= fold:
            Buffer_Read_Activ = ifmap_size**2 * self.channel
        else:
            Buffer_Read_Activ = ifmap_size**2 * self.channel * self.kernel
        self.Buffer_Activ = Buffer_Read_Activ * Norm_energy['GL_Buffer'] * (self.a_quant / 16)

        if replic < fold:
            NoC_Read_Activ = self.size * ofmap_size * ifmap_size * self.channel * self.kernel
        else:
            NoC_Read_Activ = (replic // fold) * self.size * ofmap_size * ifmap_size * self.channel
        self.NoC_Activ = NoC_Read_Activ * Norm_energy['NoC'] * (self.a_quant / 16)

        RF_Read_Activ = self.size * ofmap_size * ifmap_size * self.channel * self.kernel
        self.RF_Activ = RF_Read_Activ * Norm_energy['RF'] * (self.a_quant / 16)

        # Total energy consumption of input activation access
        self.Total_Activ = self.DRAM_Activ + self.Buffer_Activ + self.NoC_Activ + self.RF_Activ

        
        # Energy consumption of partial summations
        DRAM_Psum = ofmap_size**2 * self.kernel
        self.DRAM_Psum = DRAM_Psum * Norm_energy['DRAM'] * (self.a_quant / 16)

        Buffer_Psum = ofmap_size**2 * 2 * (self.channel - 1) * self.kernel 
        self.Buffer_Psum = Buffer_Psum * Norm_energy['GL_Buffer'] * (self.a_quant / 16)

        NoC_Psum = (self.size - 1) * ofmap_size**2 * self.channel * self.kernel
        self.NoC_Psum = NoC_Psum * Norm_energy['NoC'] * (self.a_quant / 16)

        RF_Psum = 2 * (self.size - 1) * self.size * ofmap_size**2 * self.channel * self.kernel 
        self.RF_Psum = RF_Psum * Norm_energy['RF'] * (self.a_quant / 16)
        
        # Total energy consumption of partial summation access 
        self.Total_Psum = self.DRAM_Psum + self.Buffer_Psum + self.NoC_Psum + self.RF_Psum
        
        # Total energy consumption of each memory hierarchy
        self.DRAM = self.DRAM_Weight + self.DRAM_Activ + self.DRAM_Psum
        self.Buffer = self.Buffer_Weight + self.Buffer_Activ + self.Buffer_Psum
        self.NoC = self.NoC_Weight + self.NoC_Activ + self.NoC_Psum
        self.RF = self.RF_Weight + self.RF_Activ + self.RF_Psum

        # Total energy consumption of memory access or data movement
        self.Total_Data = self.Total_Weight + self.Total_Activ + self.Total_Psum
        return self.Total_Data



class Maxpool(object): # Maxpooling layer class 
    def __init__(self,size,channel,stride,padding):
        self.size = size
        self.channel = channel
        self.stride = stride
        self.padding = padding

    def ofmap(self,ifmap_size):
        ofmap_param = {}
        ofmap_size = (ifmap_size + 2*self.padding - self.size) / self.stride + 1
        ofmap_param['size'] = int(ofmap_size)
        ofmap_param['channel'] = self.channel
        return ofmap_param



class FC(object): # Fully-connected layer class
    def __init__(self,size,channel,kernel,qw,qa,pw):
        self.size = size
        self.channel = channel
        self.kernel = kernel
        self.w_quant = qw
        self.a_quant = qa
        self.w_pruning = pw

    def ofmap(self):
        ofmap_param = {}
        ofmap_param['size'] = 1 # Ofmap of fully-connected layer must have a size of 1. 
        ofmap_param['channel'] = self.kernel
        return ofmap_param

    def Comp_Cal(self):
        self.MAC_Num = (self.size**2) * self.channel * self.kernel * (1 - self.w_pruning) 
        self.Comp_Energy = self.MAC_Num * Norm_energy['MAC'] * (self.w_quant / 16) * (self.a_quant / 16)
        return self.Comp_Energy
    
    def Data_Cal(self):
        replic = PE[0] // self.size           
        blocks = PE[1] * replic

        DRAM_Read_Weight = (self.size**2) * self.channel * self.kernel * (1 - self.w_pruning)
        self.DRAM_Weight = DRAM_Read_Weight * Norm_energy['DRAM'] * (self.w_quant / 16)

        Buffer_Read_Weight = (self.size**2) * self.channel * self.kernel * (1 - self.w_pruning)
        self.Buffer_Weight = Buffer_Read_Weight * Norm_energy['GL_Buffer'] * (self.w_quant / 16)

        NoC_Read_Weight = (self.size**2) * self.channel * self.kernel * (1 - self.w_pruning)
        self.NoC_Weight = NoC_Read_Weight * Norm_energy['NoC'] * (self.w_quant / 16)

        RF_Read_Weight = (self.size**2) * self.channel * self.kernel * (1 - self.w_pruning)
        self.RF_Weight = RF_Read_Weight * Norm_energy['RF'] * (self.w_quant / 16)

        # Total energy consumption of weight access
        self.Total_Weight = self.DRAM_Weight + self.Buffer_Weight + self.NoC_Weight + self.RF_Weight

        # Energy consumption of input activations
        DRAM_Read_Activ = self.size**2 * self.channel
        self.DRAM_Activ = DRAM_Read_Activ * Norm_energy['DRAM'] * (self.a_quant / 16)

        Buffer_Read_Activ = self.size**2 * self.channel
        self.Buffer_Activ = Buffer_Read_Activ * Norm_energy['GL_Buffer'] * (self.a_quant / 16)

        NoC_Read_Activ = self.size**2 * blocks * self.channel 
        self.NoC_Activ = NoC_Read_Activ * Norm_energy['NoC'] * (self.a_quant / 16)

        RF_Read_Activ = self.size**2 * self.kernel * self.channel
        self.RF_Activ = RF_Read_Activ * Norm_energy['RF'] * (self.a_quant / 16)

        # Total energy consumption of input activation access
        self.Total_Activ = self.DRAM_Activ + self.Buffer_Activ + self.NoC_Activ + self.RF_Activ

         # Energy consumption of partial summations
        DRAM_Psum = self.kernel
        self.DRAM_Psum = DRAM_Psum * Norm_energy['DRAM'] * (self.a_quant / 16)

        Buffer_Psum = 2 * (self.channel - 1) * self.kernel 
        self.Buffer_Psum = Buffer_Psum * Norm_energy['GL_Buffer'] * (self.a_quant / 16)

        NoC_Psum = (self.size - 1) * self.channel * self.kernel
        self.NoC_Psum = NoC_Psum * Norm_energy['NoC'] * (self.a_quant / 16)

        RF_Psum = 2 * (self.size - 1) * self.size * self.channel * self.kernel 
        self.RF_Psum = RF_Psum * Norm_energy['RF'] * (self.a_quant / 16)
        
        # Total energy consumption of partial summation access 
        self.Total_Psum = self.DRAM_Psum + self.Buffer_Psum + self.NoC_Psum + self.RF_Psum
        
        # Total energy consumption of each memory hierarchy
        self.DRAM = self.DRAM_Weight + self.DRAM_Activ + self.DRAM_Psum
        self.Buffer = self.Buffer_Weight + self.Buffer_Activ + self.Buffer_Psum
        self.NoC = self.NoC_Weight + self.NoC_Activ + self.NoC_Psum
        self.RF = self.RF_Weight + self.RF_Activ + self.RF_Psum

        # Total energy consumption of memory access or data movement
        self.Total_Data = self.Total_Weight + self.Total_Activ + self.Total_Psum
        return self.Total_Data



# Main function here
if __name__ == "__main__":
    Total_Energy = 0  # Total_Energy consumption per image processing.
    Total_Layer_Breakdown = []
    comp_value = []
    weight_value = []
    ifmap_value = []
    psum_value = []
    DRAM_value = []
    Buffer_value = []
    NoC_value = []
    RF_value = []

    conv1 = conv(11,3,96,4,0,qw[0],qa[0],pw[0])  # 96*11*11*3, p = 0, s = 4
    # Initial ifmap is 227*227*3 image.
    conv1_output = conv1.ofmap(227) 
    conv1_Comp_Energy = conv1.Comp_Cal(conv1_output['size'])
    comp_value.append(conv1_Comp_Energy)
    conv1_Data_Energy = conv1.Data_Cal(227,conv1_output['size'])
    weight_value.append(conv1.Total_Weight)
    ifmap_value.append(conv1.Total_Activ)
    psum_value.append(conv1.Total_Psum)
    DRAM_value.append(conv1.DRAM)
    Buffer_value.append(conv1.Buffer)
    NoC_value.append(conv1.NoC)
    RF_value.append(conv1.RF)
    conv1_Total_Energy = conv1_Comp_Energy + conv1_Data_Energy
    Total_Energy += conv1_Total_Energy
    Total_Layer_Breakdown.append(conv1_Total_Energy)
    print("\nconv1_ofmap =",conv1_output) # ofmap of conv1 is 55*55*96
    print("conv1's computation energy consumption =",conv1_Comp_Energy)
    print("conv1's data access energy consumption =",conv1_Data_Energy)
    # RELU_1 Activation function
    # LRN_1 Local Response Normalization

    Maxpool1 = Maxpool(3,conv1_output['channel'],2,0)
    Maxpool1_output = Maxpool1.ofmap(conv1_output['size']) 
    print("\nMaxpool1_ofmap =",Maxpool1_output) # ofmap of Maxpool1 is 27*27*96

    conv2 = conv(5,Maxpool1_output['channel'],256,1,2,qw[1],qa[1],pw[1])
    conv2_output = conv2.ofmap(Maxpool1_output['size'])
    conv2_Comp_Energy = conv2.Comp_Cal(conv2_output['size'])
    comp_value.append(conv2_Comp_Energy)
    conv2_Data_Energy = conv2.Data_Cal(Maxpool1_output['size'],conv2_output['size'])
    weight_value.append(conv2.Total_Weight)
    ifmap_value.append(conv2.Total_Activ)
    psum_value.append(conv2.Total_Psum)
    DRAM_value.append(conv2.DRAM)
    Buffer_value.append(conv2.Buffer)
    NoC_value.append(conv2.NoC)
    RF_value.append(conv2.RF)
    conv2_Total_Energy = conv2_Comp_Energy + conv2_Data_Energy
    Total_Energy += conv2_Total_Energy
    Total_Layer_Breakdown.append(conv2_Total_Energy)
    print("\nconv2_ofmap =",conv2_output) # ofmap of conv2 is 27*27*256
    print("conv2's computation energy consumption =",conv2_Comp_Energy)
    print("conv2's data access energy consumption =",conv2_Data_Energy)
    # RELU_2 Activation function
    # LRN_2 Local Response Normalization

    Maxpool2 = Maxpool(3,conv2_output['channel'],2,0)
    Maxpool2_output = Maxpool2.ofmap(conv2_output['size']) 
    print("\nMaxpool2_ofmap =",Maxpool2_output) # ofmap of Maxpool2 is 13*13*256

    conv3 = conv(3,Maxpool2_output['channel'],384,1,1,qw[2],qa[2],pw[2])
    conv3_output = conv3.ofmap(Maxpool2_output['size']) 
    conv3_Comp_Energy = conv3.Comp_Cal(conv3_output['size'])
    comp_value.append(conv3_Comp_Energy)
    conv3_Data_Energy = conv3.Data_Cal(Maxpool2_output['size'],conv3_output['size'])
    weight_value.append(conv3.Total_Weight)
    ifmap_value.append(conv3.Total_Activ)
    psum_value.append(conv3.Total_Psum)
    DRAM_value.append(conv3.DRAM)
    Buffer_value.append(conv3.Buffer)
    NoC_value.append(conv3.NoC)
    RF_value.append(conv3.RF)
    conv3_Total_Energy = conv3_Comp_Energy + conv3_Data_Energy
    Total_Energy += conv3_Total_Energy
    Total_Layer_Breakdown.append(conv3_Total_Energy)
    print("\nconv3_ofmap =",conv3_output) # ofmap of conv3 is 13*13*384
    print("conv3's computation energy consumption =",conv3_Comp_Energy)
    print("conv3's data access energy consumption =",conv3_Data_Energy)
    # RELU_3 Activation function

    conv4 = conv(3,conv3_output['channel'],384,1,1,qw[3],qa[3],pw[3])
    conv4_output = conv4.ofmap(conv3_output['size']) 
    conv4_Comp_Energy = conv4.Comp_Cal(conv4_output['size'])
    comp_value.append(conv4_Comp_Energy)
    conv4_Data_Energy = conv4.Data_Cal(conv3_output['size'],conv4_output['size'])
    weight_value.append(conv4.Total_Weight)
    ifmap_value.append(conv4.Total_Activ)
    psum_value.append(conv4.Total_Psum)
    DRAM_value.append(conv4.DRAM)
    Buffer_value.append(conv4.Buffer)
    NoC_value.append(conv4.NoC)
    RF_value.append(conv4.RF)
    conv4_Total_Energy = conv4_Comp_Energy + conv4_Data_Energy
    Total_Energy += conv4_Total_Energy
    Total_Layer_Breakdown.append(conv4_Total_Energy)
    print("\nconv4_ofmap =",conv4_output) # ofmap of conv4 is 13*13*384
    print("conv4's computation energy consumption =",conv4_Comp_Energy)
    print("conv4's data access energy consumption =",conv4_Data_Energy)
    # RELU_4 Activation function

    conv5 = conv(3,conv4_output['channel'],256,1,1,qw[4],qa[4],pw[4])
    conv5_output = conv5.ofmap(conv4_output['size']) 
    conv5_Comp_Energy = conv5.Comp_Cal(conv5_output['size'])
    comp_value.append(conv5_Comp_Energy)
    conv5_Data_Energy = conv5.Data_Cal(conv4_output['size'],conv5_output['size'])
    weight_value.append(conv5.Total_Weight)
    ifmap_value.append(conv5.Total_Activ)
    psum_value.append(conv5.Total_Psum)
    DRAM_value.append(conv5.DRAM)
    Buffer_value.append(conv5.Buffer)
    NoC_value.append(conv5.NoC)
    RF_value.append(conv5.RF)
    conv5_Total_Energy = conv5_Comp_Energy + conv5_Data_Energy
    Total_Energy += conv5_Total_Energy
    Total_Layer_Breakdown.append(conv5_Total_Energy)
    print("\nconv5_ofmap =",conv5_output) # ofmap of conv5 is 13*13*256
    print("conv5's computation energy consumption =",conv5_Comp_Energy)
    print("conv5's data access energy consumption =",conv5_Data_Energy)
    # RELU_5 Activation function

    Maxpool3 = Maxpool(3,conv5_output['channel'],2,0)
    Maxpool3_output = Maxpool3.ofmap(conv5_output['size']) 
    print("\nMaxpool3_ofmap =",Maxpool3_output) # ofmap of Maxpool3 is 6*6*256

    FC1 = FC(Maxpool3_output['size'],Maxpool3_output['channel'],4096,qw[5],qa[5],pw[5])
    FC1_output = FC1.ofmap() 
    FC1_Comp_Energy = FC1.Comp_Cal()
    comp_value.append(FC1_Comp_Energy)
    FC1_Data_Energy = FC1.Data_Cal()
    weight_value.append(FC1.Total_Weight)
    ifmap_value.append(FC1.Total_Activ)
    psum_value.append(FC1.Total_Psum)
    DRAM_value.append(FC1.DRAM)
    Buffer_value.append(FC1.Buffer)
    NoC_value.append(FC1.NoC)
    RF_value.append(FC1.RF)
    FC1_Total_Energy = FC1_Comp_Energy + FC1_Data_Energy
    Total_Energy += FC1_Total_Energy
    Total_Layer_Breakdown.append(FC1_Total_Energy)
    print("\nFC1_ofmap =",FC1_output) # ofmap of FC1 is 1*1*4096
    print("FC1's computation energy consumption =",FC1_Comp_Energy)
    print("FC1's data access energy consumption =",FC1_Data_Energy)
    # RELU_6 Activation function
    # dropout = P = 0.5

    FC2 = FC(FC1_output['size'],FC1_output['channel'],4096,qw[6],qa[6],pw[6])
    FC2_output = FC2.ofmap() 
    FC2_Comp_Energy = FC2.Comp_Cal()
    comp_value.append(FC2_Comp_Energy)
    FC2_Data_Energy = FC2.Data_Cal()
    weight_value.append(FC2.Total_Weight)
    ifmap_value.append(FC2.Total_Activ)
    psum_value.append(FC2.Total_Psum)
    DRAM_value.append(FC2.DRAM)
    Buffer_value.append(FC2.Buffer)
    NoC_value.append(FC2.NoC)
    RF_value.append(FC2.RF)
    FC2_Total_Energy = FC2_Comp_Energy + FC2_Data_Energy
    Total_Energy += FC2_Total_Energy
    Total_Layer_Breakdown.append(FC2_Total_Energy)
    print("\nFC2_ofmap =",FC2_output) # ofmap of FC2 is 1*1*4096
    print("FC2's computation energy consumption =",FC2_Comp_Energy)
    print("FC2's data access energy consumption =",FC2_Data_Energy)
    # RELU_7 Activation function
    # dropout = P = 0.5

    FC3 = FC(FC2_output['size'],FC2_output['channel'],1000,qw[7],qa[7],pw[7])
    FC3_output = FC3.ofmap() 
    FC3_Comp_Energy = FC3.Comp_Cal()
    comp_value.append(FC3_Comp_Energy)
    FC3_Data_Energy = FC3.Data_Cal()
    weight_value.append(FC3.Total_Weight)
    ifmap_value.append(FC3.Total_Activ)
    psum_value.append(FC3.Total_Psum)
    DRAM_value.append(FC3.DRAM)
    Buffer_value.append(FC3.Buffer)
    NoC_value.append(FC3.NoC)
    RF_value.append(FC3.RF)
    FC3_Total_Energy = FC3_Comp_Energy + FC3_Data_Energy
    Total_Energy += FC3_Total_Energy
    Total_Layer_Breakdown.append(FC3_Total_Energy)
    print("\nFC3_ofmap =",FC3_output) # ofmap of FC3 is 1*1*1000
    print("FC3's computation energy consumption =",FC3_Comp_Energy)
    print("FC3's data access energy consumption =",FC3_Data_Energy)
    # Softmax function
    # Classification result     

    # Print out the total energy consumption per image processing.
    print("\nTotal Energy Consumption per image =",Total_Energy) 

    # Draw the related histograms and pie charts.
    hist_name = ['conv1','conv2','conv3','conv4','con5','FC1','FC2','FC3']

    itm1 = []
    itm2 = []
    for i in range(len(hist_name)):
        itm1.append(psum_value[i]+ifmap_value[i])
    for i in range(len(hist_name)):
        itm2.append(itm1[i]+weight_value[i])

    plt.figure(1)
    plt.bar(range(len(ifmap_value)),ifmap_value,label='Ifmap',fc='r')
    plt.bar(range(len(psum_value)),psum_value,label='Psum',fc='g',bottom=ifmap_value)
    plt.bar(range(len(weight_value)),weight_value,label='Weights',fc='b',bottom=itm1)
    plt.bar(range(len(comp_value)),comp_value,label='Computation',fc='y',bottom=itm2,tick_label=hist_name)
    plt.title('Energy breakdown for each layer based on computation and memory access')
    plt.xlabel('AlexNet layer')
    plt.ylabel('Normalized Energy Consumption')
    plt.legend()

    Total_DRAM = Total_Buffer = Total_NoC = Total_RF = 0
    Total_MAC = Total_Weight = Total_ifmap = Total_Psum = 0
    for i in range(len(hist_name)):
        Total_DRAM += DRAM_value[i]
        Total_Buffer += Buffer_value[i]
        Total_NoC += NoC_value[i]
        Total_RF += RF_value[i]
        Total_MAC += comp_value[i]
        Total_Weight += weight_value[i]
        Total_ifmap += ifmap_value[i]
        Total_Psum += psum_value[i]
    Total_Memory_Breakdown = [Total_DRAM,Total_Buffer,Total_NoC,Total_RF]
    Total_Datatype_Breakdown = [Total_MAC,Total_Weight,Total_ifmap,Total_Psum]

    itm3 = []
    itm4 = []
    for i in range(len(hist_name)):
        itm3.append(DRAM_value[i]+Buffer_value[i])
    for i in range(len(hist_name)):
        itm4.append(itm3[i]+NoC_value[i])

    plt.figure(2)
    plt.bar(range(len(DRAM_value)),DRAM_value,label='DRAM',fc='r')
    plt.bar(range(len(Buffer_value)),Buffer_value,label='Global Buffer',fc='g',bottom=DRAM_value)
    plt.bar(range(len(NoC_value)),NoC_value,label='NoC',fc='b',bottom=itm3)
    plt.bar(range(len(RF_value)),RF_value,label='Register File',fc='y',bottom=itm4,tick_label=hist_name)
    plt.title('Energy breakdown for each layer based on memory hierarchy')
    plt.xlabel('AlexNet layer')
    plt.ylabel('Normalized Energy Consumption')
    plt.legend()

    plt.figure(3)
    plt.pie(Total_Memory_Breakdown,labels=['DRAM','Buffer','NoC','RF'],autopct='%.2f%%')
    plt.title('Total Energy breakdown for each memory hierarchy')
    plt.legend()

    plt.figure(4)
    plt.pie(Total_Datatype_Breakdown,labels=['Computation','Weights','Ifmap','Psum'],autopct='%.2f%%')
    plt.title('Total Energy breakdown for computation and memory access')
    plt.legend()

    plt.figure(5)
    plt.pie(Total_Layer_Breakdown,labels=hist_name,autopct='%.2f%%')
    plt.title('Total Energy breakdown for each layer')
    plt.legend()
    plt.show()

