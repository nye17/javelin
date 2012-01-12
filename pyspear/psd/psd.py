########################################################################
# program: psd.py
# author: Tom Irvine
# version: 1.4
# date: July 19, 2011
# description:  
#    
#  Determine the power spectral density of a signal.
#  The file must have two columns: time(sec) & amplitude.
#              
########################################################################

from tompy import read_two_columns,signal_stats,sample_rate_check
from tompy import GetInteger2,WriteData2
from tompy import time_history_plot

from sys import stdin
from math import sqrt,pi,log
from numpy import zeros,argmax,linspace,cos
from scipy.fftpack import fft

import matplotlib.pyplot as plt

########################################################################
   
def GetString():
    iflag=0        
    while iflag==0:
        try:
            s=stdin.readline()
            iflag=1
        except ValueError:
            print 'Invalid String'
    return s   

########################################################################

def magnitude_resolve(mmm,mH,Y):
#
    mHm1=mH-1
    z=zeros(mH,'f')
    mag_seg=zeros(mH,'f')
#    
#     for i in range (0,mH):
#       z[i]=sqrt(Y.real[i]**2+Y.imag[i]**2)
#    
    z=abs(Y)/float(mmm)
#
    mag_seg[0]=z[0]**2
#
    mag_seg[1:mHm1]=((2*z[1:mHm1])**2)/2  
#
    return mag_seg   
    
########################################################################  
  
def Hanning_initial(mmm):
    H=zeros(mmm,'f')
    tpi=2*pi    
    alpha=linspace(0,tpi,mmm)
    ae=sqrt(8./3.)
    H=ae*0.5*(1.-cos(alpha))                
    return H
  
########################################################################    

print " The input file must have two columns: time(sec) & amplitude "

a,b,num =read_two_columns()

sr,dt,mean,sd,rms,skew,kurtosis,dur=signal_stats(a,b,num)

sr,dt=sample_rate_check(a,b,num,sr,dt)

########################################################################

print " "
print " Remove mean:  1=yes  2=no "

mr_choice = GetInteger2()

print " "
print " Select Window: 1=Rectangular 2=Hanning "

h_choice = GetInteger2()

########################################################################

n=num

ss=zeros(n)
seg=zeros(n,'f')
i_seg=zeros(n)

NC=0;
for i in range (0,1000):
    nmp = 2**(i-1)
    if(nmp <= n ):
        ss[i] = 2**(i-1)
        seg[i] =float(n)/float(ss[i])
        i_seg[i] = int(seg[i])
        NC=NC+1
    else:
        break

print ' '
print ' Number of   Samples per   Time per        df    '
print ' Segments     Segment      Segment(sec)   (Hz)   dof'
   
for i in range (1,NC+1):
    j=NC+1-i
    if j>0:
        if( i_seg[j]>0 ):
            tseg=dt*ss[j]
            ddf=1./tseg
            print '%8d \t %8d \t %10.3g  %10.3g    %d' %(i_seg[j],ss[j],tseg,ddf,2*i_seg[j])
    if(i==12):
        break;

ijk=0
while ijk==0:
    print(' ')
    print(' Choose the Number of Segments:  ')
    s=stdin.readline()
    NW = int(s)
    for j in range (0,len(i_seg)):   
        if NW==i_seg[j]:
            ijk=1
            break

# check

mmm = 2**int(log(float(n)/float(NW))/log(2))

if h_choice==2:
    H=Hanning_initial(mmm)


df=1./(mmm*dt)

# begin overlap

mH=((mmm/2)-1)

print " "
print "     number of segments   NW= %d " %NW
print "       samples/segments  mmm= %d " %mmm
print " half samples/segment-1   mH=%d  " %mH
print " "
print "        df=%6.3f Hz" %df

full=zeros(mH,'f')
mag_seg=zeros(mH,'f') 

amp_seg=zeros(mmm,'f')

nov=0;
for ijk in range (1,2*NW):
        
    amp_seg[0:mmm]=b[(0+nov):(mmm+nov)]    

    nov=nov+int(mmm/2)
    
    if mr_choice==1:
        mean = sum(amp_seg)/float(mmm)
        amp_seg-=mean
        
    if h_choice==2:
        amp_seg*=H

    Y = fft(amp_seg)
 
    mag_seg = magnitude_resolve(mmm,mH,Y)    
   
    
    full+=mag_seg
    

den=df*(2*NW-1)

full/=den

ms=sum(full)

freq=zeros(mH,'f')

maxf=(mH-1)*df

freq=linspace(0,maxf,mH)
    
tempf=freq[0:mH-1]    
tempa=full[0:mH-1]
freq=tempf
full=tempa      
    
rms=sqrt(ms*df)
three_rms=3*rms

print " "
print " Overall RMS = %10.3g " % rms
print " Three Sigma = %10.3g " % three_rms

idx = argmax(full) 
    
print " "
print " Maximum:  Freq=%8.4g Hz   Amp=%8.4g " %(freq[idx],full[idx])

########################################################################

print " "
print " Write PSD data to file? 1=yes 2=no"
iacc=GetInteger2()

if(iacc==1):
    print " "
    print "Enter the output PSD filename: "
    output_file_path = stdin.readline()
    output_file = output_file_path.rstrip('\n')
    mH=len(freq)
    WriteData2(mH,freq,full,output_file)

########################################################################

pmin=10**40
pmax=10**-40

fmin=10**40
fmax=10**-40

for i in range (0, len(freq)):
    if full[i]>0 and freq[i]>0 and full[i]>pmax:
        pmax=full[i]
    if full[i]>0 and freq[i]>0 and full[i]<pmin:
        pmin=full[i]   
    if freq[i]>0 and freq[i]>fmax:
        fmax=freq[i]
    if freq[i]>0 and freq[i]<fmin:
        fmin=freq[i]          

xmax=10**-30
xmin=xmax

for i in range (-30,30):
    if(fmax<10**i):
        xmax=10**i
        break;
        
    
for i in range(30,-30,-1):
    if(fmin>10**i):
        xmin=10**i
        break;  
        
ymax=10**-30
ymin=ymax



for i in range (-30,30):
    if(pmax<10**i):
        ymax=10**i
        break;
    
for i in range(30,-30,-1):
    if(pmin>10**i):
        ymin=10**i
        break;
        

print " "
print " Is the input data dimension Accel(G) ?"
print " 1=yes  2=no"

ind=GetInteger2()

if(ind==1):
    th_label='Accel (G)'
    psd_label='Accel (G^2/Hz)'
else:
    print('Enter input amplitude unit ')    
    th_label=GetString()
    print('Enter PSD unit label, i.e.  unit^2/Hz')    
    psd_label=GetString()


print " "
print " view plots "

time_history_plot(a,b,1,'Time(sec)',th_label,'Time History','time_history')        
        
plt.gca().set_autoscale_on(False)

plt.figure(2)     
plt.plot(freq,full)
title_string='Power Spectral Density   '+str("%6.3g" %rms)+' GRMS Overall '
plt.title(title_string)
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.ylabel(psd_label)
plt.xlabel(' Frequency (Hz) ')
plt.grid(True)
plt.savefig('power_spectral_density')
plt.xscale('log')
plt.yscale('log')
plt.show()