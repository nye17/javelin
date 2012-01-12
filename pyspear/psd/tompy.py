###########################################################################
# program: tompy.py
# author: Tom Irvine
# Email: tomirvine@aol.com
# version: 2.1
# date: December 15, 2011
# description:  utility functions for use in other scripts
#
###########################################################################
import os
import re
import numpy as np

from sys import stdin

from scipy import stats

from math import pi,sqrt


import matplotlib.pyplot as plt

###########################################################################

def read_array(label_name):
    """
    Read a 2D array.
    """
    while(1):
        print " "
        print "Enter the %s matrix filename: " %label_name
        input_file_path =stdin.readline()
        file_path = input_file_path.rstrip('\n')
#
        if not os.path.exists(file_path):
            print "This file doesn't exist"
 #
        if os.path.exists(file_path):
            print "This file exists. Reading..."
            print " "
            read_data = np.loadtxt(file_path)
            break
    return read_data


def read_three_columns():
    """
    Prompt the user for the input filename.
    The input file must have three columns.
    The input file may have an arbitrary number of header and blank lines.
    Return the three columns as arrays a, b & c, respectively.
    Return the total numbers of lines as num.
    """
    while(1):
        print(" ")
        print("Enter the input filename: ")
        input_file_path =stdin.readline()
        file_path = input_file_path.rstrip('\n')
#
        if not os.path.exists(file_path):
            print "This file doesn't exist"
 #
        if os.path.exists(file_path):
            print "This file exists"
            print " "
            infile = open(file_path,"rb")
            lines = infile.readlines()
            infile.close()

            a = []
            b = []
            c = []
            num=0
            for line in lines:
#
                if re.search(r"(\d+)", line):  # matches a digit
                    iflag=0
                else:
                    iflag=1 # did not find digit
#
                if re.search(r"#", line):
                    iflag=1
#
                if iflag==0:
                    line=line.lower()
                    if re.search(r"([a-d])([f-z])", line):  # ignore header lines
                        iflag=1
                    else:
                        line = line.replace(","," ")
                        col1,col2,col3=line.split()
                        a.append(float(col1))
                        b.append(float(col2))
                        c.append(float(col3))
                        num=num+1
            break;

            a=np.array(a)
            b=np.array(b)
            c=np.array(c)

            print "\n samples = %d " % num
    return a,b,c,num

###########################################################################

def read_two_columns():
    """
    Prompt the user for the input filename.
    The input file must have two columns.
    The input file may have an arbitrary number of header and blank lines.
    Return the first & second columns as arrays a & b, respectively.
    Return the total numbers of lines as num.
    """
    while(1):
        print(" ")
        print("Enter the input filename: ")
        input_file_path =stdin.readline()
        file_path = input_file_path.rstrip('\n')
#
        if not os.path.exists(file_path):
            print "This file doesn't exist"
 #
        if os.path.exists(file_path):
            print "This file exists"
            print " "
            infile = open(file_path,"rb")
            lines = infile.readlines()
            infile.close()

            a = []
            b = []
            num=0
            for line in lines:
#
                if re.search(r"(\d+)", line):  # matches a digit
                    iflag=0
                else:
                    iflag=1 # did not find digit
#
                if re.search(r"#", line):
                    iflag=1
#
                if iflag==0:
                    line=line.lower()
                    if re.search(r"([a-d])([f-z])", line):  # ignore header lines
                        iflag=1
                    else:
                        line = line.replace(","," ")
                        col1,col2=line.split()[0:2]
                        a.append(float(col1))
                        b.append(float(col2))
                        num=num+1
            break;

            a=np.array(a)
            b=np.array(b)

            print "\n samples = %d " % num
    return a,b,num

###########################################################################

def read_one_column():
    """
    Prompt the user for the input filename.
    The input file must have one column.
    The input file may have an arbitrary number of header and blank lines.
    Return the column as array b.
    Return the total numbers of lines as num.
    """
    while(1):
        print(" ")
        print("Enter the input filename: ")
        input_file_path =stdin.readline()
        file_path = input_file_path.rstrip('\n')
#
        if not os.path.exists(file_path):
            print "This file doesn't exist"
 #
        if os.path.exists(file_path):
            print "This file exists"
            print " "
            infile = open(file_path,"rb")
            lines = infile.readlines()
            infile.close()

            b = []
            num=0
            for line in lines:
#
                if re.search(r"(\d+)", line):  # matches a digit
                    iflag=0
                else:
                    iflag=1 # did not find digit
#
                if re.search(r"#", line):
                    iflag=1
#
                if iflag==0:
                    line=line.lower()
                    if re.search(r"([a-d])([f-z])", line):  # ignore header lines
                        iflag=1
                    else:
                        line = line.replace(","," ")
                        b.append(float(line))
                        num=num+1
            break;

            b=np.array(b)

            print "\n samples = %d " % num
    return b,num

###########################################################################

def signal_stats(a,b,num):
    """
    a is the time column.
    b is the amplitude column.
    num is the number of coordinates
    Return
          sr - sample rate
          dt - time step
        mean - average
          sd - standard deviation
         rms - root mean square
        skew - skewness
    kurtosis - peakedness
         dur - duration
    """
    bmax=max(b)
    bmin=min(b)

    ave = np.mean(b)

    dur = a[num-1]-a[0];

    dt=dur/float(num-1)
    sr=1/dt


    rms=np.sqrt(np.var(b))
    sd=np.std(b)

    skewness=stats.skew(b)
    kurtosis=stats.kurtosis(b,fisher=False)

    print "\n max = %8.4g  min=%8.4g \n" % (bmax,bmin)

    print "     mean = %8.4g " % ave
    print "  std dev = %8.4g " % sd
    print "      rms = %8.4g " % rms
    print " skewness = %8.4g " % skewness
    print " kurtosis = %8.4g " % kurtosis

    print "\n  start = %8.4g sec  end = %8.4g sec" % (a[0],a[num-1])
    print "    dur = %8.4g sec \n" % dur
    return sr,dt,ave,sd,rms,skewness,kurtosis,dur

###########################################################################

def squareEach(input_matrix):
    """
    input_matrix is a 1-D array.
    Return: sumMatrix is the sum of the squares
    """
    matrix_sq=[i * i for i in input_matrix]
    sumMatrix=sum(matrix_sq)
    return sumMatrix

########################################################################

def cubeEach(input_matrix):
    """
    input_matrix is a 1-D array.
    Return: sumMatrix is the sum of the cubes
    """
    matrix_3=[i**3 for i in input_matrix]
    sumMatrix=sum(matrix_3)
    return sumMatrix

########################################################################

def quadEach(input_matrix):
    """
    input_matrix is a 1-D array.
    Return: sumMatrix is the sum of the quads
    """
    matrix_4=[i**4 for i in input_matrix]
    sumMatrix=sum(matrix_4)
    return sumMatrix

########################################################################

def sample_rate_check(a,b,num,sr,dt):
    dtmin=1e+50
    dtmax=0

    for i in range(1, num-1):
        if (a[i]-a[i-1])<dtmin:
            dtmin=a[i]-a[i-1];
            if (a[i]-a[i-1])>dtmax:
                dtmax=a[i]-a[i-1];

    print "  dtmin = %8.4g sec" % dtmin
    print "     dt = %8.4g sec" % dt
    print "  dtmax = %8.4g sec \n" % dtmax

    srmax=float(1/dtmin)
    srmin=float(1/dtmax)

    print "  srmax = %8.4g samples/sec" % srmax
    print "     sr = %8.4g samples/sec" % sr
    print "  srmin = %8.4g samples/sec" % srmin

    if((srmax-srmin) > 0.01*sr):
        print(" ")
        print(" Warning: sample rate difference ")
        sr = None
        while not sr:
            try:
                print(" Enter new sample rate ")
                s = stdin.readline()
                sr=float(s)
                dt=1/sr
            except ValueError:
                print 'Invalid Number'
    return sr,dt

########################################################################

def GetInteger2():
    nnn = None
    while nnn != 1 and nnn !=2:
        try:
            s=stdin.readline()
            nnn = int(s)
        except ValueError:
            print 'Invalid Number. Enter integer. '
    return nnn

def GetInteger3():
    nnn = None
    while nnn != 1 and nnn !=2 and nnn !=3:
        try:
            s = stdin.readline()
            nnn=int(s)
        except ValueError:
            print 'Invalid Number. Enter integer.'
    return nnn

def GetInteger_n(m):
    iflag=0
    while(iflag==0):
        try:
            s = stdin.readline()
            nnn=int(s)
            for i in range (1,m+1):
                if nnn==i:
                    iflag=1
                    break;
        except ValueError:
            print 'Invalid Number. Enter integer.'
    return nnn

#########################################################################

def enter_initial(iu):
    """
    iu = units
    v0 = initial velocity
    d0 = initial displacement
    """
    print(" ");
    if(iu==1):
        print(" Enter initial velocity (in/sec)")
    else:
        print(" Enter initial velocity (m/sec)")

    v0=enter_float()

    if(iu==1):
        print(" Enter initial displacement (in)")
    else:
        print(" Enter initial displacement (m)")

    d0=enter_float()

    return v0,d0

#########################################################################

def enter_damping():
    """
    Select damping input method.
    Return: damping ratio & Q
    """
    print " Select damping input type "
    print "   1=damping ratio "
    print "   2=Q "

    idamp = GetInteger2()

    print " "

    if idamp==1:
        print " Enter damping ratio "
    else:
        print " Enter amplification factor (Q) "

    damp_num = None

    while not damp_num:
        try:
            s =stdin.readline()
            damp_num = float(s)
        except ValueError:
            print 'Invalid Number'

    if idamp==1:
        damp=damp_num
        Q=1./(2.*damp_num)
    else:
        Q=damp_num
        damp=1./(2.*Q)

    return damp,Q

##########################################################################

def enter_fn():
    """
    Enter the natural frequency (Hz)
    """
    print " "
    print " Select units "
    print " 1=English  2=metric"
    iu=GetInteger2()

    print " "
    print " Select fn input method "
    print " 1=fn   2=fn from mass & stiffness"
    im=GetInteger2()

    if(im==1):
        print " "
        print " Enter fn (Hz) "
        fn=enter_float()
        omegan=2*pi*fn
    else:

        if(iu==1):
            print(" Enter mass (lbm)")
        else:
            print(" Enter mass (kg)")

        mass=enter_float()

        if(iu==1):
            mass/=386

        if(iu==1):
            print(" Enter stiffness (lbf/in)")
        else:
            print(" Enter stiffness (N/m)")

        stiffness=enter_float()
        omegan=sqrt(stiffness/mass)
        fn=omegan/(2*pi)

    period=1/fn
    return iu,fn,omegan,period

##########################################################################

def enter_float():
    """
    Enter a floating point number and check its validity
    """
    number_float = None
    while not number_float:
        try:
            s =stdin.readline()
            number_float = float(s)
            if number_float == 0:
                break
        except ValueError:
            print 'Invalid Number.  Enter number. '
    return number_float

##########################################################################

def enter_int():
    """
    Enter an integer and check its validity
    """
    number_int = None
    while not number_int:
        try:
            s =stdin.readline()
            number_int = int(s)
        except ValueError:
            print 'Invalid Number.  Enter number. '
    return number_int

##########################################################################

def WriteData2(nn,aa,bb,output_file_path):
    """
    Write two columns of data to an external ASCII text file
    """
    output_file = output_file_path.rstrip('\n')
    outfile = open(output_file,"w")
    for i in range (0, nn):
        outfile.write(' %10.6e \t %8.4e \n' %  (aa[i],bb[i]))
    outfile.close()

########################################################################


def WriteData3(nn,aa,bb,cc,output_file_path):
    """
    Write three columns of data to an external ASCII text file
    """
    outfile = open(output_file_path,"w")
    for i in range (0, nn):
        outfile.write(' %8.4e \t %8.4e \t %8.4e \n' %  (aa[i],bb[i],cc[i]))
    outfile.close()

#########################################################################

def time_history_plot(a,b,n,xlab,ylab,ptitle,stitle):
    """
    Plot a time history
       a=time   b=amplitude
       n=figure number

       xlab=x-axis label
       ylab=yaxis label

       ptitle=plot title
       stitle=save figure as filename
    """
    plt.figure(n)
    plt.plot(a, b, linewidth=1.0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    plt.title(ptitle)
    plt.savefig(stitle)
    plt.draw()

#########################################################################

def srs_plot_pn(srs_type,unit,fn,x_pos,x_neg,damp,stitle):
    """
    Plot and SRS with both positive and negative curves.
       srs_type = 1 for acceleration
                = 2 for pseudo velocity
                = 3 for relative displacement

           unit = 1 for English
                = 2 for metric

             fn = natural frequency

    x_pos,x_eng = postive, negative SRS

           damp = damping ratio

         stitle = output figure filename
    """


    if(srs_type !=1 and srs_type !=2 and srs_type !=3):
        srs_type=1

    if(unit !=1 and unit !=2):
        unit=1


    if(srs_type==1): # acceleration
        astr='Acceleration'

        if(unit==1): # English
            ymetric='Peak Accel (G)'
        if(unit==2): # metric
            ymetric='Peak Accel (m/sec^2)'

    if(srs_type==2): # pseudo velocity
        astr='Pseudo Velocity'

        if(unit==1): # English
            ymetric='Peak Velocity (in/sec)'
        if(unit==2): # metric
            ymetric='Peak Velocity (m/sec)'

    if(srs_type==3): # relative displacement
        astr='Relative Displacement'

        if(unit==1): # English
            ymetric='Relative Disp (in)'
        if(unit==2): # metric
            x_pos/=1000
            x_neg/=1000
            ymetric='Relative Disp (mm)'


    plt.plot(fn, x_pos, label="positive")
    plt.plot(fn, x_neg, label="negative")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
#
    Q=1/(2*damp)
    title_string= astr + ' Shock Response Spectrum Q='+str(Q)
#
    for i in range(1,200):
        if(Q==float(i)):
            title_string= astr +' Shock Response Spectrum Q='+str(i)
            break;
#
    plt.title(title_string)
    plt.xlabel('Natural Frequency (Hz) ')
    plt.ylabel(ymetric)
    plt.grid(True, which="both")
    plt.savefig(stitle)
    plt.legend(loc="upper left")
    plt.draw()

#########################################################################

def MatrixMax(input_matrix):
    """
    Return the maximum value of a matrix
    """
    return np.max(input_matrix)

#########################################################################

def WriteArray(aa,output_file_path):
    """
    Write array to file
    """
    output_file = output_file_path.rstrip('\n')
    outfile = open(output_file,"w")
#
    np.savetxt(outfile, aa,fmt='%8.4e', delimiter='\t')
    outfile.close()

##########################################################################

def SizeArray(input_matrix):
    """
    Return the size of an array
    """
    nrows=input_matrix.shape[0]
    ncolumns=input_matrix.shape[1]
    return nrows,ncolumns

################################################################################

def material():
    """
    Select material properties.
    Return elastic modulus (lbf/in^2), density(lbf sec^2/in^4), and Poisson ratio
    """
    print(' Select material ')
    print(' 1=aluminum  2=steel  3=G10  4=other ')
    imat = GetInteger_n(4)

    if(imat==1):      # aluminum
        E=1.0e+07
        rho=0.1
        mu=0.3

    if(imat==2):      # steel
        E=3.0e+07
        rho=0.285
        mu=0.3

    if(imat==3):      # G10
        E=2.7e+06
        rho=0.065
        mu=0.12

    if(imat==4):
        print(' ')
        print(' Enter elastic modulus (lbf/in^2)');
        E=enter_float()
        print(' ')
        print(' Enter mass density (lbm/in^3)')
        rho=enter_float()
        print(' ')
        print(' Enter Poisson ratio')
        mu=enter_float()

    rho=rho/386.

    return E,rho,mu
