!Last-modified: 18 Oct 2011 05:49:36 PM

module util
implicit none

contains

FUNCTION gasdev(idum)
implicit none
INTEGER(kind=4), intent(inout) :: idum
REAL(kind=8) :: gasdev
INTEGER(kind=4),SAVE :: iset = 0
REAL(kind=8),SAVE :: gset
REAL(kind=8) :: fac,rsq,v1,v2
if (iset.eq.0) then
    do
        v1=2.*ran2(idum)-1.
        v2=2.*ran2(idum)-1.
        rsq=v1**2+v2**2
        if(rsq.lt.1.D0.and.rsq.gt.0.D0)exit
    enddo
    fac=sqrt(-2.D0*log(rsq)/rsq)
    gset=v1*fac
    gasdev=v2*fac
    iset=1
else
    gasdev=gset
    iset=0
endif
return
END FUNCTION

FUNCTION ran2(idum)
implicit none
INTEGER(kind=4), intent(inout) :: idum
REAL(kind=8) :: ran2
INTEGER(kind=4),PARAMETER :: IM1=2147483563,IM2=2147483399,&
IMM1=IM1-1,IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,IR2=3791,&
NTAB=32,NDIV=1+IMM1/NTAB
REAL(kind=8),PARAMETER :: AM=1.D0/float(IM1),EPS=1.2D-7,RNMX=1.D0-EPS
INTEGER(kind=4) :: j,k
INTEGER(kind=4), SAVE :: idum2=123456789,iy=0
INTEGER(kind=4),DIMENSION(NTAB),SAVE :: iv = 0

if (idum.le.0) then
  idum=max(-idum,1)
  idum2=idum
  do j=NTAB+8,1,-1
      k=idum/IQ1
      idum=IA1*(idum-k*IQ1)-k*IR1
      if (idum.lt.0) idum=idum+IM1
      if (j.le.NTAB) iv(j)=idum
  enddo
  iy=iv(1)
endif
k=idum/IQ1
idum=IA1*(idum-k*IQ1)-k*IR1
if (idum.lt.0) idum=idum+IM1
k=idum2/IQ2
idum2=IA2*(idum2-k*IQ2)-k*IR2
if (idum2.lt.0) idum2=idum2+IM2
j=1+iy/NDIV
iy=iv(j)-idum2
iv(j)=idum
if(iy.lt.1)iy=iy+IMM1
ran2=min(AM*iy,RNMX)
return
END FUNCTION

end module util
