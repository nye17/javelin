!Last-modified: 18 Oct 2011 07:21:36 PM


!http://cens.ioc.ee/projects/f2py2e/usersguide/index.html#allocatable-arrays

! basic idea is to set up a module with jd, mag, and emag could be null or
! could be initialized with input, and the module will predict light curves
! based on the current status of the three arrays.


module predict
implicit none
real(kind=8), allocatable, dimension(:) :: jdata, mdata, edata
real(kind=8), allocatable, dimension(:) :: jwant, mwant, ewant
integer(kind=4) :: nwant
integer(kind=4) :: ndata

contains

subroutine setup_desired_lc(jd, error, nmock)
implicit none
integer(kind=4), intent(in) :: nmock
real(kind=8), dimension(nmock), intent(in) :: jd, error
print*, "Note: you can also directly allocate/initialize jwant/mwant/ewant by:"
print*, "Note: predict.jwant/mwant/ewant = ndarray and predict.nwant=n"
if (allocated(jwant)) then
    print*, "Warning: desired light curve dates have already been allocated!"
    print*, "Warning: deallocating..."
    deallocate(jwant, mwant, ewant)
endif
print*, "allocating..."
nwant = nmock
allocate(jwant(nwant), mwant(nwant), ewant(nwant))
print*, "initializing jwant and ewant to input..."
jwant = jd 
ewant = error
print*, "initializing mwant to zeros..."
mwant = 0.0D0
end subroutine

subroutine setup_observed_lc(jd, mag, error, nobs)
implicit none
integer(kind=4), intent(in) :: nobs
real(kind=8), dimension(nobs), intent(in) :: jd, mag, error
print*, "Note: you can also directly allocate/initialize jdata/mdata/edata by:"
print*, "Note: predict.jdata/mdata/edata = ndarray and predict.ndata=n"
if (allocated(jdata)) then
    print*, "Warning: observed light curve dates have already been allocated!"
    print*, "Warning: deallocating..."
    deallocate(jdata, mdata, edata)
endif
print*, "allocating..."
ndata = nobs
allocate(jdata(ndata), mdata(ndata), edata(ndata))
print*, "initializing jdata, mdata, and edata to input..."
jdata = jd 
mdata = mag 
edata = error
end subroutine

subroutine seed_from_date(iseed)
implicit none
integer(kind=4), intent(out) :: iseed
integer(kind=4) ::  T(8)
CALL DATE_AND_TIME(VALUES = T)
iseed = T(1)+2*(T(2)+6*(T(3)+5*(T(5)+10*(T(6)+85*T(7)))))
IF (MOD(iseed,2).EQ.0) iseed = iseed-1
iseed = 0 - iseed
end subroutine

SUBROUTINE unconstrained_drw(tau,sigmahat,avgmag,iseed) 
implicit none
real(kind=8), intent(in) :: tau, sigmahat, avgmag
integer(kind=4), intent(inout) :: iseed
real(kind=8) :: omega0 , signal, alpha
integer(kind=4) :: i
! regenerate iseed from date and time if input is large than or equal to 0.
if (iseed >= 0) then
    call seed_from_date(iseed)
endif
if (allocated(jwant) .and. allocated(ewant)) then
    ! dimension checks.
    if (size(jwant) .ne. size(ewant)) then
        print*, "ERROR: jwant and ewant have different length!"
    else if (nwant .eq. 0) then
        nwant = size(jwant)
    else if (nwant .ne. size(jwant)) then
        print*, "ERROR: jwant and ewant have a different length than nwant!"
    endif
    ! see Appendix, Kelly et al. (2009), note the sigma there is the sigmahat.
    omega0 = 0.5*sigmahat*sigmahat*tau 
    do i=1,nwant
      if (i.eq.1) then 
        signal = sqrt(omega0)*gasdev(iseed) 
      else 
        alpha  = exp(-(jwant(i)-jwant(i-1))/tau) 
        signal = alpha*signal + sqrt(omega0*(1.0-alpha*alpha))*gasdev(iseed)
      endif 
      mwant(i) = avgmag + signal + ewant(i)*gasdev(iseed)
    enddo 
else
    print*, "ERROR: desired light curve dates or uncertainties have not been initialized"
endif
END SUBROUTINE                                           

SUBROUTINE constrained_drw(tau, sigmahat)
implicit none
real(kind=8), intent(in) :: tau, sigmahat
real(kind=8) :: sigma, variance
real(kind=8), dimension(ndata) :: atridag, btridag, ctridag

if (allocated(jdata) .and. allocated(mdata) .and. allocated(edata)) then
    ! dimension checks.
    if (size(jdata) .ne. size(edata)) .or. (size(jdata) .ne. size(mdata)) then
        print*, "ERROR: jdata, mdata, and edata have different lengths!"
    else if (ndata .eq. 0) then
        ndata = size(jdata)
    else if (ndata .ne. size(jdata)) then
        print*, "ERROR: jdata, mdata, and edata have a different length than ndata!"
    endif
    sigma = sigmahat*sqrt(0.5D0*tau)
    variance =  sigma*sigma
    !FIXME

else
    print*, "ERROR: desired light curve dates, fluxes/mags, or uncertainties have not been initialiized"
endif


END SUBROUTINE                                           


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

end module predict
