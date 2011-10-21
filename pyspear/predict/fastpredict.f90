!Last-modified: 19 Oct 2011 09:06:56 PM

SUBROUTINE FASTPREDICT(jd,mag,emag,npt,sigma,tau,jwant,mwant,ewant,rwant,nwant,iseed)
!***********************************************************************
implicit real*8 (a-h,o-z) 
real*8,dimension(:) :: jd,mag,emag 
real*8,dimension(:) :: jwant,mwant,ewant,rwant
real*8,allocatable  :: ri(:),atridag(:),btridag(:),ctridag(:) 
real*8,allocatable  :: atridags(:),btridags(:),ctridags(:) 
real*8,allocatable  :: magnoise(:) 
real*8,allocatable  :: temp1(:),temp2(:),temp3(:) 
real*8,allocatable  :: sstar(:) 
real*8,allocatable  :: inoise(:) 
real*8,allocatable  :: achol(:),bchol(:) 
real*8 check(100,100),arg,ran2,gasdev
integer :: nmax,iseed
EXTERNAL ran2,gasdev

print*,'will predict ',nwant,' points  with iseed',iseed
variance  = sigma*sqrt(0.5D0*tau) 
variance2 = variance*variance 
                                                                        
! see Notes on Gaussian Random Functions with Exponential Correlation Fu
!     (Ornstein-Uhlenbeck Process)                                      
! also see Rybicki and Press (1995)                                     
                                                                        
! first we want to produce the predicted light curve -- just on data    
! we have that: 
!        s* = S* (S+N)^(-1) y = S* S^(-1) (S^(-1)+N^(-1))^(-1) N^(-1) y
! because both S^(-1) and S^(-1)+N^(-1) are tridiagonal           
!              tridags    tridag                                  
! so                                                              
!        s* = S* (S+N)^(-1) y = S* S^(-1) (S^(-1)+N^(-1))^(-1) N^(-1) y
!                             = S* tridags (tridag)^(-1) magnoise
!
! WARNING: note the difference between tridags and tridag (-,-)         
                                                                        
                                                                        
! this is the tridiagonal form of (S^(-1)+N^(-1))                       
nmax = max(nwant,npt)
allocate(ctridag(nmax),atridag(nmax),btridag(nmax))
allocate(ri(nmax),magnoise(nmax))
allocate(ctridags(nmax),atridags(nmax),btridags(nmax),stat=ierr)
if(ierr .ne. 0) then
    print*,"ERROR: fail to assign arrays",ierr
    stop
endif

do i=1,npt-1 
  if (jd(i+1).le.jd(i)) then 
    print*,' fails if the data is not in time order ' 
    print*,' and will have a divergence for equal time data ' 
    print*,' offender is points ',i,i+1,jd(i),jd(i+1) 
    stop 
  endif 
  arg        = abs(jd(i+1)-jd(i))/tau 
  ri(i)      = exp(-arg) 
  ctridag(i) = -ri(i)/(1.0D0-ri(i)*ri(i)) 
enddo 
do i=2,npt 
  atridag(i) = ctridag(i-1) 
enddo 
btridag(1)   = 1.0-ri(1)    *ctridag(1) 
btridag(npt) = 1.0-ri(npt-1)*ctridag(npt-1) 
do i=2,npt-1 
  btridag(i) = 1.0 - ri(i)*ctridag(i) - ri(i-1)*ctridag(i-1) 
enddo 

do i=1,npt 
  atridag(i)  = atridag(i)/variance2 
  btridag(i)  = btridag(i)/variance2 
  ctridag(i)  = ctridag(i)/variance2 
! save S^(-1)                                                           
!      tridags                                                          
  atridags(i) = atridag(i) 
  btridags(i) = btridag(i) 
  ctridags(i) = ctridag(i) 
! make the original S^(-1)+N^(-1)                                       
  btridag(i)  = btridag(i) + 1.0/emag(i)**2 
! this is N^(-1) y                                                      
  magnoise(i) = mag(i)/emag(i)**2 
enddo 

! s* = S* (S+N)^(-1) y = S* S^(-1) (S^(-1)+N^(-1))^(-1) N^(-1) y
!                      = S* tridags (tridag)^(-1) magnoise
! now solve the tridiagonal equation:
!    (S^(-1)+N^(-1)) u = N^(-1) y 
!      tridag * temp1  = magnoise      
! temp1 = (tridag)^(-1) magnoise                                        
allocate(temp1(nmax),temp2(nmax),temp3(nmax))
call tridag(atridag,btridag,ctridag,magnoise,temp1,npt) 
                                                                        
! now multiply S^(-1) temp1 = temp2                                     
! temp2 = tridags (tridag)^(-1) magnoise                                
do i=1,npt 
  temp2(i) = btridags(i)*temp1(i) 
  if (i.ne.1  ) temp2(i) = temp2(i) + atridags(i)*temp1(i-1) 
  if (i.ne.npt) temp2(i) = temp2(i) + ctridags(i)*temp1(i+1) 
enddo 
                                                                        
! now fill in the average model light curve                             
! now if we want the variance in this estimate, 
! it is:  <s*>^2 - S* (S+N)^(-1) S*
!       = variance2 - S* S^(-1) (S^(-1)+N^(-1))^(-1) N^(-1) S*
allocate(sstar(nmax))
do i=1,nwant 
  mwant(i) = 0.0 
  do j=1,npt 
    sstar(j)    = variance2*exp(-abs(jd(j)-jwant(i))/tau) 
    mwant(i)    = mwant(i) + temp2(j)*sstar(j) 
! reset magnoise to be N^(-1) S*                                        
    magnoise(j) = sstar(j)/emag(j)**2 
  enddo 
! solve  (S^(-1)+N^(-1)) u = N^(-1) S*                                  
!            tridag temp1  = magnosie                                   
  call tridag(atridag,btridag,ctridag,magnoise,temp1,npt) 
! now multiply S^(-1) temp1 = temp3                                     
  do j=1,npt 
    temp3(j) = btridags(j)*temp1(j) 
    if (j.ne.1  ) temp3(j) = temp3(j) + atridags(j)*temp1(j-1) 
    if (j.ne.npt) temp3(j) = temp3(j) + ctridags(j)*temp1(j+1) 
  enddo 
! finally dot with sstar                                                
  ewant(i) = variance2 
  do j=1,npt 
    ewant(i) = ewant(i) - sstar(j)*temp3(j) 
  enddo 
  ewant(i) = sqrt(ewant(i)) 
enddo 
                                                                        
! now get a constrained realization of the light curve                  
! first, set the inverse noise matrix for desired points = 0            
! we're assuming that the "wants" are densely sampled compared to the da
! so we find the want closest to each data point and make it the data po
! insures that we don't have to worry about the time ordering/overlap pr
! we're not going to pretty here -- brute force to save my time -- could
! made N log(M) rather than N*M                                         
! basically, if data, inverse error matrix is inverse error, otherwise z
allocate(inoise(nmax))
do i=1,nwant 
  inoise(i) = 0.0 
enddo 
do i=1,npt 
  dmin = 1.e32 
  do j=1,nwant 
    dist = abs(jd(i)-jwant(j)) 
    if (dist.lt.dmin) then 
      dmin = dist 
      jmin = j 
    endif 
  enddo 
  inoise(jmin) = 1.0/emag(i)**2 
enddo 
                                                                        
! next we build S^(-1) as a tridiagonal matrix                          
do i=1,nwant-1 
  if (jwant(i+1).le.jwant(i)) then 
    print*,' fails if the data is not in time order ' 
    print*,' and will have a divergence for equal time data ' 
    print*,' offender is points ',i,i+1,jwant(i),jwant(i+1) 
    stop 
  endif 
  arg        = abs(jwant(i+1)-jwant(i))/tau 
  ri(i)      = exp(-arg) 
  ctridag(i) = -ri(i)/(1.0-ri(i)*ri(i)) 
enddo 
do i=2,nwant 
  atridag(i) = ctridag(i-1) 
enddo 
btridag(1)     = 1.0-ri(1)      *ctridag(1) 
btridag(nwant) = 1.0-ri(nwant-1)*ctridag(nwant-1) 
do i=2,nwant-1 
  btridag(i) = 1.0 - ri(i)*ctridag(i) - ri(i-1)*ctridag(i-1) 
enddo 
do i=1,nwant 
  atridag(i) = atridag(i)/variance2 
  btridag(i) = btridag(i)/variance2 
  ctridag(i) = ctridag(i)/variance2 
enddo 
                                                                        
! now add in the noise part of the inverse so that we 
! now have (S^(-1)+N^(-1))
do i=1,nwant 
  btridag(i) = btridag(i) + inoise(i) 
enddo 
                                                                        
! now we want the Cholesky decomposition of this                        
!   achol is on the diagonal, bchol is above the diagonal               
allocate(achol(nmax),bchol(nmax))
do i=1,nwant 
  achol(i) = btridag(i) 
  bchol(i) = ctridag(i) 
enddo 
achol(1) = sqrt(achol(1)) 
do i=1,nwant-1 
  bchol(i)   = bchol(i)/achol(i) 
  achol(i+1) = sqrt(achol(i+1)-bchol(i)**2) 
enddo 
                                                                        
! now a random constrained relization has a correlation function        
!     Q = (S^(-1)+N(-1))^(-1)  = C^T C                                  
! if we pick a random deviate r and set y = C r                         
!    < y y^T > = < C r r^T C^T > = C < r r^T > C^T = C C^T = Q^T = Q    
! now we have the cholesky decomposition of Q^(-1) = U^T U              
!    Q = (U^T U)^(-1) = U^(-1) U^(-1)^T                                 
! so it would seem that  C = U^(-1)^T                                   
!    need to solve C^(-1) y = r                                         
! build tridiagonal matrix -- as transpose of the choldc from above     
! build a gaussian deviate (it actually doesn't matter if you use       
! the matrix or its transpose                                           

do i=1,nwant 
  atridag(i) = 0.0 
  if (i.ne.1) atridag(i) = bchol(i-1) 
  btridag(i) = achol(i) 
  ctridag(i) = 0.0 
  temp1(i)   = gasdev(iseed)
enddo 
! solve the equations                                                   

call tridag(atridag,btridag,ctridag,temp1,rwant,nwant) 
                                                                        
deallocate(ctridag,atridag,btridag)
deallocate(ri,magnoise)
deallocate(ctridags,atridags,btridags)
deallocate(temp1,temp2,temp3)
deallocate(sstar)
deallocate(inoise)
deallocate(achol,bchol)
RETURN 
END                                           
!***********************************************************************
                                                                        
                                                                        
                                                                        
!***********************************************************************
SUBROUTINE tridag(a,b,c,r,u,n) 
implicit real*8 (a-h,o-z) 
INTEGER n,NMAX 
REAL*8 a(n),b(n),c(n),r(n),u(n) 
INTEGER j 
REAL*8 bet,gam(n) 
if(b(1).eq.0.)pause 'tridag: rewrite equations' 
bet=b(1) 
u(1)=r(1)/bet 
do j=2,n 
  gam(j)=c(j-1)/bet 
  bet=b(j)-a(j)*gam(j) 
  if(bet.eq.0.)pause 'tridag failed' 
  u(j)=(r(j)-a(j)*u(j-1))/bet 
enddo
do j=n-1,1,-1 
  u(j)=u(j)-gam(j+1)*u(j+1) 
enddo
return 
END SUBROUTINE
                                                                        

!***********************************************************************
FUNCTION gasdev(idum)
implicit none
INTEGER(kind=4) :: idum
REAL(kind=8) :: gasdev
INTEGER(kind=4),SAVE :: iset = 0
REAL(kind=8),SAVE :: gset
REAL(kind=8) :: fac,rsq,v1,v2,ran2

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


                                                                        
!***********************************************************************
FUNCTION ran2(idum)
implicit none
INTEGER(kind=4) :: idum
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



