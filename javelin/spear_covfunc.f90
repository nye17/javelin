! Last-modified: 25 Feb 2013 01:46:47 AM

MODULE spear_covfunc
implicit none

contains

SUBROUTINE covmat_bit(mat,jd1,jd2,id1,id2,sigma,tau,slagarr,swidarr,scalearr,nx,ny,ncurve,cmin,cmax,symm)
implicit none
!f2py intent(inplace) mat
!f2py intent(in) jd1,jd2,id1,id2
!f2py intent(hide) nx,ny,ncurve
!f2py logical intent(in), optional :: symm=0
!f2py integer intent(in), optional :: cmin=0
!f2py integer intent(in), optional :: cmax=-1
!f2py intent(in) 
!f2py threadsafe
INTEGER(kind=4)  :: nx,ny,ncurve,cmin,cmax
REAL(kind=8), DIMENSION(nx,ny) :: mat
REAL(kind=8), DIMENSION(nx) :: jd1
REAL(kind=8), DIMENSION(ny) :: jd2
INTEGER(kind=4), DIMENSION(nx) :: id1
INTEGER(kind=4), DIMENSION(ny) :: id2
REAL(kind=8) :: sigma,tau
REAL(kind=8), DIMENSION(ncurve) :: slagarr,swidarr,scalearr
LOGICAL :: symm
INTEGER(kind=4)  :: i,j
REAL(kind=8) :: slag1,swid1,scale1,slag2,swid2,scale2

if (cmax .eq. -1) then
    cmax = ny
endif

if (symm) then
    do j = cmin+1,cmax
        slag2 = slagarr(id2(j))
        swid2 = swidarr(id2(j))
        scale2=scalearr(id2(j))
        do i=1,j
            slag1 = slagarr(id1(i))
            swid1 = swidarr(id1(i))
            scale1=scalearr(id1(i))
            call covmatij(mat(i,j), id1(i),id2(j),jd1(i),jd2(j),sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2)
        enddo
    enddo
else
    do j = cmin+1,cmax
        slag2 = slagarr(id2(j))
        swid2 = swidarr(id2(j))
        scale2=scalearr(id2(j))
        do i=1,nx
            slag1 = slagarr(id1(i))
            swid1 = swidarr(id1(i))
            scale1=scalearr(id1(i))
            call covmatij(mat(i,j), id1(i),id2(j),jd1(i),jd2(j),sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2)
        enddo
    enddo
endif
return
END SUBROUTINE covmat_bit

!XXX to be test.
SUBROUTINE covmatpmap_bit(mat,jd1,jd2,id1,id2,sigma,tau,slagarr,swidarr,scalearr,nx,ny,ncurve,cmin,cmax,symm)
implicit none
!f2py intent(inplace) mat
!f2py intent(in) jd1,jd2,id1,id2
!f2py intent(hide) nx,ny,ncurve
!f2py logical intent(in), optional :: symm=0
!f2py integer intent(in), optional :: cmin=0
!f2py integer intent(in), optional :: cmax=-1
!f2py intent(in) 
!f2py threadsafe
INTEGER(kind=4)  :: nx,ny,ncurve,cmin,cmax
REAL(kind=8), DIMENSION(nx,ny) :: mat
REAL(kind=8), DIMENSION(nx) :: jd1
REAL(kind=8), DIMENSION(ny) :: jd2
INTEGER(kind=4), DIMENSION(nx) :: id1
INTEGER(kind=4), DIMENSION(ny) :: id2
REAL(kind=8) :: sigma,tau
! here ncurve is not the actual number --> we count the line band flux as two.
REAL(kind=8), DIMENSION(ncurve) :: slagarr,swidarr,scalearr
LOGICAL :: symm
INTEGER(kind=4)  :: i,j
REAL(kind=8) :: slag1,swid1,scale1,slag2,swid2,scale2,scale_hidden

if (cmax .eq. -1) then
    cmax = ny
endif

! scale_hidden is the scale of the hidden continuum under line band.
scale_hidden =scalearr(3)

if (symm) then
    do j = cmin+1,cmax
        slag2 = slagarr(id2(j))
        swid2 = swidarr(id2(j))
        scale2=scalearr(id2(j))
        do i=1,j
            slag1 = slagarr(id1(i))
            swid1 = swidarr(id1(i))
            scale1=scalearr(id1(i))
            call covmatpmapij(mat(i,j), id1(i),id2(j),jd1(i),jd2(j),sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2,scale_hidden)
        enddo
    enddo
else
    do j = cmin+1,cmax
        slag2 = slagarr(id2(j))
        swid2 = swidarr(id2(j))
        scale2=scalearr(id2(j))
        do i=1,nx
            slag1 = slagarr(id1(i))
            swid1 = swidarr(id1(i))
            scale1=scalearr(id1(i))
            call covmatpmapij(mat(i,j), id1(i),id2(j),jd1(i),jd2(j),sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2,scale_hidden)
        enddo
    enddo
endif
return
END SUBROUTINE covmatpmap_bit

!XXX this is deprecated.
SUBROUTINE covmat(mat,npt,ncurve,idarr,jdarr,sigma,tau,slagarr,swidarr,scalearr)
implicit none
REAL(kind=8), DIMENSION(npt,npt),intent(out)  :: mat
INTEGER(kind=4),intent(in)  :: npt,ncurve
INTEGER(kind=4), DIMENSION(npt),intent(in)  :: idarr
REAL(kind=8), DIMENSION(npt),intent(in)  :: jdarr
REAL(kind=8), DIMENSION(ncurve),intent(in)  :: slagarr,swidarr,scalearr
REAL(kind=8),intent(in)  :: sigma,tau
INTEGER(kind=4) :: m,n,id1,id2
REAL(kind=8) :: jd1,jd2,slag1,swid1,scale1,slag2,swid2,scale2

do m=1,npt
    do n=m,npt
        id1=idarr(m)
        id2=idarr(n)
        jd1=jdarr(m)
        jd2=jdarr(n)
        slag1 =slagarr(id1)
        swid1 =swidarr(id1)
        scale1=scalearr(id1)
        slag2 =slagarr(id2)
        swid2 =swidarr(id2)
        scale2=scalearr(id2)
        call covmatij(mat(m,n), id1,id2,jd1,jd2,sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2)
        if (m .ne. n) then
            mat(n,m) = mat(m,n)
        endif
    enddo
enddo

END SUBROUTINE covmat

SUBROUTINE covmatij(covij,id1,id2,jd1,jd2,sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2)
implicit none
REAL(kind=8),intent(out) :: covij
INTEGER(kind=4),intent(in) :: id1,id2
REAL(kind=8),intent(in)  :: jd1,jd2
REAL(kind=8),intent(in)  :: sigma,tau
REAL(kind=8),intent(in)  :: slag1,swid1,scale1,slag2,swid2,scale2
REAL(kind=8) :: twidth,twidth1,twidth2
REAL(kind=8) :: tgap,tgap1,tgap2
INTEGER(kind=4) :: imax,imin

imax = max(id1,id2)
imin = min(id1,id2)

if (imin .le. 0) then
    print*,"ids can not be smaller than 1"
    covij = -1.0D0
    return
endif

if (imin .eq. imax) then
    ! between two epochs of the same light curve
    if (imin .eq. 1) then
        ! continuum auto
        covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
    else
        ! line auto
        if(swid1 .le. 0.01D0) then
            covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
        else
            covij = getcmat_lauto(id1,jd1,jd2,tau,slag1,swid1,scale1)
        endif
    endif
else
    ! between two epochs of different light curves
    if (imin .eq. 1) then
        ! continuum and line cross
        ! assume swid of the continuum is 0.0
        twidth = max(swid1, swid2)
        if (twidth .le. 0.01D0) then
            covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
        else
            covij = getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
        endif
    else 
        ! line1 and line2 cross
        twidth1 = swid1
        twidth2 = swid2
        if((twidth1.le.0.01D0).and.(twidth2.le.0.01D0)) then
            covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
        else if((twidth1 .le. 0.01D0).or.(twidth2 .le. 0.01D0)) then
            covij = getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
        else
            covij = getcmat_lcross(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
        endif
    endif
endif
covij = sigma*sigma*covij
return
END SUBROUTINE covmatij

! pmap version of covmatij, now only allows two light curves.
SUBROUTINE covmatpmapij(covij,id1,id2,jd1,jd2,sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2,scale_hidden)
implicit none
REAL(kind=8),intent(out) :: covij
INTEGER(kind=4),intent(in) :: id1,id2
REAL(kind=8),intent(in)  :: jd1,jd2
REAL(kind=8),intent(in)  :: sigma,tau
REAL(kind=8),intent(in)  :: slag1,swid1,scale1,slag2,swid2,scale2,scale_hidden
REAL(kind=8) :: twidth,twidth1,twidth2
REAL(kind=8) :: tgap,tgap1,tgap2
INTEGER(kind=4) :: imax,imin

imax = max(id1,id2)
imin = min(id1,id2)

! here imin(imax) is either one or two.
if (imin .le. 0) then
    print*,"ids can not be smaller than 1"
    covij = -1.0D0
    return
endif

if (imin .eq. imax) then
    ! between two epochs of the same light curve
    if (imin .eq. 1) then
        ! id1 = id2 = 1
        ! continuum band auto: cov(c0i,c0j)
        covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
    else
        ! id1 = id2 = 2
        ! line band auto: cov(c1i,c1j) + cov(c1i, lj) + cov(li, c1j) + cov(li, lj)
        ! Both (slag1, swid1, scale1) and (slag2, swid2, scale2) are
        ! referring to the line properties.
        ! cov(c1i,c1j)
        covij = getcmat_delta(id1,id2,jd1,jd2,tau,0.0D0,scale_hidden,0.0D0,scale_hidden)
        ! cov(c1i, lj)
        if(swid2 .le. 0.01D0) then
            ! when swid is small, no convulted covariance is needed.
            covij = covij + getcmat_delta(id1,id2,jd1,jd2,tau,0.0D0,scale_hidden,slag2,scale2)
        else
            !covij = getcmat_lauto(id1,jd1,jd2,tau,slag1,swid1,scale1)
            covij = covij + getcmat_lc(id1,id2,jd1,jd2,tau,0.0D0,0.0D0,scale_hidden,slag2,swid2,scale2)
        endif
        ! cov(li, c1j)
        ! similar to cov(cli, lj), simple mutation of indices.
        if(swid1 .le. 0.01D0) then
            covij = covij + getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,0.0D0,scale_hidden)
        else
            covij = covij + getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,0.0D0,0.0D0,scale_hidden)
        endif
        ! cov(li, lj)
        if(swid1 .le. 0.01D0) then
            covij = covij + getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
        else
            covij = covij + getcmat_lauto(id1,jd1,jd2,tau,slag1,swid1,scale1)
        endif
    endif
else
    ! between two epochs of different light curves 
    !XXX now just the continuum and line bands
    ! continuum band and line band cross cov(c0i, c1j) + cov(c0i, lj)
    ! cov(c0i, c1j)
    covij = getcmat_delta(id1,id2,jd1,jd2,tau,0.0D0,1.0D0,0.0D0,scale_hidden)
    ! cov(c0i, lj)
    twidth = max(swid1, swid2)
    if (twidth .le. 0.01D0) then
        covij = covij + getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
    else
        covij = covij + getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
    endif
endif
covij = sigma*sigma*covij
return
END SUBROUTINE covmatpmapij

FUNCTION expcov(djd,tau)
implicit none
REAL(kind=8) :: expcov,djd,tau
expcov = exp(-dabs(djd)/tau)
return
END FUNCTION expcov

FUNCTION getcmat_delta(id1,id2,jd1,jd2,tau,tspike1,scale1,tspike2,scale2)
implicit none
REAL(kind=8) :: getcmat_delta
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tspike1,tspike2,tau
REAL(kind=8) ::  scale1,scale2
getcmat_delta = exp(-abs(jd1-jd2-tspike1+tspike2)/tau)
getcmat_delta = abs(scale1*scale2)*getcmat_delta
return
END FUNCTION getcmat_delta

FUNCTION getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
implicit none
REAL(kind=8) :: getcmat_lc
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tau
REAL(kind=8) ::  twidth,emscale,tlow,thig
REAL(kind=8) ::  slag1,swid1,scale1,slag2,swid2,scale2

if ((id1.eq.1).and.(id2.ge.2)) then
    tlow = jd2-jd1-slag2-0.5D0*swid2
    thig = jd2-jd1-slag2+0.5D0*swid2
    emscale = dabs(scale2)
    twidth  = swid2
else if ((id2.eq.1).and.(id1.ge.2)) then
    tlow = jd1-jd2-slag1-0.5D0*swid1
    thig = jd1-jd2-slag1+0.5D0*swid1
    emscale = dabs(scale1)
    twidth  = swid1
    ! XXX the following code is inherited from the old spear code where the
    ! DOUBLE_HAT mode used to call getcmat_lc from the cross-correlation
    ! between two lines with one of their widths zero (DEPRECATED, but no harm
    ! if kept).
else if((id1.ge.2).and.(id2.ge.2)) then
    if (swid1.le.0.01D0) then
        tlow = jd2-(jd1-slag1)-slag2-0.5D0*swid2
        thig = jd2-(jd1-slag1)-slag2+0.5D0*swid2
        emscale = dabs(scale2*scale1)
        twidth  = swid2
    else if(swid2.le.0.01D0) then
        tlow = jd1-(jd2-slag2)-slag1-0.5D0*swid1
        thig = jd1-(jd2-slag2)-slag1+0.5D0*swid1
        emscale = dabs(scale2*scale1)
        twidth  = swid1
    endif
endif
if (thig.le.0.0D0) then
    getcmat_lc = exp( thig/tau)-exp( tlow/tau)
else if (tlow.ge.0.0D0) then
    getcmat_lc = exp(-tlow/tau)-exp(-thig/tau)
else 
    getcmat_lc = 2.0D0-exp(tlow/tau)-exp(-thig/tau)
endif
getcmat_lc = tau*(emscale/twidth)*getcmat_lc
return
END FUNCTION getcmat_lc

FUNCTION getcmat_lauto(id,jd1,jd2,tau,slag,swid,scale)
implicit none
REAL(kind=8) :: getcmat_lauto
INTEGER(kind=4) :: id
REAL(kind=8) :: jd1,jd2,tau
REAL(kind=8) :: slag,swid,scale
REAL(kind=8) :: twidth,emscale,tlow,tmid,thig

twidth  = swid
emscale = scale

tlow = jd1-jd2-twidth
tmid = jd1-jd2
thig = jd1-jd2+twidth

if((thig.le.0.0D0).or.(tlow.ge.0.0D0))then
    getcmat_lauto = exp(-abs(tmid)/tau)*(exp(0.5D0*twidth/tau)-exp(-0.5D0*twidth/tau))**2
else
    getcmat_lauto = -2.0D0*exp(-abs(tmid)/tau)+exp(-twidth/tau)*(exp(-tmid/tau)+exp(tmid/tau))
    if(tmid.ge.0.0D0)then
        getcmat_lauto = getcmat_lauto+2.0D0*(twidth-tmid)/tau
    else
        getcmat_lauto = getcmat_lauto+2.0D0*(twidth+tmid)/tau
    endif
endif
getcmat_lauto = ((emscale*tau/twidth)**2)*getcmat_lauto
return
END FUNCTION getcmat_lauto

FUNCTION getcmat_lcross(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
implicit none
REAL(kind=8) :: getcmat_lcross
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tau
REAL(kind=8) ::  slag1,swid1,scale1,slag2,swid2,scale2
REAL(kind=8) ::  twidth1,twidth2,bottleneck
REAL(kind=8) :: t1,t2,t3,t4,ti,tj,tlow,tmid1,tmid2,thig

twidth1 = swid1
twidth2 = swid2

if(twidth1.ge.twidth2) then
    t1 = slag1-0.5D0*twidth1
    t2 = slag1+0.5D0*twidth1
    t3 = slag2-0.5D0*twidth2
    t4 = slag2+0.5D0*twidth2
    bottleneck = twidth2
    ti = jd1
    tj = jd2
else
    t1 = slag2-0.5D0*twidth2
    t2 = slag2+0.5D0*twidth2
    t3 = slag1-0.5D0*twidth1
    t4 = slag1+0.5D0*twidth1
    bottleneck = twidth1
    ti = jd2
    tj = jd1
endif

tlow  = (ti-tj)-(t2-t3)
tmid1 = (ti-tj)-(t2-t4)
tmid2 = (ti-tj)-(t1-t3)
thig  = (ti-tj)-(t1-t4)

if((thig.le.0.0D0).or.(tlow.ge.0.0D0)) then
    getcmat_lcross = dexp(-dabs(tlow)/tau) +dexp(-dabs(thig)/tau)&
                    -dexp(-dabs(tmid1)/tau)-dexp(-dabs(tmid2)/tau)
else 
    getcmat_lcross = dexp(tlow/tau)+dexp(-thig/tau)&
                    -dexp(-dabs(tmid1)/tau)-dexp(-dabs(tmid2)/tau)
    if(tmid2.le.0.0D0) then
        getcmat_lcross = getcmat_lcross+2.0D0*thig/tau
    else if(tmid1.le.0.0D0) then
        getcmat_lcross = getcmat_lcross+2.0D0*bottleneck/tau
    else if(tlow .lt.0.0D0) then
        getcmat_lcross = getcmat_lcross-2.0D0*tlow/tau
    endif
endif

getcmat_lcross = (tau*tau*scale1*scale2/(twidth1*twidth2))&
                 *getcmat_lcross
RETURN
END FUNCTION getcmat_lcross

END MODULE spear_covfunc
