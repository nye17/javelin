! Last-modified: 12 Feb 2012 02:08:23 AM

MODULE spear_covfunc
implicit none

contains

FUNCTION covij(id1,id2,jd1,jd2,sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2)
implicit none
REAL(kind=8) :: covij
INTEGER(kind=4) :: id1,id2
REAL(kind=8) :: jd1,jd2
REAL(kind=8) :: sigma,tau
REAL(kind=8) :: slag1,swid1,scale1,slag2,swid2,scale2
REAL(kind=8) :: twidth,twidth1,twidth2
REAL(kind=8) :: tgap,tgap1,tgap2
INTEGER(kind=4) :: imax,imin

imax = max(id1,id2)
imin = min(id1,id2)

if (imin .le. 0) then
    print*,"ids can not be smaller than 1"
    exit
endif

if (imin .eq. imax) then
    ! between two epochs of the same light curve
    if (imin .eq. 1) then
        ! continuum auto
        covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag1,slag2,scale1,scale2)
        return
    else
        ! line auto
        covij = getcmat_lauto(id1,jd1,jd2,tau,slag1,swid1,scale1)
        return
    endif
else
    ! between two epochs of different light curves
    if (imin .eq. 1) then
        ! continuum and line
        twidth = swid(imax)
        if (twidth .le. 0.01D0) then
            covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag1,slag2,scale1,scale2)
        else
            covij = getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
        endif
    else 
        ! line1 and line2
        twidth1 = swid1
        twidth2 = swid2
        !FIXME
        covij = getcmat_lcross(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
endif


!***********

if(convmode.eq.DOUBLETOPHAT) then
    imax = max(id1,id2)
    imin = min(id1,id2)
    twidth1 = swid(id1)
    twidth2 = swid(id2)
    if(((twidth1.le.0.01D0).and.(twidth2.le.0.01D0)).or.(id1*id2.eq.1))then
        covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale)
    else if((imin.eq.1).and.(imax.ge.2))then
        if(swid(imax) .le. 0.01D0) then
            covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale)
        else
            covij = getcmat_lc(id1,id2,jd1,jd2,tau,slag,swid,scale)
        endif
    else if((imin.ge.2).and.(imax.eq.imin))then
        if(swid(imin) .le. 0.01) then
            covij = getcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale)
        else
            covij = getcmat_lauto(id1,jd1,jd2,tau,slag,swid,scale)
        endif
    else if((imin.ge.2).and.(imax.ne.imin))then
        if((twidth1 .le. 0.01).or.(twidth2 .le. 0.01)) then
            covij = getcmat_lc(id1,id2,jd1,jd2,tau,slag,swid, scale)
        else
            covij = getcmat_lcross(id1,id2,jd1,jd2,tau,slag, swid,scale)
        endif
    endif
    covij = variance*covij
    return
endif

covij = sigma*sigma*covij
END FUNCTION covij


FUNCTION expcov(djd,tau)
implicit none
REAL(kind=8) :: expcov,djd,tau
expcov = exp(-abs(djd)/tau)
return
END FUNCTION expcov

FUNCTION getcmat_delta(id1,id2,jd1,jd2,tau,tspike1,tspike2,scale1,scale2)
implicit none
REAL(kind=8) :: getcmat_delta
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tspike1,tspike2,tau
REAL(kind=8) ::  scale1,scale2
getcmat_delta = exp(-abs(jd1-jd2-tspike1+tspike2)/tau)
getcmat_delta = abs(scale1*scale2)*getcmat_delta
return
END FUNCTION getcmat_delta

FUNCTION getcmat_lc(id1,id2,jd1,jd2,tau,slag,swid,scale)
implicit none
REAL(kind=8) :: getcmat_lc
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tau
REAL(kind=8) ::  twidth,emscale,tlow,thig
REAL(kind=8),DIMENSION(ncurve) ::  scale,slag,swid

if ((id1.eq.1).and.(id2.ge.2)) then
    tlow = jd2-jd1-slag(id2)-0.5*swid(id2)
    thig = jd2-jd1-slag(id2)+0.5*swid(id2)
    emscale = abs(scale(id2))
    twidth = swid(id2)
else if((id2.eq.1).and.(id1.ge.2))then
    tlow = jd1-jd2-slag(id1)-0.5*swid(id1)
    thig = jd1-jd2-slag(id1)+0.5*swid(id1)
    emscale = abs(scale(id1))
    twidth = swid(id1)
else if((id1.ge.2).and.(id2.ge.2))then
    if  (swid(id1).le.0.01)then
        tlow = jd2-(jd1-slag(id1))-slag(id2)-0.5*swid(id2)
        thig = jd2-(jd1-slag(id1))-slag(id2)+0.5*swid(id2)
        emscale = abs(scale(id2)*scale(id1))
        twidth = swid(id2)
    else if(swid(id2).le.0.01)then
        tlow = jd1-(jd2-slag(id2))-slag(id1)-0.5*swid(id1)
        thig = jd1-(jd2-slag(id2))-slag(id1)+0.5*swid(id1)
        emscale = abs(scale(id2)*scale(id1))
        twidth = swid(id1)
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

FUNCTION getcmat_lcross(id1,id2,jd1,jd2,tau,slag,swid,scale)
implicit none
REAL(kind=8) :: getcmat_lcross
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tau
REAL(kind=8),DIMENSION(ncurve:) ::  slag,swid,scale
REAL(kind=8) ::  twidth1,twidth2,bottleneck
REAL(kind=8) :: t1,t2,t3,t4,ti,tj,tlow,tmid1,tmid2,thig

twidth1 = swid(id1)
twidth2 = swid(id2)

if(twidth1.ge.twidth2) then
    t1 = slag(id1)-0.5D0*twidth1
    t2 = slag(id1)+0.5D0*twidth1
    t3 = slag(id2)-0.5D0*twidth2
    t4 = slag(id2)+0.5D0*twidth2
    bottleneck = twidth2
    ti = jd1
    tj = jd2
else
    t1 = slag(id2)-0.5D0*twidth2
    t2 = slag(id2)+0.5D0*twidth2
    t3 = slag(id1)-0.5D0*twidth1
    t4 = slag(id1)+0.5D0*twidth1
    bottleneck = twidth1
    ti = jd2
    tj = jd1
endif

tlow  = (ti-tj)-(t2-t3)
tmid1 = (ti-tj)-(t2-t4)
tmid2 = (ti-tj)-(t1-t3)
thig  = (ti-tj)-(t1-t4)

if((thig.le.0.0D0).or.(tlow.ge.0.0D0)) then
    getcmat_lcross = exp(-abs(tlow)/tau) +exp(-abs(thig)/tau)&
                    -exp(-abs(tmid1)/tau)-exp(-abs(tmid2)/tau)
else 
    getcmat_lcross = exp(tlow/tau)+exp(-thig/tau)&
                    -exp(-abs(tmid1)/tau)-exp(-abs(tmid2)/tau)
    if(tmid2.le.0.0D0) then
        getcmat_lcross = getcmat_lcross+2.0D0*thig/tau
    else if(tmid1.le.0.0D0) then
        getcmat_lcross = getcmat_lcross+2.0D0*bottleneck/tau
    else if(tlow .lt.0.0D0) then
        getcmat_lcross = getcmat_lcross-2.0D0*tlow/tau
    endif
endif

getcmat_lcross = (tau*tau*scale(id1)*scale(id2)/(twidth1*twidth2))&
                 *getcmat_lcross
RETURN
END FUNCTION getcmat_lcross

END MODULE spear_covfunc
