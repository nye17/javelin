! Last-modified: 05 Feb 2012 01:19:07 AM

MODULE spear
implicit none
    INTEGER(KIND=4) :: convmode
    INTEGER(KIND=4) :: ncurve
! TRANSFER FUNCTION TYPE
! options for what those modes are
!  DELTAFUNC    = expecting two curves but use a delta function (swid=0)
!  TOPHAT       = tophat function
!  SINGLE       = single unsmoothed light curve (to be compared to drw in javelin.gp) 
!  PHOTOECHO    = light curves from two broad bands, one with lines, the other without.
!  DOUBLETOPHAT = tophats for different emission lines or velocity bins
!  BERET        = luminosity dependent transfer function (salpha != 0)
    INTEGER(kind=4),parameter :: DELTAFUNC    = 0, &
                                 TOPHAT       = 1, &
                                 SINGLE       = 2, &
                                 PHOTOECHO    = 3, &
                                 DOUBLETOPHAT = 4, &
                                 BERET        = 5, &
                                 SINGLE_PE    = 6
contains

FUNCTION getcmatrix(id1,id2,jd1,jd2,variance,tau,slag,swid,scale,salpha)
implicit none
REAL(kind=8) :: getcmatrix
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2
REAL(kind=8) ::  variance,tau,salpha
REAL(kind=8),DIMENSION(:) ::  slag,swid,scale
REAL(kind=8) ::  twidth,twidth1,twidth2
REAL(kind=8) ::  tgap,tgap1,tgap2
INTEGER(kind=4) ::  imax,imin
!----------------------------------------------------------------------
if(convmode.eq.DELTAFUNC .or. convmode.eq.SINGLE) then
    getcmatrix = getcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale)
    getcmatrix = variance*getcmatrix
    return
endif
!----------------------------------------------------------------------
if(convmode.eq.TOPHAT) then
    twidth = swid(2)
    if((twidth .le. 0.01D0).or.(id1*id2.eq.1)) then
        getcmatrix = getcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale)
    else if(id1.ne.id2) then
        getcmatrix = getcmat_lc(id1,id2,jd1,jd2,tau,slag,swid,scale)
    else if((id1.eq.2).and.(id2.eq.2)) then
        getcmatrix = getcmat_lauto(id1,jd1,jd2,tau,slag,swid,scale)
    endif
    getcmatrix = variance*getcmatrix
    return
endif
!----------------------------------------------------------------------
if(convmode.eq.BERET ) then
    twidth = swid(2)
    if (id1*id2.eq.1) then
        getcmatrix = getcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale)
    else 
        if(twidth .le. 0.01D0) then
        getcmatrix = beretcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale,salpha)
        else if(id1.ne.id2) then
        getcmatrix = beretcmat_lc(id1,id2,jd1,jd2,tau,slag,swid,scale,salpha)
        else if((id1.eq.2).and.(id2.eq.2)) then
        getcmatrix = beretcmat_lauto(id1,jd1,jd2,tau,slag,swid,scale,salpha)
        endif
    endif
    getcmatrix = variance*getcmatrix
    return
endif
!----------------------------------------------------------------------
if(convmode.eq.DOUBLETOPHAT) then
    imax = max(id1,id2)
    imin = min(id1,id2)
    twidth1 = swid(id1)
    twidth2 = swid(id2)
    if(((twidth1.le.0.01D0).and.(twidth2.le.0.01D0)).or.(id1*id2.eq.1))then
        getcmatrix = getcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale)
    else if((imin.eq.1).and.(imax.ge.2))then
        if(swid(imax) .le. 0.01D0) then
            getcmatrix = getcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale)
        else
            getcmatrix = getcmat_lc(id1,id2,jd1,jd2,tau,slag,swid,scale)
        endif
    else if((imin.ge.2).and.(imax.eq.imin))then
        if(swid(imin) .le. 0.01) then
            getcmatrix = getcmat_delta(id1,id2,jd1,jd2,tau,slag(id1),slag(id2),scale)
        else
            getcmatrix = getcmat_lauto(id1,jd1,jd2,tau,slag,swid,scale)
        endif
    else if((imin.ge.2).and.(imax.ne.imin))then
        if((twidth1 .le. 0.01).or.(twidth2 .le. 0.01)) then
            getcmatrix = getcmat_lc(id1,id2,jd1,jd2,tau,slag,swid, scale)
        else
            getcmatrix = getcmat_lcross(id1,id2,jd1,jd2,tau,slag, swid,scale)
        endif
    endif
    getcmatrix = variance*getcmatrix
    return
endif
!----------------------------------------------------------------------
if(convmode.eq.PHOTOECHO) then
    if(id1*id2.eq.1) then
        tgap  = jd2 - jd1
        getcmatrix = expcov(tgap,tau)
    else if((id1.eq.1).and.(id2.eq.2)) then
        tgap  = jd2 - jd1
        tgap1 = tgap - slag(2)
        getcmatrix = scale(1)*expcov(tgap,tau) + scale(2)*expcov(tgap1,tau)
    else if((id1.eq.2).and.(id2.eq.1)) then
        tgap  = jd1 - jd2
        tgap1 = tgap - slag(2)
        getcmatrix = scale(1)*expcov(tgap,tau) + scale(2)*expcov(tgap1,tau)
    else if((id1.eq.2).and.(id2.eq.2)) then
        tgap  = jd2 - jd1
        tgap1 = tgap - slag(2)
        tgap2 = tgap + slag(2)
        getcmatrix = (scale(1)*scale(1)+scale(2)*scale(2))*expcov(tgap,tau) &
                   + scale(1)*scale(2)*expcov(tgap1,tau) &
                   + scale(1)*scale(2)*expcov(tgap2,tau)
    endif
    getcmatrix = variance*getcmatrix
    return
endif
END
!***********************************************************************

!***********************************************************************
! FUN_EXPCOV
!***********************************************************************
function expcov(djd,tau)
implicit none
REAL(kind=8) :: expcov,djd,tau
expcov = exp(-abs(djd)/tau)
return
end
!***********************************************************************

!***********************************************************************
! FUN_GETCMAT_DELTA
!***********************************************************************
function getcmat_delta(id1,id2,jd1,jd2,tau,tspike1,tspike2,scale)
implicit none
!***********************************************************************
REAL(kind=8) :: getcmat_delta
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tspike1,tspike2,tau
REAL(kind=8),DIMENSION(:) ::  scale
getcmat_delta = exp(-abs(jd1-jd2-tspike1+tspike2)/tau)
getcmat_delta = abs(scale(id1)*scale(id2))*getcmat_delta
return
end
!***********************************************************************

!***********************************************************************
! FUN_BERETCMAT_DELTA
!***********************************************************************
function beretcmat_delta(id1,id2,jd1,jd2,tau,tspike1,tspike2,scale,salpha)
implicit none
!***********************************************************************
REAL(kind=8) :: beretcmat_delta
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tspike1,tspike2,tau,salpha
REAL(kind=8) ::  tspike11,tspike22
REAL(kind=8),DIMENSION(:) ::  scale
external jd2lag
tspike11 = tspike1
tspike22 = tspike2
if((tspike1.ne.0.0) .and. (tspike1.le.100))then
    call jd2lag(tspike11,jd1,scale(id1),salpha,tspike1)
endif

if((tspike2.ne.0.0) .and. (tspike2.le.100))then
    call jd2lag(tspike22,jd2,scale(id2),salpha,tspike2)
endif
beretcmat_delta = exp(-abs(jd1-jd2-tspike11+tspike22)/tau)
beretcmat_delta = abs(scale(id1)*scale(id2))*beretcmat_delta
return
end
!***********************************************************************

!***********************************************************************
! FUN_GETCMAT_LC
!***********************************************************************
function getcmat_lc(id1,id2,jd1,jd2,tau,slag,swid,scale)
implicit none
!***********************************************************************
REAL(kind=8) :: getcmat_lc
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tau
REAL(kind=8) ::  twidth,emscale,tlow,thig
REAL(kind=8),DIMENSION(:) ::  scale,slag,swid

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
end
!***********************************************************************

!***********************************************************************
! FUN_BERETCMAT_LC
!***********************************************************************
function beretcmat_lc(id1,id2,jd1,jd2,tau,slag,swid,scale,salpha)
implicit none
!***********************************************************************
REAL(kind=8) ::  beretcmat_lc
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tau,salpha
REAL(kind=8),DIMENSION(:) ::  slag,swid,scale
REAL(kind=8) ::  twidth,emscale,lag,tlow,thig
external jd2lag

if ((id1.eq.1).and.(id2.ge.2)) then
    call jd2lag(lag,jd2,scale(id2),salpha,slag(id2))
    tlow = jd2-jd1-lag-0.5D0*swid(id2)
    thig = jd2-jd1-lag+0.5D0*swid(id2)
    emscale = abs(scale(id2))
    twidth = swid(id2)
else if((id2.eq.1).and.(id1.ge.2))then
    call jd2lag(lag,jd1,scale(id1),salpha,slag(id1))
    tlow = jd1-jd2-lag-0.5D0*swid(id1)
    thig = jd1-jd2-lag+0.5D0*swid(id1)
    emscale = abs(scale(id1))
    twidth = swid(id1)
else
    print*,'No Cont. Line in GETCMAT_LC!'
    print*,'id1 ',id1,'id2',id2
    stop
endif

if (thig.le.0.0D0) then
    beretcmat_lc = exp( thig/tau)-exp( tlow/tau)
else if (tlow.ge.0.0D0) then
    beretcmat_lc = exp(-tlow/tau)-exp(-thig/tau)
else 
    beretcmat_lc = 2.0D0-exp(tlow/tau)-exp(-thig/tau)
endif

beretcmat_lc = tau*(emscale/twidth)*beretcmat_lc
return
end
!***********************************************************************

!***********************************************************************
! FUN_GETCMAT_LAUTO
!***********************************************************************
function getcmat_lauto(id,jd1,jd2,tau,slag,swid,scale)
implicit none
!***********************************************************************
REAL(kind=8) :: getcmat_lauto
INTEGER(kind=4) :: id
REAL(kind=8) :: jd1,jd2,tau
REAL(kind=8),DIMENSION(:) ::  slag,swid,scale
REAL(kind=8) :: twidth,emscale,tlow,tmid,thig

twidth = swid(id)
emscale = scale(id)

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
end
!***********************************************************************

!***********************************************************************
! FUN_BERETCMAT_LAUTO
!***********************************************************************
function beretcmat_lauto(id,jd1,jd2,tau,slag,swid,scale,salpha)
implicit none
!***********************************************************************
REAL(kind=8) :: beretcmat_lauto
INTEGER(kind=4) ::  id
REAL(kind=8) ::  jd1,jd2,tau,salpha
REAL(kind=8),DIMENSION(:) ::  slag,swid,scale
REAL(kind=8) ::  twidth,dlag,lag1,lag2,ti,tj,tlow,tmid,thig
external jd2lag

twidth = swid(id)

call jd2lag(lag1,jd1,scale(id),salpha,slag(id))
call jd2lag(lag2,jd2,scale(id),salpha,slag(id))

dlag = lag1-lag2
ti = jd1
tj = jd2

tlow  = (ti-tj)-(dlag+twidth)
tmid  = (ti-tj)- dlag
thig  = (ti-tj)-(dlag-twidth)

if((thig.le.0.0D0).or.(tlow.ge.0.0D0)) then
    beretcmat_lauto = exp(-abs(tlow)/tau) +exp(-abs(thig)/tau)-2.D0*exp(-abs(tmid)/tau)
else 
    beretcmat_lauto = exp(tlow/tau)+exp(-thig/tau)-2.D0*exp(-abs(tmid)/tau)
    if(tmid.le.0.0D0) then
        beretcmat_lauto = beretcmat_lauto+2.0D0*thig/tau
    else if(tmid .gt.0.0D0) then
        beretcmat_lauto = beretcmat_lauto-2.0D0*tlow/tau
    endif
endif

beretcmat_lauto = (tau*tau*scale(id)*scale(id)/(twidth*twidth))*beretcmat_lauto
return
end
!***********************************************************************


!***********************************************************************
!     FUN_GETCMAT_LCROSS
!***********************************************************************
function getcmat_lcross(id1,id2,jd1,jd2,tau,slag,swid,scale)
implicit none
!***********************************************************************
REAL(kind=8) :: getcmat_lcross
INTEGER(kind=4) ::  id1,id2
REAL(kind=8) ::  jd1,jd2,tau
REAL(kind=8),DIMENSION(:) ::  slag,swid,scale
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
!----------------------------------------------------------------------
RETURN
!----------------------------------------------------------------------
END
!**********************************************************************

END MODULE spear
