!***********************************************************
!     backscatter_hh_retrieval
!
!> @brief Implementation of 'HH' polarised backscatter target retrieval processor
!
!> @param[in]   n     length of control vector
!> @param[in]   x     control vector normalised by prior uncertainty
!                     (expected ordering is: S1 related parameter(s) followed by
!                     by state variables LAI,HC,SM per 'simulation point')
!> @param[in]   xcov  covariance matrix of control vector ('Cx.b')
!                     (in normalised control  vector units)
!
!> \authors MV/TK, The Inversion Lab
!> \date    February 2018
!
subroutine backscatter_hh_retrieval(n, x, xcov)
  use mo_sensimul
  use mo_util, only:print_matrix
  implicit none
  ! arguments
  integer, intent(in)      :: n          !< length of control vector
  real(kind=8), intent(in) :: x(n)       !< control vector (normalised, no physical units)
  real(kind=8), intent(in) :: xcov(n,n)  !< posterior covariance matrix
  ! local decls
  real(kind=8) :: x0(n), sxpr(n)
  real(kind=8), allocatable :: bscathh_pr(:), bscathhunc_pr(:), bscathh_po(:), bscathhunc_po(:)
  real(kind=8), allocatable :: tjac(:,:) !-- HH-backscatter Jacobian
  real(kind=8), allocatable :: ct_pr(:,:), ct_po(:,:)
  logical :: succeed
  integer :: i,j,k,l,m
  character(len=256) :: target_fname

  !-- get the prior
  call initx(n, x0, sxpr)

  !-- number of time-points
  !   (however, HH-backscatter will only be simulated at S1 acquisition times)
  m = get_npts()

  !--
  allocate( bscathh_pr(m), bscathh_po(m), bscathhunc_pr(m), bscathhunc_po(m) )
  allocate( tjac(m,n), ct_pr(m,m), ct_po(m,m) )

  !-- HH-backscatter: ***prior,posterior***
  write(*, '(a)') ' INFO::backscatter_hh_retrieval:'//&
       'start computation of HH-backscatter at prior...'
  call simulate_bscat_hh(n, x0, m, bscathh_pr)
  write(*, '(a)') ' INFO::backscatter_hh_retrieval:'//&
       '...DONE'
  write(*, '(a)') ' INFO::backscatter_hh_retrieval:'//&
       'start computation of HH-backscatter at x...'
  call simulate_bscat_hh(n, x, m, bscathh_po)
  write(*, '(a)') ' INFO::backscatter_hh_retrieval:'//&
       '...DONE'

  !-- HH-backscatter Jacobian
  write(*, '(a)') ' INFO::backscatter_hh_retrieval:'//&
       'start computation of HH-backscatter Jacobian at x...'
  call bscat_hh_jacobian(m, n, x, tjac)
  write(*, '(a)') ' INFO::backscatter_hh_retrieval:'//&
       '...DONE'

  !-- uncertainty propagation for prior/post HH-backscatter
  !-- compute: T'*Cx*T'^t (D5v2, Eq. 1.11)
  ! NOTE::since we are operating 'x' in normalised (i.e. scaled by prior sigma)
  !       coordinates, Ct_pr is just the identity
  do j=1,m
     do i=1,m
        ct_pr(i,j) = 0._8
        ct_po(i,j)  = 0._8
        do k=1,n
           !-- tjac^t(k,j) = tjac(j,k)
           ct_pr(i,j) = ct_pr(i,j) + tjac(i,k)*tjac(j,k)
           do l=1,n
              !-- tjac^t(l,j) = tjac(j,l)
              ct_po(i,j) = ct_po(i,j) + tjac(i,k)*xcov(k,l)*tjac(j,l)
           enddo
        enddo
     enddo
  enddo

  !-- extract uncertainties only (diagonal)
  !   but only when HH-backscatter was simulated
  bscathhunc_pr = sim_fill_value
  bscathhunc_po = sim_fill_value
  do i=1,m
     if( bscathh_pr(i).ne.sim_fill_value ) then
        bscathhunc_pr(i)  = sqrt(ct_pr(i,i))
        bscathhunc_po(i)  = sqrt(ct_po(i,i))
     endif
  enddo
     
  !-- write prior HH-backscatter
  target_fname = 'backscatter_hh_prior.nc'
  call ncwrt_target_vector(target_fname, 'backscatter_hh', m, bscathh_pr, bscathhunc_pr, succeed)
  if( succeed ) then
     write(*, '(a)') ' INFO::backscatter_hh_retrieval:'//&
          'generated target data file ***'//trim(target_fname)//'***'
  else
     write(*, '(a)') ' ERROR::backscatter_hh_retrieval:'//&
          'error occurred when writing target data file ***'//trim(target_fname)//'***'
  endif

  !-- write post HH-backscatter
  target_fname = 'backscatter_hh_post.nc'
  call ncwrt_target_vector(target_fname, 'backscatter_hh', m, bscathh_po, bscathhunc_po, succeed)
  if( succeed ) then
     write(*, '(a)') ' INFO::backscatter_hh_retrieval:'//&
          'generated target data file ***'//trim(target_fname)//'***'
  else
     write(*, '(a)') ' ERROR::backscatter_hh_retrieval:'//&
          'error occurred when writing target data file ***'//trim(target_fname)//'***'
  endif

  !-- dispose memory
  deallocate( bscathh_pr, bscathh_po, bscathhunc_pr, bscathhunc_po )
  deallocate( tjac, ct_pr, ct_po )

end subroutine backscatter_hh_retrieval
